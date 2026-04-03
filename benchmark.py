"""
PyTorch torch.compile phase timing benchmark.

Measures compilation time broken down by phase (Dynamo / AOT / Inductor) for
each sample model, then writes the results to compile_times.csv.

Usage
-----
    python benchmark.py [--output compile_times.csv]   # device auto-detected
    python benchmark.py --device cpu                   # override device

TORCH_LOGS
----------
The script sets  TORCH_LOGS="dynamo"  before importing torch so that PyTorch
emits timing log lines for the Dynamo phase.  These logs are captured per-sample
and written alongside the timing summary.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable

# ── TORCH_LOGS must be set BEFORE importing torch ──────────────────────────
os.environ.setdefault("TORCH_LOGS", "dynamo")

import torch
import torch._dynamo
import torch._dynamo.utils as _dynamo_utils

# ── Disable compilation caches so every sample is compiled fresh ────────────
# Without this, AOTAutogradCache / FX graph cache can produce cache hits that
# hide the real inductor compilation time.
try:
    import torch._functorch.config as _functorch_cfg
    _functorch_cfg.enable_autograd_cache = False
except Exception:
    pass
try:
    import torch._inductor.config as _inductor_cfg
    _inductor_cfg.fx_graph_cache = False
    _inductor_cfg.fx_graph_remote_cache = False
except Exception:
    pass

# ── per-sample cache reset ──────────────────────────────────────────────────

def _reset_all_caches() -> None:
    """
    Fully reset every compilation cache layer before each sample run so that
    timings reflect a cold compilation, not a cache hit.

    Layers cleared:
      1. torch._dynamo.reset()
           – Dynamo in-memory frame/code cache
           – compilation_time_metrics (timing accumulators)
      2. PyCodeCache.cache
           – Inductor in-memory map of {hash -> loaded compiled module}
           – Without this, identical or similar kernels compiled by earlier
             samples are reused silently, zeroing inductor_compile_s.
      3. FxGraphCache (in-memory component)
           – Inductor's in-memory FX graph cache; disk I/O is already
             disabled via fx_graph_cache=False but the in-memory dict
             can still serve hits within a process.
      4. AOTAutogradCache (in-memory component)
           – Same rationale as FxGraphCache.
    """
    # Layer 1 – Dynamo frame cache + timing metrics
    torch._dynamo.reset()

    # Layer 2 – Inductor PyCodeCache (compiled .so / Python modules)
    try:
        import torch._inductor.codecache as _cc
        if hasattr(_cc, "PyCodeCache") and hasattr(_cc.PyCodeCache, "cache"):
            _cc.PyCodeCache.cache.clear()
    except Exception:
        pass

    # Layer 3 – Inductor FxGraphCache in-memory store
    try:
        import torch._inductor.codecache as _cc
        if hasattr(_cc, "FxGraphCache") and hasattr(_cc.FxGraphCache, "_cache"):
            _cc.FxGraphCache._cache.clear()
    except Exception:
        pass

    # Layer 4 – AOTAutogradCache in-memory store
    try:
        import torch._inductor.codecache as _cc
        if hasattr(_cc, "AOTAutogradCache") and hasattr(_cc.AOTAutogradCache, "_cache"):
            _cc.AOTAutogradCache._cache.clear()
    except Exception:
        pass


# ── device detection ─────────────────────────────────────────────────────────

def _detect_device() -> str:
    """
    Return the best available compute device.

    Priority: MLU > CUDA > CPU.
    """
    try:
        if torch.mlu.is_available():
            return "mlu"
    except AttributeError:
        pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _sync_device(device: str) -> None:
    """Synchronize the given device; no-op on CPU."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mlu":
        try:
            torch.mlu.synchronize()
        except AttributeError:
            pass


# ── warmup ───────────────────────────────────────────────────────────────────

def _warmup(device: str) -> None:
    """
    Run one throwaway compilation to trigger all one-time initializations
    (Inductor/Triton infrastructure JIT, pass-pipeline setup, OS page cache)
    before any timed case starts.

    Without this, the first measured case pays the full cold-start cost,
    making single-case runs appear slower than the same case run inside a
    full batch (where the cost is absorbed by earlier cases).

    A _reset_all_caches() call after warmup ensures the timed cases still
    start from a clean compilation state.

    Log output is suppressed during warmup so TORCH_LOGS noise doesn't
    pollute the benchmark output.
    """
    import torch.nn as nn
    _reset_all_caches()
    # Suppress all logging during the warmup compilation
    logging.disable(logging.CRITICAL)
    try:
        m = torch.compile(nn.Linear(4, 4).to(device))
        with torch.no_grad():
            m(torch.randn(4, 4, device=device))
    finally:
        logging.disable(logging.NOTSET)
    _reset_all_caches()


# ── logging setup ───────────────────────────────────────────────────────────
_LOGFMT = "%(name)s | %(levelname)s | %(message)s"


def _attach_capture_handler(buf: io.StringIO) -> list[logging.Handler]:
    """Attach a StringIO-backed handler to the dynamo logger."""
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter(_LOGFMT))
    lg = logging.getLogger("torch._dynamo")
    lg.addHandler(handler)
    lg.setLevel(logging.INFO)
    return [(lg, handler)]


def _detach_capture_handler(handlers: list) -> None:
    for lg, h in handlers:
        lg.removeHandler(h)


# ── timing helpers ───────────────────────────────────────────────────────────

def _read_compile_times() -> dict[str, float]:
    """
    Read per-phase compile times from torch._dynamo internals.

    Phase mapping (derived from compilation_time_metrics hierarchy):
        dynamo_s        – Python bytecode tracing + guard building
        aot_s           – AOT Autograd (joint graph lowering, metadata collection)
        inductor_s      – Inductor codegen + kernel compilation
        total_compile_s – Full compilation attempt (_compile.compile_inner)

    Additional sub-phase keys (seconds):
        inductor_codegen_s  – GraphLowering.codegen
        inductor_compile_s  – compile_file (kernel compilation)
        inductor_load_s     – PyCodeCache.load_by_key_path
        pre_grad_passes_s   – _recursive_pre_grad_passes
        post_grad_passes_s  – _recursive_post_grad_passes
        joint_graph_passes_s – _recursive_joint_graph_passes
        cache_hit           – 1 if AOTAutogradCache/FX cache was hit, else 0
    """
    metrics: dict[str, list[float]] = getattr(
        _dynamo_utils, "compilation_time_metrics", {}
    )

    def _sum(key: str) -> float:
        return sum(metrics.get(key, []))

    # Dynamo: bytecode tracing + guard compilation
    dynamo_s = _sum("bytecode_tracing") + _sum("build_guards")

    # AOT Autograd: dispatcher function setup minus the inductor backend call
    aot_s = max(
        _sum("create_aot_dispatcher_function")
        - _sum("compile_fx.<locals>.fw_compiler_base"),
        0.0,
    )
    # Fallback: aot_collect_metadata as a proxy when create_aot_dispatcher_function absent
    if aot_s == 0.0:
        aot_s = _sum("aot_collect_metadata")

    # Inductor: everything inside compile_fx_inner
    inductor_s = _sum("compile_fx_inner")

    # Sub-phases
    inductor_codegen_s = _sum("GraphLowering.codegen")
    inductor_compile_s = _sum("compile_file")
    inductor_load_s = _sum("PyCodeCache.load_by_key_path")
    pre_grad_s = _sum("_recursive_pre_grad_passes")
    post_grad_s = _sum("_recursive_post_grad_passes")
    joint_graph_s = _sum("_recursive_joint_graph_passes")

    # Cache-hit detection: AOTAutogradCache or FX cache was used
    cache_hit = 1 if _sum("AOTAutogradCache.inductor_load") > 0 or _sum("FXGraphCache.load") > 0 else 0

    # Total: the outermost compile_inner timer
    total_compile_s = _sum("_compile.compile_inner")

    return {
        "dynamo_s": round(dynamo_s, 6),
        "aot_s": round(aot_s, 6),
        "backend_s": round(inductor_s, 6),
        "total_compile_s": round(total_compile_s, 6),
        "inductor_codegen_s": round(inductor_codegen_s, 6),
        "inductor_compile_s": round(inductor_compile_s, 6),
        "inductor_load_s": round(inductor_load_s, 6),
        "pre_grad_passes_s": round(pre_grad_s, 6),
        "post_grad_passes_s": round(post_grad_s, 6),
        "joint_graph_passes_s": round(joint_graph_s, 6),
        "cache_hit": cache_hit,
    }


# ── result dataclass ─────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    sample: str = ""
    device: str = ""
    # wall-clock time for the *first* forward pass (includes compilation)
    first_call_s: float = 0.0
    # wall-clock time for the *second* forward pass (compiled artifact reused)
    second_call_s: float = 0.0
    # ── high-level compile-phase breakdown ──────────────────────────────────
    dynamo_s: float = 0.0           # bytecode tracing + guard building
    aot_s: float = 0.0              # AOT Autograd (joint graph lowering)
    backend_s: float = 0.0          # Inductor: full codegen + compilation
    total_compile_s: float = 0.0    # _compile.compile_inner (wall total)
    # ── Inductor sub-phases ─────────────────────────────────────────────────
    inductor_codegen_s: float = 0.0    # GraphLowering.codegen
    inductor_compile_s: float = 0.0    # compile_file (kernel compilation)
    inductor_load_s: float = 0.0       # PyCodeCache.load_by_key_path
    # ── graph-pass timings ──────────────────────────────────────────────────
    pre_grad_passes_s: float = 0.0
    post_grad_passes_s: float = 0.0
    joint_graph_passes_s: float = 0.0
    # ── counters ────────────────────────────────────────────────────────────
    cache_hit: int = 0          # 1 if AOT/FX cache was hit (skipped inductor)
    graph_breaks: int = 0
    frames_compiled: int = 0
    # ── hardware kernel execution time (milliseconds) ───────────────────────
    #   cuda – torch.cuda.Event.elapsed_time() (GPU-side clock, ms)
    #   mlu  – torch.mlu.Event.elapsed_time()  (device-side clock, ms)
    #   cpu  – wall clock converted to ms
    kernel_time_ms: float = 0.0
    # hardware execution time of the same op in eager (non-compiled) mode (ms)
    eager_time_ms: float = 0.0
    # ── accuracy: compiled vs eager (0.0 when not computed) ─────────────────
    # Outputs are cast to fp32 before comparison to avoid fp16 overflow.
    max_abs_err:  float = 0.0   # max element-wise absolute error
    mean_abs_err: float = 0.0   # mean element-wise absolute error
    max_rel_err:  float = 0.0   # max element-wise relative error
    cosine_sim:   float = 0.0   # cosine similarity (1.0 = identical)
    # error message if compilation failed
    error: str = ""

    @classmethod
    def csv_header(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    def csv_row(self) -> list:
        return [getattr(self, f.name) for f in fields(self)]


# ── accuracy helpers ─────────────────────────────────────────────────────────

def _compute_accuracy(
    eager_out: torch.Tensor | tuple,
    compiled_out: torch.Tensor | tuple,
) -> tuple[float, float, float, float]:
    """
    Compare compiled output against eager reference.

    Tensors are cast to fp32 before comparison so that fp16 magnitude
    differences don't confuse the relative-error denominator.

    Returns (max_abs_err, mean_abs_err, max_rel_err, cosine_sim).
    """
    def _to_flat_fp32(out):
        if isinstance(out, (tuple, list)):
            return torch.cat([t.detach().float().flatten() for t in out])
        return out.detach().float().flatten()

    ref = _to_flat_fp32(eager_out)
    cmp = _to_flat_fp32(compiled_out)

    abs_err = (ref - cmp).abs()
    max_abs  = abs_err.max().item()
    mean_abs = abs_err.mean().item()

    rel_err  = (abs_err / ref.abs().clamp(min=1e-8)).max().item()
    cos_sim  = torch.nn.functional.cosine_similarity(
        ref.unsqueeze(0), cmp.unsqueeze(0)
    ).item()

    return (
        round(max_abs,  8),
        round(mean_abs, 8),
        round(rel_err,  8),
        round(cos_sim,  8),
    )


# ── per-sample benchmark ─────────────────────────────────────────────────────

def _run_sample(
    name: str,
    get_model_and_input: Callable,
    device: str,
    backend: str = "inductor",
    compute_accuracy: bool = False,
) -> tuple[BenchResult, str]:
    """Compile and run one sample; return (BenchResult, captured_logs)."""
    result = BenchResult(sample=name, device=device)

    # Full cache reset: Dynamo frame cache, Inductor PyCodeCache,
    # FxGraphCache, AOTAutogradCache – all layers cleared before each run.
    _reset_all_caches()

    log_buf = io.StringIO()
    handlers = _attach_capture_handler(log_buf)

    try:
        model, inputs = get_model_and_input(device=device)
        model.eval()

        compiled = torch.compile(model, backend=backend, fullgraph=False)

        # ── first call: triggers full compilation ──
        _sync_device(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            compiled(*inputs)
        _sync_device(device)
        result.first_call_s = time.perf_counter() - t0

        # ── second call: compiled artifact is reused ──────────────────────
        _sync_device(device)
        t1 = time.perf_counter()
        with torch.no_grad():
            compiled(*inputs)
        _sync_device(device)
        result.second_call_s = time.perf_counter() - t1
        # kernel_time_ms is left as 0.0 here; filled in by _collect_kernel_times
        # after the main benchmark pass so that parallel runs don't cause
        # concurrent kernel execution that skews hardware timing.

        # ── collect compile-phase timings ──
        ct = _read_compile_times()
        result.dynamo_s = ct["dynamo_s"]
        result.aot_s = ct["aot_s"]
        result.backend_s = ct["backend_s"]
        result.total_compile_s = ct["total_compile_s"]
        result.inductor_codegen_s = ct["inductor_codegen_s"]
        result.inductor_compile_s = ct["inductor_compile_s"]
        result.inductor_load_s = ct["inductor_load_s"]
        result.pre_grad_passes_s = ct["pre_grad_passes_s"]
        result.post_grad_passes_s = ct["post_grad_passes_s"]
        result.joint_graph_passes_s = ct["joint_graph_passes_s"]
        result.cache_hit = ct["cache_hit"]

        # ── accuracy: compiled vs eager ──────────────────────────────────────
        if compute_accuracy:
            with torch.no_grad():
                eager_out    = model(*inputs)
                compiled_out = compiled(*inputs)
            (result.max_abs_err,
             result.mean_abs_err,
             result.max_rel_err,
             result.cosine_sim) = _compute_accuracy(eager_out, compiled_out)

        # ── frame / graph-break counters ──
        metrics = getattr(_dynamo_utils, "compilation_time_metrics", {})
        result.frames_compiled = len(metrics.get("_compile.compile_inner", []))

        logs = log_buf.getvalue()
        result.graph_breaks = logs.count("Graph break")

    except Exception as exc:  # noqa: BLE001
        result.error = str(exc)
        logs = log_buf.getvalue()
    finally:
        _detach_capture_handler(handlers)

    return result, logs


# ── subprocess worker (module-level so it is picklable) ──────────────────────

def _pin_worker_to_cores(slot: int, n_workers: int) -> None:
    """
    Pin the current process to a dedicated subset of available CPU cores.

    Divides the set of CPUs visible to this process evenly among workers.
    Each worker slot gets an exclusive slice so that workers do not share
    physical cores, which reduces scheduler preemption and L1/L2 cache
    thrashing during compilation.

    No-op on systems without sched_setaffinity (macOS, Windows, sandboxed
    environments) — falls back silently.
    """
    try:
        available = sorted(os.sched_getaffinity(0))
        if len(available) <= 1:
            return
        cores_per_worker = max(1, len(available) // n_workers)
        start = (slot % n_workers) * cores_per_worker
        assigned = set(available[start: start + cores_per_worker])
        os.sched_setaffinity(0, assigned)
    except (AttributeError, OSError):
        pass


def _worker_init(slot_counter: object, n_workers: int) -> None:
    """
    ProcessPoolExecutor initializer — runs once in each worker process.

    Atomically claims the next available slot from the shared counter and
    pins the process to the corresponding CPU core slice.
    """
    with slot_counter.get_lock():  # type: ignore[attr-defined]
        slot = slot_counter.value
        slot_counter.value += 1
    _pin_worker_to_cores(slot, n_workers)


def _worker(task: tuple) -> tuple[int, dict, str]:
    """
    Entry point for each ProcessPoolExecutor worker.

    task = (idx, variant_name, module_path, fn_kwargs, device, backend)

    All elements are plain picklable values.  The worker imports the sample
    module (which is a no-op after fork since it is already in sys.modules),
    rebuilds the get_model_and_input callable from fn_kwargs, and delegates
    to _run_sample.  Returns (idx, result_as_dict, captured_logs) so the
    main process can reconstruct a BenchResult and maintain submission order.
    """
    idx, variant_name, module_path, fn_kwargs, device, backend, compute_accuracy = task

    import importlib
    mod = importlib.import_module(module_path)

    def get_fn(device: str = "cpu"):
        return mod.get_model_and_input(**fn_kwargs, device=device)

    result, logs = _run_sample(variant_name, get_fn, device, backend, compute_accuracy)
    return idx, asdict(result), logs


# ── sequential kernel timing pass ────────────────────────────────────────────

def _collect_kernel_times(
    all_tasks: list[tuple],
    device: str,
    backend: str,
) -> tuple[dict[int, float], dict[int, float]]:
    """
    Re-run every case sequentially in the main process and measure hardware
    kernel execution time for both compiled and eager (non-compiled) modes
    using device-side event clocks.

    This is intentionally separate from the main benchmark pass so that
    parallel workers (--workers > 1) do not execute kernels concurrently,
    which would skew device-side timing.

    Returns (kernel_times, eager_times) — both indexed by task idx (ms).
    TORCH_LOGS output is suppressed to keep the output clean.
    """
    import importlib

    def _time_fn(fn, inputs, device):
        """
        Return average execution time of fn(*inputs) in ms over 10 steps.

        For CPU: wall-clock average over 10 runs.
        For CUDA/MLU: torch.profiler with schedule(wait=2, warmup=2, active=10),
        device_time_us averaged across the 10 active steps.
        """
        _WAIT    = 2
        _WARMUP  = 2
        _ACTIVE  = 10
        _TOTAL   = _WAIT + _WARMUP + _ACTIVE

        if device == "cpu":
            # run TOTAL times; average last ACTIVE
            for _ in range(_WAIT + _WARMUP):
                with torch.no_grad():
                    fn(*inputs)
            t = time.perf_counter()
            for _ in range(_ACTIVE):
                with torch.no_grad():
                    fn(*inputs)
            return round((time.perf_counter() - t) * 1000.0 / _ACTIVE, 6)

        activities = [torch.profiler.ProfilerActivity.CPU]
        if device == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        elif device == "mlu":
            try:
                activities.append(torch.profiler.ProfilerActivity.MLU)
            except AttributeError:
                pass

        with torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=_WAIT, warmup=_WARMUP, active=_ACTIVE, repeat=1,
            ),
            record_shapes=False,
            with_flops=False,
        ) as prof:
            for _ in range(_TOTAL):
                with torch.no_grad():
                    fn(*inputs)
                prof.step()

        # Sum self device time across active steps (µs), average per step → ms
        total_us = sum(
            e.self_device_time_total for e in prof.key_averages()
            if e.self_device_time_total > 0
        )
        if total_us > 0:
            return round(total_us / 1000.0 / _ACTIVE, 6)
        # Fallback: self CPU time if no device events recorded
        cpu_us = sum(e.self_cpu_time_total for e in prof.key_averages())
        return round(cpu_us / 1000.0 / _ACTIVE, 6)

    n = len(all_tasks)
    if n > 1:
        print(f"\n[kernel timing] {n} cases — compiled vs eager (sequential) …", flush=True)
    kernel_times: dict[int, float] = {}
    eager_times: dict[int, float] = {}

    for i, task in enumerate(all_tasks):
        idx, variant_name, module_path, fn_kwargs, device_, backend_, _ = task
        logging.disable(logging.CRITICAL)
        try:
            mod = importlib.import_module(module_path)
            model, inputs = mod.get_model_and_input(**fn_kwargs, device=device)
            model.eval()
            compiled = torch.compile(model, backend=backend, fullgraph=False)

            # warmup compiled (triggers compilation)
            with torch.no_grad():
                compiled(*inputs)
            # warmup eager
            with torch.no_grad():
                model(*inputs)

            kernel_times[idx] = _time_fn(compiled, inputs, device)
            eager_times[idx]   = _time_fn(model,    inputs, device)
        except Exception:
            kernel_times[idx] = 0.0
            eager_times[idx]  = 0.0
        finally:
            logging.disable(logging.NOTSET)

        if n > 1 and (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{n}]", flush=True)

    return kernel_times, eager_times


# ── statistics helpers ────────────────────────────────────────────────────────

# Fields excluded from statistical summary (non-numeric or per-case identifiers)
_STATS_EXCLUDE = {"sample", "device", "error"}


_STAT_GROUPS: list[tuple[str, list[str]]] = [
    ("Performance (ms)", [
        "kernel_time_ms",
        "eager_time_ms",
        # speedup injected programmatically
    ]),
    ("Compilation (s)", [
        "total_compile_s",
        "dynamo_s",
        "aot_s",
        "backend_s",
        "inductor_codegen_s",
        "inductor_compile_s",
        "inductor_load_s",
    ]),
    ("Graph Passes (s)", [
        "pre_grad_passes_s",
        "post_grad_passes_s",
        "joint_graph_passes_s",
    ]),
    ("Wall-clock (s)", [
        "first_call_s",
        "second_call_s",
    ]),
    ("Counters", [
        "cache_hit",
        "graph_breaks",
        "frames_compiled",
    ]),
    ("Accuracy (compiled vs eager)", [
        "max_abs_err",
        "mean_abs_err",
        "max_rel_err",
        "cosine_sim",
    ]),
]


def _compute_stats(good: list[BenchResult]) -> dict[str, dict[str, float]]:
    """Compute max/min/avg for every numeric field plus speedup."""
    numeric_fields = [
        f.name for f in fields(BenchResult)
        if f.name not in _STATS_EXCLUDE and f.type in ("float", "int")
    ]
    if not numeric_fields:
        numeric_fields = [
            f.name for f in fields(BenchResult)
            if f.name not in _STATS_EXCLUDE
            and isinstance(getattr(good[0], f.name), (int, float))
        ]
    stats: dict[str, dict[str, float]] = {}
    for col in numeric_fields:
        vals = [getattr(r, col) for r in good]
        stats[col] = {"max": max(vals), "min": min(vals), "avg": sum(vals) / len(vals)}
    speedup_vals = [
        r.eager_time_ms / r.kernel_time_ms
        for r in good
        if r.kernel_time_ms > 0 and r.eager_time_ms > 0
    ]
    if speedup_vals:
        avg_kernel = stats.get("kernel_time_ms", {}).get("avg", 0.0)
        avg_eager  = stats.get("eager_time_ms",  {}).get("avg", 0.0)
        avg_speedup = (avg_eager / avg_kernel) if avg_kernel > 0 else 0.0
        stats["speedup (eager/compiled)"] = {
            "max": max(speedup_vals),
            "min": min(speedup_vals),
            "avg": avg_speedup,
        }
    return stats


def _render_stats_section(
    stats: dict[str, dict[str, float]],
    section_title: str,
    n_good: int,
    n_total: int,
) -> str:
    """Render one grouped stats block (all _STAT_GROUPS) as a string."""
    col_w = 16
    label_w = 32
    total_w = label_w + col_w * 3 + 9
    buf = io.StringIO()

    def _emit(line: str = "") -> None:
        buf.write(line + "\n")

    sep = "━" * total_w
    _emit(f"\n{sep}")
    _emit(f"  {section_title}  (n={n_good})")
    _emit(sep)

    def _hdr(title: str) -> None:
        _emit(f"\n╔{'═' * (total_w - 2)}╗")
        _emit(f"║  {title:<{total_w - 4}}║")
        _emit(f"╠{'═' * (total_w - 2)}╣")
        _emit(f"║  {'':>{label_w - 2}}  {'max':>{col_w}}  {'min':>{col_w}}  {'avg':>{col_w}} ║")
        _emit(f"╠{'─' * (total_w - 2)}╣")

    def _row(label: str, col: str, is_float: bool = True) -> None:
        if col not in stats:
            return
        s = stats[col]
        fmt = f"{{:>{col_w}.6f}}" if is_float else f"{{:>{col_w}.0f}}"
        _emit(f"║  {label:<{label_w - 2}}  {fmt.format(s['max'])}  {fmt.format(s['min'])}  {fmt.format(s['avg'])} ║")

    def _footer() -> None:
        _emit(f"╚{'═' * (total_w - 2)}╝")

    has_accuracy = any(col in stats and stats[col]["max"] != 0.0
                       for col in ("cosine_sim", "max_abs_err"))

    for group_title, cols in _STAT_GROUPS:
        if group_title.startswith("Accuracy") and not has_accuracy:
            continue
        _hdr(group_title)
        for col in cols:
            if col not in stats:
                continue
            _row(col, col, isinstance(stats[col]["avg"], float))
            if col == "eager_time_ms" and "speedup (eager/compiled)" in stats:
                _row("speedup (eager/compiled)", "speedup (eager/compiled)", True)
        _footer()

    if n_total > n_good:
        _emit(f"\n  {n_total - n_good} error(s) excluded from statistics")

    return buf.getvalue()


def _write_stats(
    results: list[BenchResult],
    out_path: Path,
    sample_to_type: dict[str, str] | None = None,
) -> None:
    """
    Compute statistics and write to <stem>_summary.txt.
    Stats are grouped by case_type when sample_to_type is provided.
    No stats rows are appended to the CSV.
    """
    if not results:
        return

    good = [r for r in results if not r.error]
    if not good:
        print("\n[stats] No successful results to summarise.")
        return

    buf = io.StringIO()

    # ── per-case_type sections ────────────────────────────────────────────────
    if sample_to_type:
        # group results by case_type, preserving insertion order
        type_groups: dict[str, list[BenchResult]] = {}
        for r in good:
            ct = sample_to_type.get(r.sample, "other")
            type_groups.setdefault(ct, []).append(r)

        for ct, ct_results in type_groups.items():
            stats = _compute_stats(ct_results)
            buf.write(_render_stats_section(
                stats, ct, len(ct_results),
                sum(1 for r in results if sample_to_type.get(r.sample) == ct),
            ))

        # overall section only when there are multiple types
        if len(type_groups) > 1:
            stats = _compute_stats(good)
            buf.write(_render_stats_section(stats, "overall", len(good), len(results)))
    else:
        stats = _compute_stats(good)
        buf.write(_render_stats_section(stats, "overall", len(good), len(results)))

    table = buf.getvalue()

    # ── print to stdout ───────────────────────────────────────────────────────
    print(table, end="")

    # ── write summary file (stats only, separate from CSV) ───────────────────
    summary_path = out_path.with_name(out_path.stem + "_summary.txt")
    summary_path.write_text(table, encoding="utf-8")
    print(f"Summary written to  : {summary_path.resolve()}")


# ── main ──────────────────────────────────────────────────────────────────────

SAMPLES: dict[str, str] = {
    "elementwise": "samples.elementwise",
    "gemm": "samples.gemm",
}


def _print_row(result: BenchResult) -> None:
    if result.error:
        print(f"{result.sample:<44} ERROR: {result.error}")
    else:
        acc_str = f"  cos={result.cosine_sim:.6f}" if result.cosine_sim != 0.0 else ""
        print(
            f"{result.sample:<44} {result.device:<5}"
            f" {result.first_call_s:>9.3f}s"
            f" {result.second_call_s:>9.3f}s"
            f" │ compile {result.total_compile_s:>8.3f}s"
            f"  dynamo {result.dynamo_s:>7.3f}s"
            f"  backend {result.backend_s:>7.3f}s"
            f"{acc_str}"
        )


def _build_tasks(
    sample_names: list[str],
    device: str,
    backend: str,
    variants_filter: set[str] | None = None,
) -> list[tuple]:
    """
    Build the flat task list from all requested samples.

    Each task is a picklable tuple:
        (idx, variant_name, module_path, fn_kwargs, device, backend, compute_accuracy)

    Modules that expose get_variant_specs() are expanded into one task per
    variant.  All others produce a single task with fn_kwargs={}.

    compute_accuracy is True when the module sets COMPUTE_ACCURACY = True.

    If variants_filter is given, only variants whose name is in the set
    are included.
    """
    import importlib
    tasks = []
    for name in sample_names:
        module_path = SAMPLES[name]
        mod = importlib.import_module(module_path)
        compute_accuracy = bool(getattr(mod, "COMPUTE_ACCURACY", True))
        if hasattr(mod, "get_variant_specs"):
            specs = mod.get_variant_specs()
        else:
            specs = [(name, {})]
        for variant_name, fn_kwargs in specs:
            if variants_filter is None or variant_name in variants_filter:
                tasks.append((
                    len(tasks), variant_name, module_path,
                    fn_kwargs, device, backend, compute_accuracy,
                ))
    return tasks


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="torch.compile phase timing benchmark")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "mlu"],
                        help="Device to run on (default: auto-detect — MLU > CUDA > CPU)")
    parser.add_argument("--output", default="compile_times.csv",
                        help="Output CSV path (default: compile_times.csv)")
    parser.add_argument("--backend", default="inductor",
                        help="torch.compile backend (default: inductor)")
    parser.add_argument("--logs-dir", default="logs",
                        help="Directory to write per-sample TORCH_LOGS (default: logs)")
    parser.add_argument("--case_type", nargs="*", default=list(SAMPLES.keys()),
                        choices=list(SAMPLES.keys()),
                        help="Which sample types to run (default: all)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker processes (default: 1 = sequential)")
    parser.add_argument("--case_name", nargs="+", default=None, metavar="CASE_NAME",
                        help="Run only these specific variant names, e.g. "
                             "elementwise_ni2_no1_sz256_2d_high (default: run all)")
    args = parser.parse_args(argv)

    if args.device is None:
        args.device = _detect_device()
        print(f"[device] auto-detected: {args.device}", flush=True)
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU", file=sys.stderr)
        args.device = "cpu"
    elif args.device == "mlu":
        try:
            available = torch.mlu.is_available()
        except AttributeError:
            available = False
        if not available:
            print("[warn] MLU not available, falling back to CPU", file=sys.stderr)
            args.device = "cpu"

    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    variants_filter = set(args.case_name) if args.case_name else None
    all_tasks = _build_tasks(args.case_type, args.device, args.backend, variants_filter)

    _warmup(args.device)

    if variants_filter and not all_tasks:
        known = _build_tasks(args.case_type, args.device, args.backend)
        known_names = [t[1] for t in known]
        unmatched = variants_filter - {t[1] for t in known}
        print(f"[error] No case_name matched: {sorted(unmatched)}", file=sys.stderr)
        print(f"        Available case names in selected case types ({len(known_names)}):", file=sys.stderr)
        for name in known_names[:10]:
            print(f"          {name}", file=sys.stderr)
        if len(known_names) > 10:
            print(f"          ... ({len(known_names) - 10} more)", file=sys.stderr)
        sys.exit(1)
    n = len(all_tasks)

    print(f"{'Sample':<44} {'Dev':<5} {'1st call':>9} {'2nd call':>9}"
          f"   {'Total cmp':>12}  {'Dynamo':>11}  {'Backend':>11}  {'cosine_sim'}")
    print("-" * 125)

    # ordered_results[idx] = (BenchResult, logs)
    ordered_results: dict[int, tuple[BenchResult, str]] = {}

    if args.workers == 1:
        # ── sequential: run directly in the main process, no subprocess ─────
        import importlib
        for idx, variant_name, module_path, fn_kwargs, device, backend, compute_accuracy in all_tasks:
            mod = importlib.import_module(module_path)

            def get_fn(device: str = device, _mod=mod, _kw=fn_kwargs):
                return _mod.get_model_and_input(**_kw, device=device)

            result, logs = _run_sample(variant_name, get_fn, device, backend, compute_accuracy)
            ordered_results[idx] = (result, logs)
            _print_row(result)
            (logs_dir / f"{result.sample}.log").write_text(logs, encoding="utf-8")
    else:
        # ── parallel: spawn worker processes ────────────────────────────────
        # Results arrive out of submission order; print immediately for live
        # progress and sort by idx before writing the CSV.
        workers = min(args.workers, n)
        print(f"[parallel] {n} variants across {workers} workers", flush=True)
        ctx = mp.get_context("spawn")
        slot_counter = ctx.Value("i", 0)
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(slot_counter, workers),
        ) as exe:
            future_to_idx = {exe.submit(_worker, task): task[0] for task in all_tasks}
            done = 0
            for fut in as_completed(future_to_idx):
                try:
                    idx, result_dict, logs = fut.result()
                except Exception as exc:
                    # Recover gracefully: record the error against the task
                    idx = future_to_idx[fut]
                    task = all_tasks[idx]
                    result = BenchResult(sample=task[1], device=args.device, error=str(exc))
                    logs = ""
                    result_dict = asdict(result)
                result = BenchResult(**result_dict)
                ordered_results[idx] = (result, logs)
                done += 1
                print(f"[{done}/{n}] ", end="")
                _print_row(result)
                (logs_dir / f"{result.sample}.log").write_text(logs, encoding="utf-8")

    # ── sequential kernel timing pass ───────────────────────────────────────
    # Always runs in the main process (single kernel at a time) so that
    # device-side event clocks are not disturbed by concurrent kernels.
    kernel_times, eager_times = _collect_kernel_times(all_tasks, args.device, args.backend)
    for idx in range(n):
        ordered_results[idx][0].kernel_time_ms = kernel_times.get(idx, 0.0)
        ordered_results[idx][0].eager_time_ms  = eager_times.get(idx, 0.0)

    # ── write CSV in submission order ───────────────────────────────────────
    out_path = Path(args.output)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(BenchResult.csv_header())
        for idx in range(n):
            result, _ = ordered_results[idx]
            writer.writerow(result.csv_row())

    # ── compute and append statistics ───────────────────────────────────────
    # build sample→case_type map from task list for per-type stats grouping
    sample_to_type: dict[str, str] = {}
    for task in all_tasks:
        _, variant_name, module_path, *_ = task
        for ct, mod_path in SAMPLES.items():
            if mod_path == module_path:
                sample_to_type[variant_name] = ct
                break

    all_results = [ordered_results[i][0] for i in range(n)]
    _write_stats(all_results, out_path, sample_to_type)

    print(f"\nResults written to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
