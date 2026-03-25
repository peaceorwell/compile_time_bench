"""
PyTorch torch.compile phase timing benchmark.

Measures compilation time broken down by phase (Dynamo / AOT / Inductor) for
each sample model, then writes the results to compile_times.csv.

Usage
-----
    python benchmark.py [--device cpu|cuda] [--output compile_times.csv]

TORCH_LOGS
----------
The script sets  TORCH_LOGS="+dynamo,+aot,+inductor"  before importing torch
so that PyTorch emits structured log lines for every compilation event.  These
logs are captured per-sample and written alongside the timing summary.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Callable

# ── TORCH_LOGS must be set BEFORE importing torch ──────────────────────────
os.environ.setdefault("TORCH_LOGS", "+dynamo,+aot,+inductor")

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

# ── logging setup ───────────────────────────────────────────────────────────
_TORCH_LOG_NAMESPACES = [
    "torch._dynamo",
    "torch._inductor",
    "torch._functorch",
    "torch.fx",
]

_LOGFMT = "%(name)s | %(levelname)s | %(message)s"


def _attach_capture_handler(buf: io.StringIO) -> list[logging.Handler]:
    """Attach a StringIO-backed handler to every relevant torch logger."""
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter(_LOGFMT))
    handlers = []
    for ns in _TORCH_LOG_NAMESPACES:
        lg = logging.getLogger(ns)
        lg.addHandler(handler)
        lg.setLevel(logging.DEBUG)
        handlers.append((lg, handler))
    return handlers


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
    # error message if compilation failed
    error: str = ""

    @classmethod
    def csv_header(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    def csv_row(self) -> list:
        return [getattr(self, f.name) for f in fields(self)]


# ── per-sample benchmark ─────────────────────────────────────────────────────

def _run_sample(
    name: str,
    get_model_and_input: Callable,
    device: str,
    backend: str = "inductor",
) -> tuple[BenchResult, str]:
    """Compile and run one sample; return (BenchResult, captured_logs)."""
    result = BenchResult(sample=name, device=device)

    # Reset dynamo state so timings are isolated per sample.
    torch._dynamo.reset()

    log_buf = io.StringIO()
    handlers = _attach_capture_handler(log_buf)

    try:
        model, inputs = get_model_and_input(device=device)
        model.eval()

        compiled = torch.compile(model, backend=backend, fullgraph=False)

        # ── first call: triggers full compilation ──
        t0 = time.perf_counter()
        with torch.no_grad():
            compiled(*inputs)
        if device == "cuda":
            torch.cuda.synchronize()
        result.first_call_s = time.perf_counter() - t0

        # ── second call: compiled artifact is reused ──
        t1 = time.perf_counter()
        with torch.no_grad():
            compiled(*inputs)
        if device == "cuda":
            torch.cuda.synchronize()
        result.second_call_s = time.perf_counter() - t1

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


# ── main ──────────────────────────────────────────────────────────────────────

SAMPLES: dict[str, str] = {
    "mlp": "samples.mlp",
    "cnn": "samples.cnn",
    "transformer": "samples.transformer",
    "resnet": "samples.resnet",
    "lstm": "samples.lstm",
    "elementwise": "samples.elementwise",
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="torch.compile phase timing benchmark")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Device to run on (default: cpu)")
    parser.add_argument("--output", default="compile_times.csv",
                        help="Output CSV path (default: compile_times.csv)")
    parser.add_argument("--backend", default="inductor",
                        help="torch.compile backend (default: inductor)")
    parser.add_argument("--logs-dir", default="logs",
                        help="Directory to write per-sample TORCH_LOGS (default: logs)")
    parser.add_argument("--samples", nargs="*", default=list(SAMPLES.keys()),
                        choices=list(SAMPLES.keys()),
                        help="Which samples to run (default: all)")
    args = parser.parse_args(argv)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU", file=sys.stderr)
        args.device = "cpu"

    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    results: list[BenchResult] = []

    print(f"{'Sample':<40} {'Device':<6} {'1st call':>10} {'2nd call':>10} "
          f"{'Dynamo':>10} {'AOT':>10} {'Backend':>10} {'Total cmp':>10}")
    print("-" * 112)

    import importlib

    for name in args.samples:
        module_path = SAMPLES[name]
        mod = importlib.import_module(module_path)

        # If the module exposes get_all_variants(), expand into sub-runs;
        # otherwise fall back to the single get_model_and_input() entry point.
        if hasattr(mod, "get_all_variants"):
            variant_list = mod.get_all_variants()
        else:
            variant_list = [(name, mod.get_model_and_input)]

        for variant_name, get_fn in variant_list:
            result, logs = _run_sample(
                name=variant_name,
                get_model_and_input=get_fn,
                device=args.device,
                backend=args.backend,
            )
            results.append(result)

            # Save per-variant TORCH_LOGS
            (logs_dir / f"{variant_name}.log").write_text(logs, encoding="utf-8")

            if result.error:
                print(f"{variant_name:<40} ERROR: {result.error}")
            else:
                print(
                    f"{variant_name:<40} {result.device:<6}"
                    f" {result.first_call_s:>10.4f}s"
                    f" {result.second_call_s:>10.4f}s"
                    f" {result.dynamo_s:>10.4f}s"
                    f" {result.aot_s:>10.4f}s"
                    f" {result.backend_s:>10.4f}s"
                    f" {result.total_compile_s:>10.4f}s"
                )

    # ── write CSV ──────────────────────────────────────────────────────────
    out_path = Path(args.output)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(BenchResult.csv_header())
        for r in results:
            writer.writerow(r.csv_row())

    print(f"\nResults written to: {out_path.resolve()}")
    print(f"Per-sample logs  : {logs_dir.resolve()}/")


if __name__ == "__main__":
    main()
