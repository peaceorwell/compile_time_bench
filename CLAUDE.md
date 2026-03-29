# compile_time_bench

Benchmarks `torch.compile()` compilation phase timings across parametric operator categories.

## Project Structure

```
benchmark.py          – main entry point
samples/
  elementwise.py      – parametric elementwise fusion stress test (1,408 variants)
  gemm.py             – matmul / batch_matmul dimension sweep (512 variants)
logs/                 – per-case TORCH_LOGS output (written at runtime)
results/              – suggested directory for output CSV files
```

## Key Concepts

- Two-phase execution: **Phase 1** (compilation + accuracy, supports `--workers N`) then **Phase 2** (kernel timing, always serial in main process)
- Phase 1 uses `_reset_all_caches()` before each case; Phase 2 reuses Phase 1 disk cache
- Compile-phase metrics extracted from `torch._dynamo.utils.compilation_time_metrics`
- Accuracy: compiled output vs. eager output, cast to fp32, metrics: max_abs_err, mean_abs_err, max_rel_err, cosine_sim
- Kernel timing: `torch.{cuda,mlu}.Event.elapsed_time()` (device) or wall-clock (CPU)
- Parallel workers pinned to dedicated CPU cores via `os.sched_setaffinity`

## Running

```bash
# All cases on CPU (default)
python benchmark.py

# Specific device
python benchmark.py --device cuda
python benchmark.py --device mlu

# Specific case type or case name
python benchmark.py --case_type gemm
python benchmark.py --case_name matmul_m1024_n1024_k1024_fp32

# Parallel compilation
python benchmark.py --workers 4

# Custom output
python benchmark.py --output results/my_run.csv
```

## Requirements

```
torch >= 2.1.0
```

## Development Notes

- `TORCH_LOGS="dynamo"` is set at module level before importing torch
- All compile caches are disabled globally (`TORCHINDUCTOR_CACHE_DIR`, `torch._dynamo` flags)
- `multiprocessing.get_context("spawn")` is used for worker processes
- When `--workers 1`, no subprocess is spawned (runs in-process)
- Sample modules must export `get_model_and_input(**kwargs)` and `get_variant_specs() -> list[tuple[str, dict]]`
- Optional module-level `COMPUTE_ACCURACY = True` flag controls accuracy measurement (defaults to True)
