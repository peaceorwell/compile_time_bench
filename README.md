# compile_time_bench

Benchmarks `torch.compile()` compilation phase timings across a variety of
model architectures.  Uses `TORCH_LOGS` and `torch._dynamo` internal metrics
to capture detailed per-phase timing, and optionally measures hardware kernel
execution time and numerical accuracy (compiled vs. eager).

## Project layout

```
benchmark.py          – main entry point
samples/
  elementwise.py      – parametric elementwise fusion stress test (1 408 variants)
  gemm.py             – matmul / batch_matmul dimension sweep (512 variants)
logs/                 – per-case TORCH_LOGS output (written at runtime)
results/              – suggested directory for output CSV files
```

## Output columns

### Timing

| Column | Description |
|---|---|
| `first_call_s` | Wall-clock time of the **first** forward pass (triggers full compilation) |
| `second_call_s` | Wall-clock time of the **second** forward pass (compiled artifact reused) |
| `dynamo_s` | Dynamo phase: Python bytecode tracing + guard building |
| `aot_s` | AOT Autograd phase: joint graph lowering |
| `backend_s` | Inductor backend: full codegen + kernel compilation |
| `total_compile_s` | `_compile.compile_inner` wall-clock total |
| `inductor_codegen_s` | Inductor sub-phase: `GraphLowering.codegen` |
| `inductor_compile_s` | Inductor sub-phase: `compile_file` (kernel compilation) |
| `inductor_load_s` | Inductor sub-phase: `PyCodeCache.load_by_key_path` |
| `pre_grad_passes_s` | `_recursive_pre_grad_passes` |
| `post_grad_passes_s` | `_recursive_post_grad_passes` |
| `joint_graph_passes_s` | `_recursive_joint_graph_passes` |
| `kernel_time_ms` | Hardware kernel execution time in **milliseconds** (device-side event clock) |

### Counters

| Column | Description |
|---|---|
| `cache_hit` | `1` if AOT/FX cache was hit (Inductor skipped) |
| `graph_breaks` | Number of graph breaks detected by Dynamo |
| `frames_compiled` | Number of frames/graphs compiled |

### Accuracy (compiled vs. eager)

Outputs are cast to fp32 before comparison to avoid fp16 overflow artefacts.

| Column | Description |
|---|---|
| `max_abs_err` | Max element-wise absolute error |
| `mean_abs_err` | Mean element-wise absolute error |
| `max_rel_err` | Max element-wise relative error (relative to eager magnitude) |
| `cosine_sim` | Cosine similarity of flattened output vectors (`1.0` = identical) |

## Sample modules

| `--case_type` | Description | Variants |
|---|---|---|
| `elementwise` | Parametric elementwise fusion: arithmetic + activations | 1 408 |
| `gemm` | `matmul` and `batch_matmul` dimension sweep | 512 |

### elementwise variant naming

```
elementwise_ni{N}_no{M}_sz{S}[_{bcast_mode}][_perm]_{dtype}

N         – n_inputs  ∈ {1, 2, 3, 4}
M         – n_outputs ∈ {1, 2, 3, 4}
S         – last-dim size ∈ {16, 256, 8192, 32768}
bcast_mode – (n_inputs ≥ 2 only) no_bcast | 2d_high | 2d_low |
              3d_high | 3d_mid | 3d_low | 3d_hl
_perm     – inputs[0] has its last two dims permuted (non-contiguous)
dtype     – fp32 | fp16
```

### gemm variant naming

```
matmul_m{M}_n{N}_k{K}_{dtype}
batch_matmul_b{B}_m{M}_n{N}_k{K}_{dtype}

M, N, K ∈ {64, 256, 1024, 4096}
B       ∈ {1, 8, 32}
dtype   ∈ {fp32, fp16}
```

## Execution phases

The benchmark runs in two sequential phases:

1. **Compilation phase** – runs all cases (optionally in parallel via `--workers`).
   Records all compile-time metrics, `first_call_s`, `second_call_s`, and
   accuracy metrics.  Parallel workers are each pinned to a dedicated slice of
   CPU cores to reduce scheduling contention.

2. **Kernel timing phase** – always runs sequentially in the main process,
   reusing compiled artifacts from phase 1 (disk cache).  Measures
   `kernel_time_ms` with `torch.{cuda,mlu}.Event.elapsed_time()` (GPU/MLU)
   or wall-clock (CPU).  Running this phase serially ensures kernels are not
   executed concurrently, which would skew hardware timing.

## Statistics

After writing the per-case CSV rows, the benchmark appends three summary rows
(`[max]`, `[min]`, `[avg]`) for every numeric column, and prints the same
table to stdout.

## Usage

```bash
# Run all case types on CPU (default)
python benchmark.py

# Run on CUDA or MLU (Cambricon)
python benchmark.py --device cuda
python benchmark.py --device mlu

# Run specific case types
python benchmark.py --case_type mlp cnn resnet
python benchmark.py --case_type gemm
python benchmark.py --case_type elementwise

# Run a specific case by name
python benchmark.py --case_name matmul_m1024_n1024_k1024_fp32
python benchmark.py --case_name elementwise_ni2_no1_sz256_no_bcast_fp16

# Combine: narrow scope to a type, then pick a case
python benchmark.py --case_type elementwise \
    --case_name elementwise_ni4_no4_sz8192_3d_hl_perm_fp16

# Parallel compilation across N worker processes (kernel timing is always serial)
python benchmark.py --workers 4

# Custom output path
python benchmark.py --output results/my_run.csv

# Custom backend
python benchmark.py --backend eager
```

Results are written to `compile_times.csv` (configurable via `--output`),
with `[max]`/`[min]`/`[avg]` summary rows appended at the end.
Per-case `TORCH_LOGS` output is saved under `logs/<case_name>.log`.

## TORCH_LOGS

The script sets `TORCH_LOGS="dynamo"` before importing torch.  Log output
during the warmup pass and the kernel timing pass is suppressed to keep
stdout clean.

Override before running for more verbosity:

```bash
TORCH_LOGS="+dynamo" python benchmark.py
```

## Requirements

```
torch >= 2.1.0
```
