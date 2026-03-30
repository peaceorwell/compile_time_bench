# compile_time_bench

**[中文](README_CN.md) | English**

Benchmarks `torch.compile()` compilation phase timings across two parametric
operator categories.  Uses `TORCH_LOGS` and `torch._dynamo` internal metrics
to capture detailed per-phase timing, measures hardware kernel execution time,
and compares compiled vs. eager numerical accuracy for every case.

## Project layout

```
benchmark.py          – main entry point
samples/
  elementwise.py      – parametric elementwise fusion stress test (1,408 variants)
  gemm.py             – matmul / batch_matmul dimension sweep (512 variants)
logs/                 – per-case TORCH_LOGS output (written at runtime)
results/              – suggested directory for output CSV files
```

## Output columns

### Timing (seconds unless noted)

| Column | Description |
|---|---|
| `first_call_s` | Wall-clock time of the **first** forward pass (triggers full compilation) |
| `second_call_s` | Wall-clock time of the **second** forward pass (compiled artifact reused) |
| `dynamo_s` | Dynamo phase: Python bytecode tracing + guard building |
| `aot_s` | AOT Autograd phase: joint graph lowering and metadata collection |
| `backend_s` | Inductor backend: full codegen + kernel compilation |
| `total_compile_s` | Total compile time (`_compile.compile_inner` wall-clock) |
| `inductor_codegen_s` | Inductor sub-phase: `GraphLowering.codegen` (IR → C++/Triton) |
| `inductor_compile_s` | Inductor sub-phase: `compile_file` (C++/Triton → `.so`) |
| `inductor_load_s` | Inductor sub-phase: `PyCodeCache.load_by_key_path` (load compiled kernel) |
| `pre_grad_passes_s` | Pre-grad graph transformation passes |
| `post_grad_passes_s` | Post-grad graph transformation passes |
| `joint_graph_passes_s` | Joint (forward+backward) graph transformation passes |
| `kernel_time_ms` | Hardware kernel execution time in **milliseconds** (device-side event clock) |

### Counters

| Column | Description |
|---|---|
| `cache_hit` | `1` if AOT/FX cache was hit and Inductor compilation was skipped |
| `graph_breaks` | Number of graph breaks detected by Dynamo |
| `frames_compiled` | Number of frames/graphs compiled |

### Accuracy (compiled vs. eager)

Measured during the compilation phase (phase 1) for every case.
Outputs are cast to fp32 before comparison to avoid fp16 magnitude issues.

| Column | Description |
|---|---|
| `max_abs_err` | Max element-wise absolute error |
| `mean_abs_err` | Mean element-wise absolute error |
| `max_rel_err` | Max element-wise relative error (normalised by eager output magnitude) |
| `cosine_sim` | Cosine similarity of flattened output vectors (`1.0` = identical) |

## Sample modules

| `--case_type` | Description | Variants |
|---|---|---|
| `elementwise` | Parametric elementwise fusion: arithmetic (+−×÷) and activation ops | 1,408 |
| `gemm` | `matmul` and `batch_matmul` dimension sweep | 512 |

### elementwise variant naming

```
elementwise_ni{N}_no{M}_sz{S}[_{bcast_mode}][_perm]_{dtype}

N          – n_inputs  ∈ {1, 2, 3, 4}
M          – n_outputs ∈ {1, 2, 3, 4}
S          – last-dim size ∈ {16, 256, 8192, 32768}
bcast_mode – (n_inputs ≥ 2 only)
             no_bcast | 2d_high | 2d_low | 3d_high | 3d_mid | 3d_low | 3d_hl
_perm      – inputs[0] has its last two dims permuted (non-contiguous view)
dtype      – fp32 | fp16
```

Activation count per case scales with `max(n_inputs, n_outputs)`, capped at 4.
Activation ops used: `sigmoid`, `tanh`, `relu`, `sqrt`.

### gemm variant naming

```
matmul_m{M}_n{N}_k{K}_{dtype}
batch_matmul_b{B}_m{M}_n{N}_k{K}_{dtype}

M, N, K ∈ {64, 256, 1024, 4096}
B       ∈ {1, 8, 32}
dtype   ∈ {fp32, fp16}
```

## Execution phases

Each case runs a total of **6 forward passes** across two sequential phases:

**Phase 1 — Compilation benchmark** (supports `--workers N` for parallelism)

| Pass | Purpose |
|---|---|
| 1st | Triggers `torch.compile` — records `first_call_s` and all compile-phase metrics |
| 2nd | Reuses compiled artifact — records `second_call_s` |
| 3rd | Eager forward — accuracy reference |
| 4th | Compiled forward — accuracy comparison → `max_abs_err`, `cosine_sim`, … |

Parallel workers are each pinned to a dedicated CPU core slice via
`os.sched_setaffinity` to reduce scheduling contention and compile-time variance.

**Phase 2 — Kernel timing** (always serial in the main process)

| Pass | Purpose |
|---|---|
| 5th | Re-compiles using phase-1 disk cache (fast) |
| 6th | Timed with `torch.{cuda,mlu}.Event.elapsed_time()` — records `kernel_time_ms` |

Running phase 2 serially ensures only one kernel executes at a time, giving
accurate device-side hardware timing.  On CPU, wall-clock is used instead.

## Statistics

After writing the per-case CSV rows, the benchmark appends three summary rows
(`[max]`, `[min]`, `[avg]`) for every numeric column, and prints the same
table to stdout.

## Usage

```bash
# Run all cases (device auto-detected: MLU > CUDA > CPU)
python benchmark.py

# Run a specific case type
python benchmark.py --case_type gemm
python benchmark.py --case_type elementwise

# Run a specific case by name
python benchmark.py --case_name matmul_m1024_n1024_k1024_fp32
python benchmark.py --case_name elementwise_ni2_no1_sz256_no_bcast_fp16

# Narrow to a type, then filter by name
python benchmark.py --case_type elementwise \
    --case_name elementwise_ni4_no4_sz8192_3d_hl_perm_fp16

# Parallel compilation across N worker processes (kernel timing is always serial)
python benchmark.py --workers 4

# Custom output path
python benchmark.py --output results/my_run.csv

# Override device explicitly
python benchmark.py --device cpu

# Change backend (default: inductor)
python benchmark.py --backend aot_eager
```

Results are written to `compile_times.csv` (configurable via `--output`),
with `[max]`/`[min]`/`[avg]` summary rows appended at the end.
Per-case `TORCH_LOGS` output is saved under `logs/<case_name>.log`.

## TORCH_LOGS

The script sets `TORCH_LOGS="dynamo"` before importing torch.  Log output
during the warmup pass and the kernel timing phase is suppressed to keep
stdout clean.

Override before running for more verbosity:

```bash
TORCH_LOGS="+dynamo" python benchmark.py
```

## Requirements

```
torch >= 2.1.0
```
