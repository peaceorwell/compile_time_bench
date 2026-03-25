# compile_time_bench

Benchmarks `torch.compile()` compilation phase timings across a variety of model architectures, using `TORCH_LOGS` to capture detailed per-phase logs.

## What it measures

For each sample model the benchmark records:

| Column | Description |
|---|---|
| `first_call_s` | Wall-clock time of the first forward pass (includes full compilation) |
| `second_call_s` | Wall-clock time of the second forward pass (compiled artifact reused) |
| `dynamo_s` | Time spent in **Dynamo** (Python tracing / graph capture) |
| `aot_s` | Time spent in **AOT Autograd** (joint graph lowering) |
| `backend_s` | Time spent in the **Inductor** backend (kernel codegen / compilation) |
| `total_compile_s` | Sum of all compile-phase times |
| `frames_compiled` | Number of frames/graphs compiled |
| `graph_breaks` | Number of graph breaks detected |

## Sample models

| Sample | Architecture |
|---|---|
| `mlp` | 4-layer MLP (784 → 512 → 256 → 128 → 10) |
| `cnn` | 3-block CNN with BatchNorm + MaxPool |
| `transformer` | 4-layer Transformer encoder with multi-head attention |
| `resnet` | ResNet-like with 3 stages of residual blocks |
| `lstm` | 2-layer LSTM classifier |
| `elementwise` | Element-wise fusion stress test |

## Usage

```bash
# Run all case types on CPU (default)
python benchmark.py

# Run on CUDA
python benchmark.py --device cuda

# Run on MLU (Cambricon)
python benchmark.py --device mlu

# Run specific case types
python benchmark.py --case_type mlp cnn resnet

# Run a specific case by name
python benchmark.py --case_name elementwise_ni2_no1_sz256_2d_high

# Run multiple specific cases
python benchmark.py --case_name mlp elementwise_ni4_no4_sz32768_3d_hl

# Combine: narrow scope to a type, then pick a case
python benchmark.py --case_type elementwise --case_name elementwise_ni2_no1_sz256_2d_high

# Parallel execution across N worker processes
python benchmark.py --workers 4

# Change output path
python benchmark.py --output results/my_run.csv
```

Results are saved to `compile_times.csv` (configurable via `--output`).
Per-case `TORCH_LOGS` output is saved to `logs/<case_name>.log`.

## TORCH_LOGS

The script sets `TORCH_LOGS="dynamo"` before importing torch, capturing
Dynamo-phase timing lines. The captured log text is written to
`logs/<sample>.log` for inspection.

You can override this before running:

```bash
TORCH_LOGS="+dynamo" python benchmark.py   # verbose dynamo logs
```

## Requirements

```
torch >= 2.1.0
```
