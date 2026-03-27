"""
GEMM benchmark sample: matmul and batch_matmul.

Combination grid
----------------
op_type  ∈ {matmul, batch_matmul}
M        ∈ {64, 256, 1024, 4096}
N        ∈ {64, 256, 1024, 4096}
K        ∈ {64, 256, 1024, 4096}
batch    ∈ {1, 8, 32}              – batch_matmul only
dtype    ∈ {fp32, fp16}

Total variants
--------------
  matmul       :  4 × 4 × 4 × 2           =  128
  batch_matmul :  3 × 4 × 4 × 4 × 2       =  384
  Grand total  :                              512

Accuracy comparison
-------------------
COMPUTE_ACCURACY = True signals benchmark.py to compare the compiled
output against the eager reference for every run.  Metrics recorded:

  max_abs_err   – max element-wise absolute error
  mean_abs_err  – mean element-wise absolute error
  max_rel_err   – max element-wise relative error  (vs. eager magnitude)
  cosine_sim    – cosine similarity of flattened output vectors
"""
from __future__ import annotations

import itertools

import torch
import torch.nn as nn

# ── flag consumed by benchmark._build_tasks ───────────────────────────────────
COMPUTE_ACCURACY = True

# ── dimension choices ─────────────────────────────────────────────────────────
M_CHOICES     = [64, 256, 1024, 4096]
N_CHOICES     = [64, 256, 1024, 4096]
K_CHOICES     = [64, 256, 1024, 4096]
BATCH_CHOICES = [1, 8, 32]
DTYPE_CHOICES = ["fp32", "fp16"]

_TORCH_DTYPE = {"fp32": torch.float32, "fp16": torch.float16}


# ── models ────────────────────────────────────────────────────────────────────

class MatMulModel(nn.Module):
    """Wraps a single torch.matmul (M×K) @ (K×N) → (M×N)."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)


class BatchMatMulModel(nn.Module):
    """Wraps a single torch.bmm (B×M×K) @ (B×K×N) → (B×M×N)."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.bmm(a, b)


# ── factory ───────────────────────────────────────────────────────────────────

def get_model_and_input(
    op_type: str = "matmul",
    M: int = 256,
    N: int = 256,
    K: int = 256,
    batch: int = 1,
    dtype: str = "fp32",
    device: str = "cpu",
) -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    td = _TORCH_DTYPE[dtype]
    if op_type == "matmul":
        model = MatMulModel().to(device=device, dtype=td)
        a = torch.randn(M, K, dtype=td, device=device)
        b = torch.randn(K, N, dtype=td, device=device)
    else:
        model = BatchMatMulModel().to(device=device, dtype=td)
        a = torch.randn(batch, M, K, dtype=td, device=device)
        b = torch.randn(batch, K, N, dtype=td, device=device)
    return model, (a, b)


def get_variant_specs() -> list[tuple[str, dict]]:
    """
    Return (variant_name, fn_kwargs) pairs for all combinations.

    Naming:
      matmul_m{M}_n{N}_k{K}_{dtype}
      batch_matmul_b{B}_m{M}_n{N}_k{K}_{dtype}
    """
    specs: list[tuple[str, dict]] = []

    for M, N, K, dtype in itertools.product(M_CHOICES, N_CHOICES, K_CHOICES, DTYPE_CHOICES):
        specs.append((
            f"matmul_m{M}_n{N}_k{K}_{dtype}",
            {"op_type": "matmul", "M": M, "N": N, "K": K, "dtype": dtype},
        ))

    for batch, M, N, K, dtype in itertools.product(
        BATCH_CHOICES, M_CHOICES, N_CHOICES, K_CHOICES, DTYPE_CHOICES
    ):
        specs.append((
            f"batch_matmul_b{batch}_m{M}_n{N}_k{K}_{dtype}",
            {"op_type": "batch_matmul", "M": M, "N": N, "K": K, "batch": batch, "dtype": dtype},
        ))

    return specs
