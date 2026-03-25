"""
Parametric elementwise-ops sample.

Covers all combinations of:
  n_inputs  ∈ {1, 2, 3, 4}   – number of input tensors
  n_outputs ∈ {1, 2, 3, 4}   – number of output tensors
  size      ∈ {16, 256, 8192, 32768} – number of elements per tensor (1-D)

Total: 4 × 4 × 4 = 64 variants.
"""
from __future__ import annotations

import itertools
from typing import Callable

import torch
import torch.nn as nn

N_INPUTS_CHOICES  = [1, 2, 3, 4]
N_OUTPUTS_CHOICES = [1, 2, 3, 4]
SIZE_CHOICES      = [16, 256, 8192, 32768]


class ElementwiseOps(nn.Module):
    """
    Chains n_inputs tensors through elementwise ops and produces n_outputs
    tensors.  All operations are pointwise so Inductor can fuse them into
    a single kernel.
    """

    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        super().__init__()
        if not (1 <= n_inputs <= 4):
            raise ValueError(f"n_inputs must be 1-4, got {n_inputs}")
        if not (1 <= n_outputs <= 4):
            raise ValueError(f"n_outputs must be 1-4, got {n_outputs}")
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...] | torch.Tensor:
        # ── combine all inputs into one intermediate tensor ──────────────────
        mid = inputs[0]
        if self.n_inputs >= 2:
            mid = torch.sigmoid(mid) * torch.tanh(inputs[1])
        if self.n_inputs >= 3:
            mid = mid + torch.relu(inputs[2] - 0.5)
        if self.n_inputs >= 4:
            mid = mid * torch.exp(-inputs[3].abs())

        # ── derive n_outputs from the intermediate ───────────────────────────
        out0 = torch.sqrt(mid.abs() + 1e-6)
        out1 = torch.log1p(mid.abs())
        out2 = torch.tanh(mid) * 2.0
        out3 = mid / (mid.abs().amax() + 1e-8)

        all_outputs = (out0, out1, out2, out3)
        result = all_outputs[: self.n_outputs]
        return result[0] if self.n_outputs == 1 else result


# ── factory helpers ───────────────────────────────────────────────────────────

def get_model_and_input(
    n_inputs: int = 2,
    n_outputs: int = 1,
    size: int = 256,
    device: str = "cpu",
) -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    model = ElementwiseOps(n_inputs, n_outputs).to(device)
    inputs = tuple(torch.randn(size, device=device) for _ in range(n_inputs))
    return model, inputs


def get_all_variants() -> list[tuple[str, Callable]]:
    """
    Return a list of (variant_name, get_model_and_input_fn) for every
    combination of n_inputs × n_outputs × size.
    """
    variants: list[tuple[str, Callable]] = []
    for ni, no, sz in itertools.product(N_INPUTS_CHOICES, N_OUTPUTS_CHOICES, SIZE_CHOICES):
        name = f"elementwise_ni{ni}_no{no}_sz{sz}"

        def _make(n_inputs=ni, n_outputs=no, size=sz) -> Callable:
            def _fn(device: str = "cpu"):
                return get_model_and_input(n_inputs, n_outputs, size, device)
            return _fn

        variants.append((name, _make()))
    return variants
