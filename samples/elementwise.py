"""
Parametric elementwise-ops sample with broadcast and permute dimensions.

Combination grid
----------------
n_inputs  ∈ {1, 2, 3, 4}
n_outputs ∈ {1, 2, 3, 4}
size      ∈ {16, 256, 8192, 32768}   – last-dimension width
bcast_mode (n_inputs == 1 → no broadcast, unchanged):

  For n_inputs >= 2, one of the inputs (index 1) is given a shape that
  triggers broadcasting against the base shape.  The remaining inputs
  always use the base shape.

  2-D modes  (base shape = (BATCH, size)):
    no_bcast    – all inputs (BATCH, size), no broadcast
    2d_high     – inputs[1] = (1,     size)   broadcast along batch
    2d_low      – inputs[1] = (BATCH, 1   )   broadcast along size

  3-D modes  (base shape = (OUTER, BATCH, size)):
    3d_high     – inputs[1] = (1,     BATCH, size)   broadcast along outer
    3d_mid      – inputs[1] = (OUTER, 1,     size)   broadcast along batch
    3d_low      – inputs[1] = (OUTER, BATCH, 1   )   broadcast along size
    3d_hl       – inputs[1] = (1,     BATCH, 1   )   broadcast along outer+size

permute_input (applied to inputs[0]):
  False – inputs[0] is contiguous, shape unchanged
  True  – inputs[0] has its last two dimensions swapped (non-contiguous view)
          n_inputs=1: tensor is built as 2-D (BATCH, size) then permuted to
                      (size, BATCH) so that permute is meaningful
          n_inputs>=2, 2-D base: (BATCH, size) → permuted to (size, BATCH)
          n_inputs>=2, 3-D base: (OUTER, BATCH, size) → permuted to (OUTER, size, BATCH)

Fixed spatial dims: BATCH = 4, OUTER = 2.

Total variants
--------------
  Per non-permute variant: 352 (same as before)
  Per permute variant    : 352
  Grand total            : 704
"""
from __future__ import annotations

import itertools
from typing import Callable

import torch
import torch.nn as nn

# ── axis sizes ────────────────────────────────────────────────────────────────
N_INPUTS_CHOICES  = [1, 2, 3, 4]
N_OUTPUTS_CHOICES = [1, 2, 3, 4]
SIZE_CHOICES      = [16, 256, 8192, 32768]

_BATCH = 4   # batch / middle dimension
_OUTER = 2   # outer dimension (3-D only)

# ── broadcast-mode catalogue ──────────────────────────────────────────────────
# Each entry: (suffix, n_dims, special_shape_fn(size))
# special_shape_fn returns the shape for inputs[1]; None means same as base.
_BCAST_MODES: list[tuple[str, int, Callable | None]] = [
    # 2-D modes
    ("no_bcast", 2, None),
    ("2d_high",  2, lambda sz: (1,      sz)),
    ("2d_low",   2, lambda sz: (_BATCH, 1)),
    # 3-D modes
    ("3d_high",  3, lambda sz: (1,      _BATCH, sz)),
    ("3d_mid",   3, lambda sz: (_OUTER, 1,      sz)),
    ("3d_low",   3, lambda sz: (_OUTER, _BATCH, 1)),
    ("3d_hl",    3, lambda sz: (1,      _BATCH, 1)),
]


def _base_shape(n_dims: int, size: int) -> tuple[int, ...]:
    """Return the 'normal' (non-broadcast) input shape."""
    if n_dims == 2:
        return (_BATCH, size)
    return (_OUTER, _BATCH, size)


def _permute_last2(t: torch.Tensor) -> torch.Tensor:
    """Swap the last two dimensions of a tensor (must be >= 2-D)."""
    ndim = t.dim()
    perm = list(range(ndim))
    perm[-2], perm[-1] = perm[-1], perm[-2]
    return t.permute(*perm)


def _build_inputs(
    n_inputs: int,
    size: int,
    bcast_mode: str,
    device: str,
    permute_input: bool = False,
) -> tuple[torch.Tensor, ...]:
    """
    Build the input tensor tuple for a given broadcast mode.

    n_inputs == 1, permute_input=False → 1-D tensor (size,)
    n_inputs == 1, permute_input=True  → 2-D tensor (BATCH, size), then
                                         permuted to (size, BATCH)
    n_inputs >= 2 → inputs[1] gets the special broadcast shape;
                    all other inputs use the base shape.
                    When permute_input=True, inputs[0] has its last two
                    dimensions swapped.
    """
    if n_inputs == 1:
        if permute_input:
            t = torch.randn(_BATCH, size, device=device)
            return (_permute_last2(t),)
        return (torch.randn(size, device=device),)

    # Look up mode descriptor
    mode_entry = next(e for e in _BCAST_MODES if e[0] == bcast_mode)
    _, n_dims, special_fn = mode_entry

    base = _base_shape(n_dims, size)
    special = special_fn(size) if special_fn is not None else base

    shapes = [base] * n_inputs
    shapes[1] = special
    tensors = [torch.randn(s, device=device) for s in shapes]
    if permute_input:
        tensors[0] = _permute_last2(tensors[0])
    return tuple(tensors)


# ── model ─────────────────────────────────────────────────────────────────────

class ElementwiseOps(nn.Module):
    """
    Chains n_inputs tensors through pointwise ops and produces n_outputs
    tensors.  Broadcasting is handled naturally by PyTorch and Inductor
    can still fuse the resulting kernel(s).
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
    bcast_mode: str = "no_bcast",
    permute_input: bool = False,
    device: str = "cpu",
) -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    model = ElementwiseOps(n_inputs, n_outputs).to(device)
    inputs = _build_inputs(n_inputs, size, bcast_mode, device, permute_input)
    return model, inputs


def get_variant_specs() -> list[tuple[str, dict]]:
    """
    Return (variant_name, fn_kwargs) pairs for every combination of
    n_inputs × n_outputs × size × bcast_mode × permute_input.

    fn_kwargs are plain picklable dicts passed directly to
    get_model_and_input(), suitable for both sequential and parallel
    execution via multiprocessing.

    n_inputs == 1 produces no bcast_mode suffix (mode is irrelevant for 1-D).
    permute_input=True variants are suffixed with '_perm'.

    Total: 352 (no-permute) + 352 (permute) = 704 variants.
    """
    specs: list[tuple[str, dict]] = []
    for ni, no, sz in itertools.product(N_INPUTS_CHOICES, N_OUTPUTS_CHOICES, SIZE_CHOICES):
        if ni == 1:
            modes = [("", "no_bcast")]
        else:
            modes = [(f"_{mode_name}", mode_name) for mode_name, _, _ in _BCAST_MODES]

        for mode_suffix, mode_name in modes:
            base_name = f"elementwise_ni{ni}_no{no}_sz{sz}{mode_suffix}"
            for permute in (False, True):
                perm_suffix = "_perm" if permute else ""
                specs.append((
                    f"{base_name}{perm_suffix}",
                    {
                        "n_inputs": ni,
                        "n_outputs": no,
                        "size": sz,
                        "bcast_mode": mode_name,
                        "permute_input": permute,
                    },
                ))
    return specs
