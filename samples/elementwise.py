"""Element-wise operations sample — tests kernel fusion."""
import torch
import torch.nn as nn


class ElementwiseOps(nn.Module):
    """A compute graph with many fusable element-wise ops."""

    def forward(self, x, y):
        a = torch.sigmoid(x) * torch.tanh(y)
        b = torch.relu(a - 0.5) + torch.exp(-x.abs())
        c = torch.sqrt(b.clamp(min=1e-6)) * torch.log1p(y.abs())
        return c / (c.norm(dim=-1, keepdim=True) + 1e-8)


def get_model_and_input(device="cpu"):
    model = ElementwiseOps().to(device)
    x = torch.randn(128, 512, device=device)
    y = torch.randn(128, 512, device=device)
    return model, (x, y)
