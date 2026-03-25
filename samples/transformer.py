"""Transformer encoder sample with multi-head self-attention."""
import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=256, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_head = d_model // num_heads
        self.num_heads = num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(self.d_head)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, num_heads=8, ff_dim=1024):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=256, num_heads=8, num_layers=4, ff_dim=1024, num_classes=10):
        super().__init__()
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.cls_head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.blocks(x)
        return self.cls_head(x.mean(dim=1))


def get_model_and_input(device="cpu"):
    model = TransformerEncoder().to(device)
    x = torch.randn(8, 64, 256, device=device)
    return model, (x,)
