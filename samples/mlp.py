"""Simple Multi-Layer Perceptron sample."""
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=(512, 256, 128), output_dim=10):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def get_model_and_input(device="cpu"):
    model = MLP().to(device)
    x = torch.randn(64, 784, device=device)
    return model, (x,)
