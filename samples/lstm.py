"""LSTM-based sequence model sample."""
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, num_layers=2, num_classes=10):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])


def get_model_and_input(device="cpu"):
    model = LSTMModel().to(device)
    x = torch.randn(32, 50, 128, device=device)
    return model, (x,)
