from __future__ import annotations

import torch.nn as nn


class FeedForward(nn.Module):
    """Position-wise feed-forward network used inside transformer blocks."""

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        layers = [
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(approximate="tanh"),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(intermediate_size, hidden_size))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
