from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feedforward import FeedForward


class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward layers."""

    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.feedforward = FeedForward(hidden_size, intermediate_size, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout(x) + residual

        residual = x
        x = self.norm2(x)
        x = self.feedforward(x)
        x = self.dropout(x) + residual
        return x
