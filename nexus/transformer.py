from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feedforward import FeedForward


class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward layers."""

    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int, num_key_value_heads: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, num_key_value_heads, dropout)
        self.feedforward = FeedForward(hidden_size, intermediate_size, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        residual = x
        x = self.norm1(x)
        if use_cache:
            attn_out, present_key_value = self.attention(
                x, past_key_value=past_key_value, use_cache=True, attn_mask=attn_mask
            )
        else:
            attn_out = self.attention(x, past_key_value=None, use_cache=False, attn_mask=attn_mask)
            present_key_value = None

        x = self.dropout(attn_out) + residual

        residual = x
        x = self.norm2(x)
        x = self.feedforward(x)
        x = self.dropout(x) + residual

        if use_cache:
            return x, present_key_value
        return x
