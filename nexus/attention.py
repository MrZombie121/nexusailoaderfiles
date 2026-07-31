from __future__ import annotations

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention for decoder-only transformer blocks."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        mask = self._build_causal_mask(seq_len, x.device)
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_probs = torch.softmax(scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(context)

    @staticmethod
    def _build_causal_mask(seq_length: int, device: torch.device) -> torch.Tensor:
        return torch.tril(torch.ones(seq_length, seq_length, device=device)).unsqueeze(0).unsqueeze(0)
