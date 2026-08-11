from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with Grouped Query Attention (GQA) support."""

    def __init__(self, hidden_size: int, num_heads: int, num_key_value_heads: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_heads
        
        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError("num_heads must be divisible by num_key_value_heads for GQA")
            
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Repeat K and V for GQA
        if self.num_key_value_groups > 1:
            k = k[:, :, None, :, :].expand(batch_size, self.num_key_value_heads, self.num_key_value_groups, seq_len, self.head_dim)
            k = k.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
            
            v = v[:, :, None, :, :].expand(batch_size, self.num_key_value_heads, self.num_key_value_groups, seq_len, self.head_dim)
            v = v.reshape(batch_size, self.num_heads, seq_len, self.head_dim)

        # Scaled Dot-Product Attention (FlashAttention compatible)
        context = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True
        )

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(context)
