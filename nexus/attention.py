from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expands KV heads for Grouped Query Attention (GQA) without allocating new duplicate memory."""
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        .reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    )


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with Grouped Query Attention (GQA) and KV-cache support."""

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

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_key_value = (k, v) if use_cache else None

        # Repeat K and V for GQA using zero-copy view expansion
        if self.num_key_value_groups > 1:
            k_att = repeat_kv(k, self.num_key_value_groups)
            v_att = repeat_kv(v, self.num_key_value_groups)
        else:
            k_att = k
            v_att = v

        # When seq_len == 1 (generating with cache), the new query attends to all past keys
        is_causal = (seq_len > 1) and (attn_mask is None)

        # Scaled Dot-Product Attention (FlashAttention-2 / cuDNN compatible)
        context = F.scaled_dot_product_attention(
            q.contiguous(),
            k_att.contiguous(),
            v_att.contiguous(),
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        out = self.o_proj(context)

        if use_cache:
            return out, present_key_value
        return out
