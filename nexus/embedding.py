from __future__ import annotations

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Embedding layer for token IDs."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)


class PositionalEmbedding(nn.Module):
    """Learnable positional embeddings for sequence positions."""

    def __init__(self, max_position_embeddings: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_position_embeddings, hidden_size)

    def forward(self, seq_length: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(seq_length, device=device).unsqueeze(0)
        return self.embedding(positions)
