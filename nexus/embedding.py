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
        self.register_buffer("positions", torch.arange(max_position_embeddings), persistent=False)

    def forward(
        self,
        seq_length: int,
        start_pos: int = 0,
        position_ids: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if position_ids is not None:
            return self.embedding(position_ids)
        positions = self.positions[start_pos : start_pos + seq_length]
        return self.embedding(positions.unsqueeze(0))
