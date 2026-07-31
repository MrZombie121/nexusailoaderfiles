from __future__ import annotations

import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    """Cross-entropy loss for next-token prediction."""

    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, vocab_size = logits.shape
        return self.loss_fn(logits.view(batch_size * seq_len, vocab_size), targets.view(batch_size * seq_len))
