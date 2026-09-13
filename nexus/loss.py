from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_perplexity(loss_val: float | torch.Tensor) -> float:
    """Computes perplexity from cross-entropy loss, safely clamped against overflow."""
    if isinstance(loss_val, torch.Tensor):
        loss_val = loss_val.item()
    try:
        # Clamp exponent to prevent math overflow
        loss_val = max(0.0, min(float(loss_val), 50.0))
        return math.exp(loss_val)
    except (OverflowError, ValueError):
        return float("inf")


class CrossEntropyLoss(nn.Module):
    """Fast cross-entropy loss for next-token prediction with optional label smoothing."""

    def __init__(self, ignore_index: int = -100, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )
