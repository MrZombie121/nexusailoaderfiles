from __future__ import annotations

from typing import Any

import torch


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a learning rate scheduler based on training config."""
    max_steps = int(config["training"].get("max_steps", 10000))
    warmup_steps = int(config["training"].get("warmup_steps", 100))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, max_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
