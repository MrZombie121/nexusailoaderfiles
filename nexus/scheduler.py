from __future__ import annotations

import math
from typing import Any
import torch


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a learning rate scheduler with warmup and cosine or linear decay."""
    train_cfg = config.get("training", {})
    max_steps = max(1, int(train_cfg.get("max_steps", 10000)))
    warmup_steps = max(0, int(train_cfg.get("warmup_steps", 100)))
    schedule_type = str(train_cfg.get("lr_scheduler_type", train_cfg.get("scheduler", "cosine"))).lower()
    min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.1))

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))

        if current_step >= max_steps:
            return min_lr_ratio

        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)

        if schedule_type == "cosine":
            # Cosine decay down to min_lr_ratio
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
        elif schedule_type == "constant":
            return 1.0
        else:
            # Linear decay down to min_lr_ratio
            return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
