from __future__ import annotations

from typing import Any

import torch


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    """Create optimizer from config settings."""
    lr = float(config["training"].get("learning_rate", 3e-4))
    weight_decay = float(config["training"].get("weight_decay", 0.01))
    betas = tuple(config["training"].get("betas", (0.9, 0.95)))
    eps = float(config["training"].get("eps", 1e-8))
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
