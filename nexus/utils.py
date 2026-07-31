from __future__ import annotations

import random
from pathlib import Path

import torch


def set_seed(seed: int) -> None:
    """Set global random seeds for torch and python random."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
