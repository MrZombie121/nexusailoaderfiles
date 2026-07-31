from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Configuration for a decoder-only language model."""

    vocab_size: int = 32000
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_layers: int = 8
    num_heads: int = 8
    max_position_embeddings: int = 2048
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0


@dataclass
class TrainingConfig:
    """Training parameters for the project."""

    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 10000
    warmup_steps: int = 100
    checkpoint_dir: str = "checkpoints"
    logging_dir: str = "logs"
    seed: int = 42
    use_mixed_precision: bool = False


def build_default_config(model_size: str = "100M") -> dict[str, Any]:
    """Create a default config bucket for supported model sizes."""

    presets: dict[str, dict[str, Any]] = {
        "100M": {"hidden_size": 512, "intermediate_size": 2048, "num_layers": 8, "num_heads": 8},
        "300M": {"hidden_size": 768, "intermediate_size": 3072, "num_layers": 12, "num_heads": 12},
        "1B": {"hidden_size": 1536, "intermediate_size": 6144, "num_layers": 24, "num_heads": 16},
        "3B": {"hidden_size": 2560, "intermediate_size": 10240, "num_layers": 32, "num_heads": 32},
        "6B": {"hidden_size": 4096, "intermediate_size": 16384, "num_layers": 32, "num_heads": 32},
    }

    if model_size not in presets:
        raise ValueError(f"Unsupported model size: {model_size}")

    model_cfg = ModelConfig(**presets[model_size])
    training_cfg = TrainingConfig()
    return {"model": asdict(model_cfg), "training": asdict(training_cfg)}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config from disk."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_config(path: str | Path, config: dict[str, Any]) -> None:
    """Persist YAML config to disk."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True)
