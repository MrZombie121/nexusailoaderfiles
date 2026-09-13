from __future__ import annotations

from typing import Any
import torch

try:
    from transformers.optimization import Adafactor
    HAS_ADAFACTOR = True
except ImportError:
    HAS_ADAFACTOR = False


def get_parameter_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Separates parameters into weight-decay and no-decay groups (biases, LayerNorms)."""
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or "bias" in name or "norm" in name or "embedding" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    """Create optimizer with decoupled weight decay from config settings."""
    lr = float(config["training"].get("learning_rate", 3e-4))
    weight_decay = float(config["training"].get("weight_decay", 0.01))
    param_groups = get_parameter_groups(model, weight_decay)

    opt_name = str(config["training"].get("optimizer", "adamw")).lower()
    use_adafactor = bool(config["training"].get("use_adafactor", False)) or opt_name == "adafactor"
    if use_adafactor and HAS_ADAFACTOR:
        return Adafactor(
            param_groups,
            lr=lr,
            scale_parameter=False,
            relative_step=False,
        )

    betas = tuple(config["training"].get("betas", (0.9, 0.95)))
    eps = float(config["training"].get("eps", 1e-8))

    is_cuda = False
    try:
        p = next(model.parameters())
        is_cuda = p.is_cuda
    except StopIteration:
        pass

    if is_cuda and torch.cuda.is_available():
        try:
            return torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps, fused=True)
        except Exception:
            pass

    return torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps)
