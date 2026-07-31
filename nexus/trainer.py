from __future__ import annotations

import math
from typing import Any

import torch

from .loss import CrossEntropyLoss
from .model import NexusModel
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from .tokenizer import SimpleTokenizer
from .utils import ensure_directory


class Trainer:
    """Training loop for NexusAI."""

    def __init__(self, config: dict[str, Any], dataloader: torch.utils.data.DataLoader, tokenizer: SimpleTokenizer) -> None:
        self.config = config
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_cfg = config["model"]
        self.model = NexusModel(vocab_size=tokenizer.vocab_size,
            hidden_size=int(model_cfg["hidden_size"]), intermediate_size=int(model_cfg["intermediate_size"]),
            num_layers=int(model_cfg["num_layers"]), num_heads=int(model_cfg["num_heads"]),
            max_position_embeddings=int(model_cfg.get("max_position_embeddings", 2048)),
            dropout=float(model_cfg.get("dropout", 0.1))).to(self.device)
        self.optimizer = build_optimizer(self.model, config)
        accumulation_steps = int(config["training"].get("gradient_accumulation_steps", 1))
        scheduler_config = {**config, "training": {**config["training"],
            "max_steps": max(1, math.ceil(int(config["training"].get("max_steps", 1000)) / accumulation_steps))}}
        self.scheduler = build_scheduler(self.optimizer, scheduler_config)
        self.loss_fn = CrossEntropyLoss()
        self.save_optimizer_state = bool(self.config["training"].get("save_optimizer_state", False))
        self.checkpoint_dir = ensure_directory(self.config["training"].get("checkpoint_dir", "checkpoints"))

    def load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state.get("model_state", state))
        if state.get("optimizer_state"):
            self.optimizer.load_state_dict(state["optimizer_state"])
        if state.get("scheduler_state"):
            self.scheduler.load_state_dict(state["scheduler_state"])

    def train(self) -> None:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        accumulation_steps = int(self.config["training"].get("gradient_accumulation_steps", 1))
        max_steps = int(self.config["training"].get("max_steps", 1000))
        batches_per_epoch = len(self.dataloader)
        if batches_per_epoch == 0:
            raise ValueError("DataLoader пуст: невозможно начать обучение")

        step = 0
        epoch = 0
        optimizer_step = 0
        while step < max_steps:
            epoch += 1
            for input_ids, targets in self.dataloader:
                if step >= max_steps:
                    break
                step += 1
                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)
                logits = self.model(input_ids)
                loss = self.loss_fn(logits, targets) / accumulation_steps
                loss.backward()

                if step % accumulation_steps == 0 or step == max_steps:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1
                    print(
                        f"epoch={epoch} step={step} loss={loss.item() * accumulation_steps:.4f} "
                        f"lr={self.scheduler.get_last_lr()[0]:.6f}"
                    )

                    if optimizer_step % 10 == 0:
                        self._save_checkpoint(self.checkpoint_dir / f"checkpoint_{optimizer_step}.pt", epoch, step)

        self._save_checkpoint(self.checkpoint_dir / "latest.pt", epoch, step)

    def _save_checkpoint(self, path: Any, epoch: int, step: int) -> None:
        # FP16 model-only checkpoints are much smaller than FP32 weights plus
        # AdamW's two extra parameter buffers. Loading into the model still
        # casts the tensors back to the model dtype.
        model_state = {
            key: value.detach().cpu().half() if torch.is_floating_point(value) else value.detach().cpu()
            for key, value in self.model.state_dict().items()
        }
        payload = {"model_state": model_state, "config": self.config,
                   "vocab_size": self.tokenizer.vocab_size, "epoch": epoch, "step": step,
                   "checkpoint_format": "model_fp16"}
        if self.save_optimizer_state:
            payload["optimizer_state"] = self.optimizer.state_dict()
            payload["scheduler_state"] = self.scheduler.state_dict()
        torch.save(payload, path)
