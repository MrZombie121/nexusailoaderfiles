from __future__ import annotations

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
        self.scheduler = build_scheduler(self.optimizer, config)
        self.loss_fn = CrossEntropyLoss()
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

        optimizer_step = 0
        for step, (input_ids, targets) in enumerate(self.dataloader, start=1):
            if step > max_steps:
                break

            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            logits = self.model(input_ids)
            loss = self.loss_fn(logits, targets) / accumulation_steps
            loss.backward()

            if step % accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                optimizer_step += 1
                print(
                    f"step={step} loss={loss.item() * accumulation_steps:.4f} lr={self.scheduler.get_last_lr()[0]:.6f}"
                )

                if optimizer_step % 10 == 0:
                    torch.save(
                        {"model_state": self.model.state_dict()},
                        self.checkpoint_dir / f"checkpoint_{optimizer_step}.pt",
                    )

        torch.save({"model_state": self.model.state_dict(), "optimizer_state": self.optimizer.state_dict(),
                    "scheduler_state": self.scheduler.state_dict(), "config": self.config,
                    "vocab_size": self.tokenizer.vocab_size}, self.checkpoint_dir / "latest.pt")
