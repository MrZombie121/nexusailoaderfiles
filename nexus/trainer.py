from __future__ import annotations

import math
from typing import Any
import torch
from itertools import cycle

# Попытка импорта XLA для TPU
try:
    import torch_xla.core.xla_model as xm
    HAS_XLA = True
except ImportError:
    HAS_XLA = False

# Импорт для смешанной точности (AMP)
try:
    from torch.amp import autocast, GradScaler
    HAS_AMP = True
except ImportError:
    HAS_AMP = False

from .loss import CrossEntropyLoss
from .model import NexusModel
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from .tokenizer import SimpleTokenizer
from .utils import ensure_directory

class Trainer:
    """Training loop for NexusAI with support for GPU (AMP) and TPU."""

    def __init__(self, config: dict[str, Any], dataloader: torch.utils.data.DataLoader, tokenizer: SimpleTokenizer) -> None:
        self.config = config
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        
        use_mixed_precision = bool(config.get("training", {}).get("use_mixed_precision", True))

        if HAS_XLA:
            import os
            if use_mixed_precision:
                os.environ["XLA_USE_BF16"] = "1" # Нативная поддержка bfloat16 для TPU
            try:
                import torch_xla
                self.device = torch_xla.device() # Новый синтаксис (без DeprecationWarning)
            except AttributeError:
                self.device = xm.xla_device() # Старый синтаксис для совместимости
            self.scaler = None
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.scaler = GradScaler("cuda") if HAS_AMP and self.device.type == 'cuda' and use_mixed_precision else None

        model_cfg = config["model"]
        
        # Загрузка модели сразу в bfloat16 для экономии RAM/VRAM
        dtype = torch.bfloat16 if use_mixed_precision and HAS_XLA else torch.float32
        
        # Избегаем meta-устройства, так как оно требует .to_empty() и ручной инициализации
        init_device = torch.device("cpu" if HAS_XLA else self.device)
        with torch.device(init_device): # init on cpu first for TPU
            self.model = NexusModel(
                vocab_size=tokenizer.vocab_size,
                hidden_size=int(model_cfg["hidden_size"]),
                intermediate_size=int(model_cfg["intermediate_size"]),
                num_layers=int(model_cfg["num_layers"]),
                num_heads=int(model_cfg["num_heads"]),
                num_key_value_heads=int(model_cfg.get("num_key_value_heads", model_cfg["num_heads"])),
                max_position_embeddings=int(model_cfg.get("max_position_embeddings", 1024)),
                dropout=float(model_cfg.get("dropout", 0.0)),
                gradient_checkpointing=bool(config.get("training", {}).get("gradient_checkpointing", True))
            )
            
        # Для TPU конвертируем перед отправкой
        self.model = self.model.to(dtype).to(self.device)

        # Оптимизация для XLA: компиляция графа (если torch >= 2.0)
        # В PyTorch/XLA torch.compile работает отлично и ускоряет обучение
        if HAS_XLA and hasattr(torch, "compile"):
            # self.model = torch.compile(self.model, backend="openxla") # Раскомментировать при необходимости
            pass
        
        self.optimizer = build_optimizer(self.model, config)
        accumulation_steps = int(config["training"].get("gradient_accumulation_steps", 1))
        
        scheduler_config = {
            **config,
            "training": {**config["training"], "max_steps": max(1, math.ceil(int(config["training"].get("max_steps", 1000)) / accumulation_steps))}
        }
        self.scheduler = build_scheduler(self.optimizer, scheduler_config)
        self.loss_fn = CrossEntropyLoss()
        self.save_optimizer_state = bool(self.config["training"].get("save_optimizer_state", False))
        self.checkpoint_dir = ensure_directory(self.config["training"].get("checkpoint_dir", "checkpoints"))

    def train(self) -> None:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        accumulation_steps = int(self.config["training"].get("gradient_accumulation_steps", 1))
        max_steps = int(self.config["training"].get("max_steps", 1000))
        checkpoint_every_epochs = max(1, int(self.config["training"].get("checkpoint_every_epochs", 1)))

        if len(self.dataloader) == 0:
            raise ValueError("DataLoader пуст")

        step = 0
        epoch = 0
        optimizer_step = 0
        
        for input_ids, targets in cycle(self.dataloader):
            if step >= max_steps: break
            step += 1
            if step % len(self.dataloader) == 1: epoch += 1

            input_ids, targets = input_ids.to(self.device), targets.to(self.device)
            
            # Обучение с использованием AMP (Mixed Precision) для экономии памяти
            if self.scaler:
                with autocast("cuda"):
                    logits = self.model(input_ids)
                    loss = self.loss_fn(logits, targets) / accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                logits = self.model(input_ids)
                loss = self.loss_fn(logits, targets) / accumulation_steps
                loss.backward()
                
            loss_val = loss.item() * accumulation_steps
            del logits, loss # Освобождаем память от графов до шага оптимизатора

            if step % accumulation_steps == 0 or step == max_steps:
                if HAS_XLA:
                    xm.optimizer_step(self.optimizer)
                    xm.mark_step()
                elif self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                
                if optimizer_step % 10 == 0:
                    print(f"step={step}/{max_steps} loss={loss_val:.4f} lr={self.scheduler.get_last_lr()[0]:.6f}")
                    
            if step % len(self.dataloader) == 0 and epoch % checkpoint_every_epochs == 0:
                self._save_checkpoint(self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt", epoch, step)

        self._save_checkpoint(self.checkpoint_dir / "latest.pt", epoch, step)

    def _save_checkpoint(self, path: Any, epoch: int, step: int) -> None:
        model_state = {k: v.detach().cpu().half() if torch.is_floating_point(v) else v.detach().cpu() for k, v in self.model.state_dict().items()}
        payload = {"model_state": model_state, "config": self.config, "vocab_size": self.tokenizer.vocab_size, "epoch": epoch, "step": step}
        if self.save_optimizer_state:
            payload["optimizer_state"] = self.optimizer.state_dict()
            payload["scheduler_state"] = self.scheduler.state_dict()
        torch.save(payload, path)
