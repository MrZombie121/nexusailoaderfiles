from __future__ import annotations

import math
import time
from itertools import cycle
from pathlib import Path
from typing import Any
import torch

# CUDA Tensor Cores optimization
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

# TPU XLA import
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    if not hasattr(torch, "xla"):
        torch.xla = torch_xla
    HAS_XLA = True
except ImportError:
    HAS_XLA = False

# Automatic Mixed Precision (AMP)
try:
    from torch.amp import autocast, GradScaler
    HAS_AMP = True
except ImportError:
    HAS_AMP = False

from .loss import CrossEntropyLoss, compute_perplexity
from .model import NexusModel
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from .tokenizer import SimpleTokenizer
from .utils import ensure_directory


class Trainer:
    """High-performance training loop for NexusAI supporting GPU (AMP/TF32/FlashAttention), TPU (XLA), and CPU."""

    def __init__(
        self,
        config: dict[str, Any],
        dataloader: torch.utils.data.DataLoader,
        tokenizer: SimpleTokenizer,
        val_dataloader: torch.utils.data.DataLoader | None = None,
    ) -> None:
        self.config = config
        self.raw_dataloader = dataloader
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.tokenizer = tokenizer

        train_cfg = config.get("training", {})
        use_mixed_precision = bool(train_cfg.get("use_mixed_precision", True))
        precision_dtype = str(train_cfg.get("precision_dtype", "bfloat16" if HAS_XLA else "float16")).lower()

        if HAS_XLA:
            import os
            if use_mixed_precision:
                os.environ["XLA_USE_BF16"] = "1"
            try:
                import torch_xla
                self.device = torch_xla.device()
            except AttributeError:
                self.device = xm.xla_device()
            self.scaler = None
            self.autocast_dtype = None
            try:
                self.dataloader = pl.MpDeviceLoader(self.raw_dataloader, self.device)
            except Exception:
                self.dataloader = self.raw_dataloader
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.device.type == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                if hasattr(torch, "set_float32_matmul_precision"):
                    torch.set_float32_matmul_precision("high")

            if self.device.type == "cuda" and use_mixed_precision and HAS_AMP:
                if precision_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
                    self.autocast_dtype = torch.bfloat16
                    self.scaler = None
                else:
                    self.autocast_dtype = torch.float16
                    self.scaler = GradScaler("cuda")
            else:
                self.autocast_dtype = None
                self.scaler = None

        model_cfg = config["model"]
        hidden_dim = int(model_cfg["hidden_size"])
        # For 3B and 6B models, initialize directly in half/bfloat16 to avoid 12-24 GB FP32 VRAM spike
        if self.device.type == "cuda" and use_mixed_precision and hidden_dim >= 2560:
            dtype = self.autocast_dtype if self.autocast_dtype is not None else torch.float16
        else:
            dtype = torch.bfloat16 if (use_mixed_precision and HAS_XLA) else torch.float32

        init_device = torch.device("cpu" if HAS_XLA else self.device)
        with torch.device(init_device):
            self.model = NexusModel(
                vocab_size=tokenizer.vocab_size,
                hidden_size=int(model_cfg["hidden_size"]),
                intermediate_size=int(model_cfg["intermediate_size"]),
                num_layers=int(model_cfg["num_layers"]),
                num_heads=int(model_cfg["num_heads"]),
                num_key_value_heads=int(model_cfg.get("num_key_value_heads", model_cfg["num_heads"])),
                max_position_embeddings=int(model_cfg.get("max_position_embeddings", 2048)),
                dropout=float(model_cfg.get("dropout", 0.0)),
                gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", False)),
            )

        self.model = self.model.to(dtype).to(self.device)

        self.optimizer = build_optimizer(self.model, config)

        # Multi-GPU support via DataParallel vs single-GPU torch.compile
        self.is_multi_gpu = False
        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            num_gpus = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
            print(
                f"🚀 Обнаружено {num_gpus} GPU ({', '.join(gpu_names)}). "
                f"Задействуем параллельное обучение torch.nn.DataParallel на всех видеокартах!"
            )
            self.model = torch.nn.DataParallel(self.model)
            self.is_multi_gpu = True
        else:
            use_compile = bool(train_cfg.get("compile", True))
            if use_compile and hasattr(torch, "compile") and self.device.type == "cuda":
                try:
                    self.model = torch.compile(self.model)
                    print("Модель оптимизирована с помощью torch.compile (kernel fusion)")
                except Exception as e:
                    print(f"torch.compile пропущен ({e}), используется standard eager mode")

        accumulation_steps = max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))
        steps_per_epoch = max(1, len(self.raw_dataloader))
        if "max_steps" in train_cfg and train_cfg["max_steps"] is not None and int(train_cfg["max_steps"]) > 0:
            total_steps = int(train_cfg["max_steps"])
            config["training"]["epochs"] = total_steps / steps_per_epoch
        elif "epochs" in train_cfg and train_cfg["epochs"] is not None and float(train_cfg["epochs"]) > 0:
            epochs = float(train_cfg["epochs"])
            total_steps = max(1, math.ceil(epochs * steps_per_epoch))
            config["training"]["max_steps"] = total_steps
        else:
            total_steps = int(train_cfg.get("max_steps", 200))
            config["training"]["epochs"] = total_steps / steps_per_epoch

        scheduler_config = {
            **config,
            "training": {**config["training"], "max_steps": max(1, math.ceil(total_steps / accumulation_steps))},
        }
        self.scheduler = build_scheduler(self.optimizer, scheduler_config)

        label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
        self.loss_fn = CrossEntropyLoss(ignore_index=-100, label_smoothing=label_smoothing)

        self.max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
        self.save_optimizer_state = bool(train_cfg.get("save_optimizer_state", False))
        self.checkpoint_dir = Path(ensure_directory(train_cfg.get("checkpoint_dir", "checkpoints")))
        self.checkpoint_name = train_cfg.get("checkpoint_name", "latest.pt")
        self.best_loss = float("inf")

    def load_checkpoint(self, path: Any) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint.get("model_state", checkpoint)
        raw_model = getattr(self.model, "module", self.model)
        raw_model = getattr(raw_model, "_orig_mod", raw_model)
        raw_model.load_state_dict(state_dict)
        if "optimizer_state" in checkpoint and hasattr(self, "optimizer"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint and hasattr(self, "scheduler"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        if "best_loss" in checkpoint:
            self.best_loss = float(checkpoint["best_loss"])

    def evaluate(self, dataloader: torch.utils.data.DataLoader | None = None) -> tuple[float, float]:
        """Runs evaluation over validation dataloader and returns (loss, perplexity)."""
        loader = dataloader or self.val_dataloader
        if loader is None or len(loader) == 0:
            return 0.0, 0.0

        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in loader:
                if HAS_XLA:
                    input_ids, targets = batch
                else:
                    input_ids = batch[0].to(self.device, non_blocking=True)
                    targets = batch[1].to(self.device, non_blocking=True)

                if self.autocast_dtype is not None:
                    with autocast("cuda", dtype=self.autocast_dtype):
                        logits = self.model(input_ids)
                        loss = self.loss_fn(logits, targets)
                else:
                    logits = self.model(input_ids)
                    loss = self.loss_fn(logits, targets)

                total_loss += loss.item()
                total_batches += 1

        self.model.train()
        avg_loss = total_loss / max(1, total_batches)
        ppl = compute_perplexity(avg_loss)
        return avg_loss, ppl

    def train(self) -> None:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        accumulation_steps = max(1, int(self.config["training"].get("gradient_accumulation_steps", 1)))
        steps_per_epoch = max(1, len(self.raw_dataloader))
        train_cfg = self.config.get("training", {})
        if "max_steps" in train_cfg and train_cfg["max_steps"] is not None and int(train_cfg["max_steps"]) > 0:
            max_steps = int(train_cfg["max_steps"])
            total_epochs_est = max_steps / steps_per_epoch
        elif "epochs" in train_cfg and train_cfg["epochs"] is not None and float(train_cfg["epochs"]) > 0:
            total_epochs_est = float(train_cfg["epochs"])
            max_steps = max(1, math.ceil(total_epochs_est * steps_per_epoch))
        else:
            max_steps = int(train_cfg.get("max_steps", 200))
            total_epochs_est = max_steps / steps_per_epoch

        total_epochs = max(1, math.ceil(total_epochs_est))
        checkpoint_every_epochs = max(1, int(self.config["training"].get("checkpoint_every_epochs", 1)))
        log_interval = max(1, int(self.config["training"].get("log_interval", self.config["training"].get("log_every", 1))))
        eval_interval = int(self.config["training"].get("eval_interval", 0))

        if len(self.raw_dataloader) == 0:
            raise ValueError("DataLoader пуст")

        step = 0
        optimizer_step = 0
        accumulated_loss = 0.0
        micro_steps_in_window = 0
        tokens_in_window = 0
        window_start_time = time.time()
        train_start_time = time.time()

        epochs_display = f"{total_epochs_est:.3f} эпохи" if total_epochs_est < 1.0 else f"~{total_epochs} эпох"
        print(
            f"=== СТАРТ ОБУЧЕНИЯ: {max_steps} батчей ({epochs_display}) | batch={self.config['training'].get('batch_size', 1)} | accum={accumulation_steps} ==="
        )

        try:
            for batch in cycle(self.dataloader):
                if step >= max_steps:
                    break
                step += 1
                micro_steps_in_window += 1
                epoch = (step - 1) // steps_per_epoch + 1

                if HAS_XLA:
                    input_ids, targets = batch
                else:
                    input_ids = batch[0].to(self.device, non_blocking=True)
                    targets = batch[1].to(self.device, non_blocking=True)

                tokens_in_window += input_ids.numel()

                # Forward with AMP
                if self.autocast_dtype is not None:
                    with autocast("cuda", dtype=self.autocast_dtype):
                        logits = self.model(input_ids)
                        loss = self.loss_fn(logits, targets) / accumulation_steps
                else:
                    logits = self.model(input_ids)
                    loss = self.loss_fn(logits, targets) / accumulation_steps

                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                accumulated_loss += loss.detach()

                if step % accumulation_steps == 0 or step == max_steps:
                    # Optimizer step & clipping
                    if self.scaler:
                        if self.max_grad_norm > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    elif HAS_XLA:
                        if self.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        xm.optimizer_step(self.optimizer)
                        xm.mark_step()
                    else:
                        if self.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1

                    # Logging interval
                    if optimizer_step % log_interval == 0 or step == max_steps:
                        elapsed = max(time.time() - window_start_time, 1e-4)
                        tok_per_sec = tokens_in_window / elapsed
                        loss_val = (
                            accumulated_loss.item()
                            if torch.is_tensor(accumulated_loss)
                            else accumulated_loss
                        ) * (accumulation_steps / max(1, micro_steps_in_window))
                        ppl = compute_perplexity(loss_val)
                        lr = self.scheduler.get_last_lr()[0]

                        # Calculate remaining time (ETA)
                        total_elapsed = time.time() - train_start_time
                        steps_done = step
                        remaining_steps = max(0, max_steps - steps_done)
                        sec_per_step = total_elapsed / max(1, steps_done)
                        eta_sec = remaining_steps * sec_per_step
                        if eta_sec < 60:
                            eta_str = f"{eta_sec:.0f}s"
                        elif eta_sec < 3600:
                            eta_str = f"{eta_sec / 60:.1f}m"
                        else:
                            eta_str = f"{eta_sec / 3600:.1f}h"

                        print(
                            f"epoch={epoch}/{total_epochs} step={step}/{max_steps} loss={loss_val:.4f} ppl={ppl:.2f} lr={lr:.6f} tok/s={tok_per_sec:.0f} eta={eta_str}",
                            flush=True,
                        )

                        accumulated_loss = 0.0
                        micro_steps_in_window = 0
                        tokens_in_window = 0
                        window_start_time = time.time()

                    # Periodic validation
                    if eval_interval > 0 and optimizer_step % eval_interval == 0 and self.val_dataloader:
                        val_loss, val_ppl = self.evaluate()
                        print(f"--> [Validation] step={step} val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}", flush=True)
                        if val_loss < self.best_loss:
                            self.best_loss = val_loss
                            self._save_checkpoint(self.checkpoint_dir / "best_model.pt", epoch, step, val_loss=val_loss)

                # Periodic checkpointing by epoch
                if step % steps_per_epoch == 0 and epoch % checkpoint_every_epochs == 0:
                    self._save_checkpoint(self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt", epoch, step)

        except KeyboardInterrupt:
            print("\n[Trainer] Обучение прервано пользователем! Сохраняем аварийный чекпойнт перед выходом...")
            self._save_checkpoint(self.checkpoint_dir / "emergency_interrupted.pt", epoch, step)
            self._save_checkpoint(self.checkpoint_dir / "latest.pt", epoch, step)
            return

        # Final checkpoint save
        self._save_checkpoint(self.checkpoint_dir / self.checkpoint_name, epoch, step)
        if self.checkpoint_name != "latest.pt":
            self._save_checkpoint(self.checkpoint_dir / "latest.pt", epoch, step)
        print(f"\n=== ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО: {step} шагов. Чекпойнты сохранены в {self.checkpoint_dir} ===")

    def _save_checkpoint(self, path: Path | str, epoch: int, step: int, val_loss: float | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw_model = getattr(self.model, "module", self.model)
        raw_model = getattr(raw_model, "_orig_mod", raw_model)
        model_state = {
            k: v.detach().cpu().half() if torch.is_floating_point(v) else v.detach().cpu()
            for k, v in raw_model.state_dict().items()
        }
        payload: dict[str, Any] = {
            "model_state": model_state,
            "config": self.config,
            "vocab_size": self.tokenizer.vocab_size,
            "epoch": epoch,
            "step": step,
            "best_loss": self.best_loss,
        }
        if val_loss is not None:
            payload["val_loss"] = val_loss
        if self.save_optimizer_state:
            payload["optimizer_state"] = self.optimizer.state_dict()
            payload["scheduler_state"] = self.scheduler.state_dict()
        torch.save(payload, path)
