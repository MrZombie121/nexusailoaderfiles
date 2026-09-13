from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import build_default_config, load_config, save_config
from nexus.data_loader import build_train_val_dataloaders, read_texts
from nexus.tokenizer import SimpleTokenizer
from nexus.trainer import Trainer
from nexus.utils import ensure_directory, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NexusAI with high performance")
    parser.add_argument("--config", type=str, default="configs/100M.yaml", help="Path to YAML config")
    parser.add_argument("--data", type=str, nargs="+", default=["datasets/processed"], help="Data file or folder paths")
    parser.add_argument("--model-size", type=str, default="100M", help="Default model size preset")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--epochs", type=float, default=None, help="Override training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--val-ratio", type=float, default=0.03, help="Fraction of data for validation")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from checkpoint path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(42)

    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(config_path)
    else:
        config = build_default_config(args.model_size)
        save_config(config_path, config)

    # CLI parameter overrides
    if "training" not in config:
        config["training"] = {}
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.resume is not None:
        config["training"]["resume_from"] = args.resume

    ensure_directory("datasets/tokenizer")
    texts = list(read_texts(args.data))
    if not texts:
        print(f"Предупреждение: не найдено текстов в {args.data}. Ищу в datasets/raw...")
        texts = list(read_texts(["datasets/raw"]))

    tokenizer = SimpleTokenizer()
    tokenizer_path = Path("datasets/tokenizer/vocab.json")
    resume = config["training"].get("resume_from")

    if resume and Path(resume).parent.joinpath("vocab.json").exists():
        tokenizer = SimpleTokenizer.load(Path(resume).parent.joinpath("vocab.json"))
    elif tokenizer_path.exists():
        tokenizer = SimpleTokenizer.load(tokenizer_path)
    elif texts:
        tokenizer.fit_from_texts(texts)
        tokenizer.save(tokenizer_path)
    else:
        tokenizer.save(tokenizer_path)

    batch_size = int(config["training"].get("batch_size", 8))
    max_length = int(config["training"].get("max_seq_length", config["training"].get("max_length", args.max_length)))
    if args.max_length != 128:
        max_length = args.max_length

    num_workers = int(config["training"].get("num_workers", 2 if sys.platform != "win32" else 0))
    train_loader, val_loader = build_train_val_dataloaders(
        texts=texts,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        val_ratio=args.val_ratio,
        num_workers=num_workers,
    )

    trainer = Trainer(
        config=config,
        dataloader=train_loader,
        tokenizer=tokenizer,
        val_dataloader=val_loader,
    )

    if resume:
        print(f"Загрузка контрольной точки: {resume}")
        trainer.load_checkpoint(resume)

    trainer.train()


if __name__ == "__main__":
    main()
