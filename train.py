from __future__ import annotations

import argparse
from pathlib import Path

from config import build_default_config, load_config, save_config
from nexus.data_loader import build_dataloader, read_texts
from nexus.tokenizer import SimpleTokenizer
from nexus.trainer import Trainer
from nexus.utils import ensure_directory, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NexusAI")
    parser.add_argument("--config", type=str, default="configs/100M.yaml")
    parser.add_argument("--data", type=str, nargs="+", default=["datasets/raw"])
    parser.add_argument("--model-size", type=str, default="100M")
    parser.add_argument("--max-length", type=int, default=128)
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

    ensure_directory("datasets/tokenizer")
    texts = list(read_texts(args.data))

    tokenizer = SimpleTokenizer()
    tokenizer_path = Path("datasets/tokenizer/vocab.json")
    resume = config["training"].get("resume_from")
    if resume and tokenizer_path.exists():
        tokenizer = SimpleTokenizer.load(tokenizer_path)
    if texts:
        tokenizer.fit_from_texts(texts)
        if not resume:
            tokenizer.fit_from_texts(texts)
        tokenizer.save(tokenizer_path)
    else:
        tokenizer.save("datasets/tokenizer/vocab.json")

    batch_size = int(config["training"].get("batch_size", 8))
    dataloader = build_dataloader(texts, tokenizer, batch_size=batch_size, max_length=args.max_length)
    trainer = Trainer(config=config, dataloader=dataloader, tokenizer=tokenizer)
    if resume:
        trainer.load_checkpoint(resume)
    trainer.train()


if __name__ == "__main__":
    main()
