from __future__ import annotations

import argparse
import torch
from config import load_config

from nexus.generation import generate_text
from nexus.model import NexusModel
from nexus.tokenizer import SimpleTokenizer
from nexus.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with NexusAI")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt")
    parser.add_argument("--prompt", type=str, default="NexusAI")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(42)

    tokenizer = SimpleTokenizer.load("datasets/tokenizer/vocab.json")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config") or (load_config(args.config) if args.config else None)
    kwargs = {"vocab_size": tokenizer.vocab_size}
    if config:
        kwargs.update({key: config["model"][key] for key in ("hidden_size", "intermediate_size", "num_layers", "num_heads")})
        kwargs["max_position_embeddings"] = config["model"].get("max_position_embeddings", 2048)
        kwargs["dropout"] = config["model"].get("dropout", 0.1)
    model = NexusModel(**kwargs)
    model.load_state_dict(checkpoint.get("model_state", checkpoint))
    model.eval()

    generated = generate_text(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens)
    print(generated)


if __name__ == "__main__":
    main()
