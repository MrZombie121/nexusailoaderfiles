from __future__ import annotations

import argparse
import sys
from pathlib import Path
import torch

from config import load_config
from nexus.generation import generate_stream, generate_text
from nexus.model import NexusModel
from nexus.tokenizer import SimpleTokenizer
from nexus.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run advanced inference with NexusAI")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt", help="Path to .pt checkpoint")
    parser.add_argument("--prompt", type=str, default="Привет! Кто ты?", help="Input prompt")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0 for greedy)")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K filtering")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-P (Nucleus) filtering")
    parser.add_argument("--min-p", type=float, default=0.05, help="Min-P filtering")
    parser.add_argument("--repetition-penalty", type=float, default=1.15, help="Repetition penalty")
    parser.add_argument("--stream", action="store_true", help="Stream tokens to stdout")
    parser.add_argument("--config", type=str, default=None, help="Optional config path override")
    return parser.parse_args()


def load_model_and_tokenizer(checkpoint_path: str, config_override: str | None = None) -> tuple[NexusModel, SimpleTokenizer]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Resolve tokenizer
    vocab_candidates = [
        Path(checkpoint_path).parent / "vocab.json",
        Path("datasets/tokenizer/vocab.json"),
    ]
    vocab_path = next((p for p in vocab_candidates if p.exists()), None)
    if not vocab_path:
        raise FileNotFoundError("Словарь vocab.json не найден ни рядом с чекпойнтом, ни в datasets/tokenizer/vocab.json")

    tokenizer = SimpleTokenizer.load(vocab_path)
    config = checkpoint.get("config") or (load_config(config_override) if config_override else None)
    if not config:
        raise ValueError("В чекпойнте отсутствует config. Укажите --config.")

    m = config["model"]
    model = NexusModel(
        vocab_size=tokenizer.vocab_size,
        hidden_size=int(m["hidden_size"]),
        intermediate_size=int(m["intermediate_size"]),
        num_layers=int(m["num_layers"]),
        num_heads=int(m["num_heads"]),
        num_key_value_heads=int(m.get("num_key_value_heads", m["num_heads"])),
        max_position_embeddings=int(m.get("max_position_embeddings", 2048)),
        dropout=float(m.get("dropout", 0.0)),
    )

    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, tokenizer


def main() -> None:
    args = parse_args()
    set_seed(42)

    model, tokenizer = load_model_and_tokenizer(args.checkpoint, args.config)

    print(f"\n--- Промпт: {args.prompt} ---\nОтвет:")

    if args.stream:
        for chunk in generate_stream(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
        ):
            # Print latest token chunk directly
            sys.stdout.write(chunk[len(sys.stdout.getvalue()) if hasattr(sys.stdout, 'getvalue') else 0:])
            sys.stdout.flush()
        print()
    else:
        answer = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            return_full_text=False,
        )
        print(answer)


if __name__ == "__main__":
    main()
