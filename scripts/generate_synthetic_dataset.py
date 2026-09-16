from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexus.synthetic import SyntheticDataGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic procedural training dataset")
    parser.add_argument("--output", type=str, default="datasets/processed/synthetic.jsonl", help="Output path for JSONL file")
    parser.add_argument("--samples", type=int, default=30000, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    generator = SyntheticDataGenerator(seed=args.seed)
    print(f"Генерация {args.samples} синтетических примеров в {out_path}...")

    with out_path.open("w", encoding="utf-8") as f:
        for i in range(args.samples):
            text = generator.generate_sample()
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            if (i + 1) % 10000 == 0 or (i + 1) == args.samples:
                print(f"  Сгенерировано: {i + 1}/{args.samples}")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Готово! Файл сохранён: {out_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
