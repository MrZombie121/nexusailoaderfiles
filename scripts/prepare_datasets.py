"""Download the selected Hugging Face datasets and save them as JSONL.

The conversion intentionally follows the simple and reliable pattern:
``load_dataset(...)[split].to_json(..., force_ascii=False)``.
Use ``--max-rows`` when testing or when a dataset should be capped.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from datasets import Dataset, concatenate_datasets, load_dataset


OUTPUTS = {
    "code": "evilfreelancer/golang-en-ru",
    "math": "lighteval/QazUNTv2",
    "conversation": "SonexaAI/ru_eng-dataset",
}


def _text(row: dict[str, Any], category: str) -> str:
    if category == "code":
        # The dataset is a parallel Go corpus; support its common field names.
        left = row.get("english", row.get("en", row.get("source", "")))
        right = row.get("russian", row.get("ru", row.get("target", "")))
        return f"English: {left}\nРусский: {right}".strip()
    if category == "math":
        question = row.get("question", row.get("problem", ""))
        solution = row.get("solution", row.get("full_answer", row.get("answer", "")))
        options = row.get("options", row.get("choices", ""))
        return f"Задача: {question}\nВарианты: {options}\nРешение: {solution}".strip()
    prompt = row.get("prompt", row.get("instruction", ""))
    response = row.get("response", row.get("output", ""))
    return f"Пользователь: {prompt}\nАссистент: {response}".strip()


def _load(category: str) -> Dataset:
    dataset_id = OUTPUTS[category]
    if category == "math":
        # QazUNTv2 exposes separate English and Russian configurations.
        parts = [load_dataset(dataset_id, name=lang, split="train") for lang in ("en", "ru")]
        return concatenate_datasets(parts)
    return load_dataset(dataset_id, split="train")


def prepare(category: str, output_dir: Path, max_rows: int | None) -> Path:
    dataset = _load(category)
    if max_rows is not None:
        dataset = dataset.select(range(min(max_rows, len(dataset))))
    dataset = dataset.map(lambda row: {"text": _text(row, category)})
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{category}.jsonl"
    dataset.to_json(output_path, force_ascii=False)
    print(f"{category}: {len(dataset)} rows -> {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download bilingual HF datasets and convert them to JSONL")
    parser.add_argument("--output", default="datasets/processed", help="Output directory")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap per output dataset")
    parser.add_argument("--only", choices=list(OUTPUTS), nargs="+", help="Prepare selected datasets only")
    args = parser.parse_args()
    for category in args.only or OUTPUTS:
        prepare(category, Path(args.output), args.max_rows)


if __name__ == "__main__":
    main()
