from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Any

import torch
from torch.utils.data import DataLoader

from .dataset import TextDataset
from .tokenizer import SimpleTokenizer


def get_data_loader(
    data_dir: str | Path,
    batch_size: int,
    max_length: int = 128,
) -> DataLoader:
    """Helper to read data and build a DataLoader."""
    tokenizer = SimpleTokenizer.load(Path(data_dir).parent / "tokenizer" / "vocab.json")
    texts = read_texts([data_dir])
    return build_dataloader(texts, tokenizer, batch_size=batch_size, max_length=max_length)


def build_dataloader(
    texts: Iterable[str],
    tokenizer: SimpleTokenizer,
    batch_size: int,
    max_length: int = 128,
    shuffle: bool = True,
) -> DataLoader:
    """Build a PyTorch DataLoader for tokenized text sequences."""
    dataset = TextDataset(texts, tokenizer, max_length=max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate)


def read_texts(paths: Iterable[str | Path]) -> Iterator[str]:
    """Read text from txt/md/json/jsonl and parquet files recursively."""
    for raw_path in paths:
        path = Path(raw_path)
        files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
        for file_path in files:
            suffix = file_path.suffix.lower()
            if suffix in {".txt", ".md"}:
                yield file_path.read_text(encoding="utf-8", errors="ignore")
            elif suffix in {".json", ".jsonl"}:
                yield from _json_texts(file_path)
            elif suffix == ".parquet":
                yield from _parquet_texts(file_path)


def _json_texts(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as handle:
        if path.suffix.lower() == ".jsonl":
            values = (json.loads(line) for line in handle if line.strip())
        else:
            values = json.load(handle)
            values = values if isinstance(values, list) else [values]
        for value in values:
            text = _value_to_text(value)
            if text:
                yield text


def _parquet_texts(path: Path) -> Iterator[str]:
    import pandas as pd
    for value in pd.read_parquet(path).to_dict(orient="records"):
        text = _value_to_text(value)
        if text:
            yield text


def _value_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, dict):
        return str(value).strip()
    if isinstance(value.get("text"), str):
        return value["text"].strip()
    prompt = value.get("prompt", value.get("instruction", value.get("question", "")))
    answer = value.get("response", value.get("output", value.get("answer", "")))
    if prompt or answer:
        return f"{prompt}\n{answer}".strip()
    return json.dumps(value, ensure_ascii=False)


def _collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return input_ids, labels
