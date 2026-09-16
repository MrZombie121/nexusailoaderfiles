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
    num_workers: int = 0,
    pin_memory: bool | None = None,
) -> DataLoader:
    """Build an optimized PyTorch DataLoader for tokenized text sequences."""
    dataset = TextDataset(texts, tokenizer, max_length=max_length)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )


def build_train_val_dataloaders(
    texts: list[str],
    tokenizer: SimpleTokenizer,
    batch_size: int,
    max_length: int = 128,
    val_ratio: float = 0.05,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader | None]:
    """Splits texts into train and validation sets and builds DataLoaders for both."""
    if val_ratio <= 0.0 or len(texts) < 20:
        train_loader = build_dataloader(texts, tokenizer, batch_size, max_length, shuffle=True, num_workers=num_workers)
        return train_loader, None

    val_size = max(1, int(len(texts) * val_ratio))
    train_texts = texts[:-val_size]
    val_texts = texts[-val_size:]

    train_loader = build_dataloader(train_texts, tokenizer, batch_size, max_length, shuffle=True, num_workers=num_workers)
    val_loader = build_dataloader(val_texts, tokenizer, batch_size, max_length, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def build_synthetic_dataloaders(
    tokenizer: SimpleTokenizer,
    batch_size: int,
    max_length: int = 128,
    train_samples: int = 100000,
    val_samples: int = 2000,
    num_workers: int = 0,
    languages: list[str] | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Builds purely procedural synthetic DataLoaders generating billions of tokens on-the-fly (UKR + RUS + ENG)."""
    from .synthetic import ProceduralDataset
    langs = languages or ["uk", "ru", "en"]
    train_ds = ProceduralDataset(tokenizer=tokenizer, total_samples=train_samples, max_length=max_length, seed=42, languages=langs)
    val_ds = ProceduralDataset(tokenizer=tokenizer, total_samples=val_samples, max_length=max_length, seed=9999, languages=langs)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate, pin_memory=pin_memory, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate, pin_memory=pin_memory, num_workers=num_workers)
    return train_loader, val_loader


def read_texts(paths: Iterable[str | Path]) -> Iterator[str]:
    """Read text from txt/md/json/jsonl and parquet files recursively, or stream procedurally."""
    for raw_path in paths:
        if str(raw_path).lower() == "synthetic":
            from .synthetic import SyntheticDataGenerator
            gen = SyntheticDataGenerator(seed=42)
            for _ in range(50000):
                yield gen.generate_sample()
            continue

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
