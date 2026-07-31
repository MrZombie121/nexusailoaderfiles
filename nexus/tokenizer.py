from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable


class SimpleTokenizer:
    """A small tokenizer for prototype training and decoding."""

    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"

    def __init__(self, vocab: dict[str, int] | None = None) -> None:
        self.vocab = vocab or {}
        self.inverse_vocab = {value: key for key, value in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def bos_token_id(self) -> int:
        return self.vocab.get(self.bos_token, 1)

    @property
    def eos_token_id(self) -> int:
        return self.vocab.get(self.eos_token, 2)

    @property
    def pad_token_id(self) -> int:
        return self.vocab.get(self.pad_token, 0)

    @property
    def unk_token_id(self) -> int:
        return self.vocab.get(self.unk_token, 3)

    def fit_from_texts(self, texts: Iterable[str]) -> None:
        counter = Counter()
        for text in texts:
            counter.update(text)

        tokens = [self.pad_token, self.bos_token, self.eos_token, self.unk_token] + sorted(set(counter.keys()))
        self.vocab = {token: idx for idx, token in enumerate(tokens)}
        self.inverse_vocab = {value: key for key, value in self.vocab.items()}

    def encode(self, text: str) -> list[int]:
        return [self.vocab.get(char, self.unk_token_id) for char in text]

    def decode(self, ids: Iterable[int]) -> str:
        decoded = []
        for token_id in ids:
            token = self.inverse_vocab.get(token_id, self.unk_token)
            if token in {self.bos_token, self.eos_token, self.pad_token}:
                continue
            decoded.append(token)
        return "".join(decoded)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.vocab, handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "SimpleTokenizer":
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            vocab = json.load(handle)
        return cls(vocab=vocab)
