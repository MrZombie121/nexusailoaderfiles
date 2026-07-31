from __future__ import annotations

from typing import Iterable

import torch

from .tokenizer import SimpleTokenizer


class TextDataset(torch.utils.data.Dataset):
    """Dataset wrapper for training over tokenized text."""

    def __init__(self, texts: Iterable[str], tokenizer: SimpleTokenizer, max_length: int = 128) -> None:
        self.texts = [text for text in texts if isinstance(text, str) and text.strip()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        text = self.texts[index]
        token_ids = self.tokenizer.encode(text)
        token_ids = token_ids[: self.max_length - 1]
        token_ids.append(self.tokenizer.eos_token_id)

        if len(token_ids) < 2:
            token_ids = [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id]

        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        targets = torch.tensor(token_ids[1:], dtype=torch.long)
        return input_ids, targets
