from __future__ import annotations

from typing import Iterable
import torch

from .tokenizer import SimpleTokenizer


class TextDataset(torch.utils.data.Dataset):
    """Dataset wrapper for training over tokenized text with BOS/EOS tags and fast pre-tokenization."""

    def __init__(
        self,
        texts: Iterable[str],
        tokenizer: SimpleTokenizer,
        max_length: int = 128,
        pretokenize: bool = True,
        add_bos: bool = True,
    ) -> None:
        self.texts = [text for text in texts if isinstance(text, str) and text.strip()]
        self.tokenizer = tokenizer
        self.max_length = max(4, max_length)
        self.pretokenize = pretokenize
        self.add_bos = add_bos

        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        if self.pretokenize:
            bos_id = self.tokenizer.bos_token_id
            eos_id = self.tokenizer.eos_token_id
            # Space for bos + content + eos
            content_limit = self.max_length - (2 if self.add_bos else 1)
            for text in self.texts:
                content_tokens = self.tokenizer.encode(text)[:content_limit]
                token_ids = ([bos_id] if self.add_bos else []) + content_tokens + [eos_id]
                if len(token_ids) < 2:
                    token_ids = [bos_id, eos_id]
                inp = torch.tensor(token_ids[:-1], dtype=torch.long)
                tgt = torch.tensor(token_ids[1:], dtype=torch.long)
                self.samples.append((inp, tgt))

    def __len__(self) -> int:
        return len(self.samples) if self.pretokenize else len(self.texts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pretokenize:
            return self.samples[index]

        text = self.texts[index]
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        content_limit = self.max_length - (2 if self.add_bos else 1)
        content_tokens = self.tokenizer.encode(text)[:content_limit]
        token_ids = ([bos_id] if self.add_bos else []) + content_tokens + [eos_id]

        if len(token_ids) < 2:
            token_ids = [bos_id, eos_id]

        inp = torch.tensor(token_ids[:-1], dtype=torch.long)
        tgt = torch.tensor(token_ids[1:], dtype=torch.long)
        return inp, tgt
