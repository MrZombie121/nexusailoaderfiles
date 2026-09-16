from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Any

try:
    import sentencepiece as spm
    HAS_SPM = True
except ImportError:
    HAS_SPM = False


class BPETokenizer:
    """Byte-Pair Encoding (BPE) subword tokenizer for Russian, English, Code, and Mathematics.
    
    Supports SentencePiece acceleration when available with a pure-Python fallback.
    Default vocabulary size is 8,192 tokens.
    """

    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        model_path: str | Path | None = None,
    ) -> None:
        self.vocab = vocab or {}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.sp_processor: Any = None

        if model_path is not None and HAS_SPM and Path(model_path).exists():
            try:
                self.sp_processor = spm.SentencePieceProcessor(model_file=str(model_path))
                if not self.vocab:
                    self.vocab = {self.sp_processor.id_to_piece(i): i for i in range(self.sp_processor.get_piece_size())}
                    self.inverse_vocab = {v: k for k, v in self.vocab.items()}
            except Exception:
                self.sp_processor = None

    @property
    def vocab_size(self) -> int:
        if self.sp_processor is not None:
            return self.sp_processor.get_piece_size()
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

    def encode(self, text: str) -> list[int]:
        """Encodes text to a list of BPE token IDs."""
        if not text:
            return []

        # Fast path with SentencePiece C++ engine
        if self.sp_processor is not None:
            return self.sp_processor.encode(text)

        # Pure Python fallback using longest subword matching
        pieces = []
        words = text.split(" ")
        for i, word in enumerate(words):
            prefix = " " if i > 0 or text.startswith(" ") else ""
            w = prefix + word
            start = 0
            while start < len(w):
                matched = False
                for end in range(len(w), start, -1):
                    sub = w[start:end]
                    if sub in self.vocab:
                        pieces.append(self.vocab[sub])
                        start = end
                        matched = True
                        break
                if not matched:
                    # Byte fallback
                    char_bytes = w[start].encode("utf-8")
                    for b in char_bytes:
                        byte_token = f"<0x{b:02X}>"
                        pieces.append(self.vocab.get(byte_token, self.unk_token_id))
                    start += 1

        return pieces

    def decode(self, ids: Iterable[int]) -> str:
        """Decodes token IDs back to a readable string."""
        if self.sp_processor is not None:
            clean_ids = [
                int(tid)
                for tid in ids
                if int(tid) not in {self.bos_token_id, self.eos_token_id, self.pad_token_id}
            ]
            return self.sp_processor.decode(clean_ids)

        # Fallback decode
        parts = []
        byte_buffer = bytearray()

        for token_id in ids:
            tid = int(token_id)
            if tid in {self.bos_token_id, self.eos_token_id, self.pad_token_id}:
                continue
            piece = self.inverse_vocab.get(tid, "")
            if not piece:
                continue

            # Byte fallback check e.g. <0xAB>
            if piece.startswith("<0x") and piece.endswith(">") and len(piece) == 6:
                try:
                    byte_val = int(piece[3:5], 16)
                    byte_buffer.append(byte_val)
                    continue
                except ValueError:
                    pass

            if byte_buffer:
                parts.append(byte_buffer.decode("utf-8", errors="replace"))
                byte_buffer.clear()

            parts.append(piece.replace(" ", " "))

        if byte_buffer:
            parts.append(byte_buffer.decode("utf-8", errors="replace"))

        return "".join(parts).lstrip(" ")

    def fit_from_texts(self, texts: Iterable[str], vocab_size: int = 8192) -> None:
        """Trains SentencePiece BPE model from an iterable of texts."""
        if not HAS_SPM:
            raise ImportError("sentencepiece is required to train BPE vocabulary. Install with: pip install sentencepiece")

        import tempfile
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
            for text in texts:
                clean = text.strip().replace("\n", " ")
                if clean:
                    f.write(clean + "\n")
            tmp_path = f.name

        model_prefix = tmp_path + "_spm"
        spm.SentencePieceTrainer.train(
            input=tmp_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            byte_fallback=True,
            hard_vocab_limit=False,
            character_coverage=0.9995,
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3,
            pad_piece="<pad>",
            bos_piece="<bos>",
            eos_piece="<eos>",
            unk_piece="<unk>",
        )

        model_file = model_prefix + ".model"
        self.sp_processor = spm.SentencePieceProcessor(model_file=model_file)
        self.vocab = {self.sp_processor.id_to_piece(i): i for i in range(self.sp_processor.get_piece_size())}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def save(self, path: str | Path) -> None:
        """Saves vocab.json and optionally tokenizer.model."""
        target = Path(path)
        target_dir = target if target.is_dir() else target.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        vocab_path = target if target.suffix == ".json" else target_dir / "vocab.json"
        with vocab_path.open("w", encoding="utf-8") as handle:
            json.dump(self.vocab, handle, ensure_ascii=False, indent=2)

        # If sentencepiece model is present, save copy
        if self.sp_processor is not None and hasattr(self.sp_processor, "serialized_model_proto"):
            model_path = target_dir / "tokenizer.model"
            with model_path.open("wb") as mf:
                mf.write(self.sp_processor.serialized_model_proto())

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Loads BPE tokenizer from vocab.json or tokenizer.model."""
        target = Path(path)
        target_dir = target if target.is_dir() else target.parent

        model_file = target if target.name.endswith(".model") else target_dir / "tokenizer.model"
        vocab_file = target if target.name.endswith(".json") else target_dir / "vocab.json"

        vocab: dict[str, int] = {}
        if vocab_file.exists():
            with vocab_file.open("r", encoding="utf-8") as handle:
                vocab = json.load(handle)

        spm_path = model_file if model_file.exists() else None
        return cls(vocab=vocab, model_path=spm_path)


# Backward-compatible alias for existing code
SimpleTokenizer = BPETokenizer
