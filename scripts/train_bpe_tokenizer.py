from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexus.data_loader import read_texts
from nexus.synthetic import SyntheticDataGenerator


def train_bpe(
    data_paths: list[str | Path],
    output_dir: str | Path = "datasets/tokenizer",
    vocab_size: int = 8192,
    include_synthetic: bool = True,
    synthetic_samples: int = 20000,
) -> None:
    import sentencepiece as spm

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== ОБУЧЕНИЕ BPE ТОКЕНИЗАТОРА (Размер словаря: {vocab_size}) ===")
    print("Сбор обучающих текстов из:", data_paths)

    corpus_fd, corpus_path = tempfile.mkstemp(suffix=".txt", text=True)
    total_lines = 0

    with open(corpus_fd, "w", encoding="utf-8") as out_f:
        # 1. Read real datasets
        for text in read_texts(data_paths):
            clean = text.strip().replace("\n", " ")
            if len(clean) > 5:
                out_f.write(clean + "\n")
                total_lines += 1

        # 2. Include synthetic domain patterns (math, code, dialogs)
        if include_synthetic:
            print(f"Добавление {synthetic_samples} процедурных примеров для расширения покрытия кода и математики...")
            gen = SyntheticDataGenerator(seed=42)
            for _ in range(synthetic_samples):
                sample = gen.generate_sample().replace("\n", " ")
                out_f.write(sample + "\n")
                total_lines += 1

    print(f"Всего подготовлено предложений: {total_lines}")

    model_prefix = str(out_dir / "spm_temp")

    print(f"Запуск алгоритма SentencePiece BPE (целевой размер: {vocab_size})...")
    spm.SentencePieceTrainer.train(
        input=corpus_path,
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

    # Rename model file to standard name
    final_model_path = out_dir / "tokenizer.model"
    temp_model_path = Path(model_prefix + ".model")
    if temp_model_path.exists():
        if final_model_path.exists():
            final_model_path.unlink()
        temp_model_path.rename(final_model_path)

    # Clean up temp files
    temp_vocab_path = Path(model_prefix + ".vocab")
    if temp_vocab_path.exists():
        temp_vocab_path.unlink()
    try:
        os.remove(corpus_path)
    except Exception:
        pass

    # Load trained model to build vocab.json
    sp = spm.SentencePieceProcessor(model_file=str(final_model_path))
    actual_size = sp.get_piece_size()
    print(f"BPE успешно обучен! Итоговый размер словаря: {actual_size}")

    # Build vocab.json mapping: token -> id
    vocab: dict[str, int] = {}
    for i in range(actual_size):
        piece = sp.id_to_piece(i)
        vocab[piece] = i

    vocab_json_path = out_dir / "vocab.json"
    with vocab_json_path.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    # Build config
    config = {
        "tokenizer_type": "BPE",
        "vocab_size": actual_size,
        "byte_fallback": True,
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "unk_token_id": 3,
    }
    config_path = out_dir / "tokenizer_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"Сохранены файлы токенизатора:")
    print(f"  - {final_model_path} ({final_model_path.stat().st_size / 1024:.1f} KB)")
    print(f"  - {vocab_json_path} ({vocab_json_path.stat().st_size / 1024:.1f} KB)")
    print(f"  - {config_path}")

    # Test encoding and decoding
    test_cases = [
        "Привет, как твои дела?",
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "Задача: Реши уравнение 2x + 15 = 45. Ответ: x = 15.",
        "Hello! I am NexusAI, an intelligent assistant for coding and math."
    ]
    print("\n=== ТЕСТ ТОКЕНИЗАЦИИ ===")
    for text in test_cases:
        ids = sp.encode(text)
        decoded = sp.decode(ids)
        print(f"Текст ({len(text)} симв.) -> {len(ids)} токенов")
        print(f"  IDs: {ids[:10]}...")
        assert decoded == text, f"Mismatch: {decoded} != {text}"
    print("Все тесты кодирования и декодирования успешно пройдены!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SentencePiece BPE tokenizer for NexusAI")
    parser.add_argument("--data", nargs="+", default=["datasets/processed"], help="Data paths to train on")
    parser.add_argument("--output", type=str, default="datasets/tokenizer", help="Output directory")
    parser.add_argument("--vocab-size", type=int, default=8192, help="Target vocabulary size (8192 to 16384)")
    parser.add_argument("--no-synthetic", action="store_true", help="Do not augment with synthetic samples")
    args = parser.parse_args()

    train_bpe(
        data_paths=args.data,
        output_dir=args.output,
        vocab_size=args.vocab_size,
        include_synthetic=not args.no_synthetic,
    )


if __name__ == "__main__":
    main()
