from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexus.data_loader import read_texts

# Ukrainian training text seeds covering core vocabulary, grammar, and domains
UKRAINIAN_TEXTS = [
    # Загальні фрази та діалоги
    "Привіт! Як твої справи? Чим я можу тобі сьогодні допомогти?",
    "Доброго дня! Я штучний інтелект NexusAI, створений для відповідей на запитання, написання коду та розв'язання задач.",
    "Як справи? Все чудово, готовий до роботи та навчання.",
    "Будь ласка, розкажи докладніше про цей проєкт та його особливості.",
    "Дякую за допомогу! Це дуже корисна та вичерпна інформація.",
    "Що ти думаєш про майбутнє технологій та генеративного штучного інтелекту?",
    "Поясни це простими словами, щоб зрозуміла навіть дитина.",
    "Столиця України — місто Київ, яке має багатовікову історію та культуру.",
    "Українська мова — солов'їна, милозвучна та надзвичайно багата мова.",
    "Київська Русь, гетьманщина, козацтво та сучасна незалежна Україна.",
    "Карпати, Чорне море, Дніпро, степи та ліси України.",
    
    # Математика та логіка
    "Розв'яжи лінійне рівняння: 2x + 15 = 45. Відповідь: x = 15.",
    "Обчисли значення математичного виразу: (34 + 56) * 12 / 4.",
    "У кошику було 25 яблук і 15 груш. Скільки всього фруктів у кошику?",
    "Теорема Піфагора стверджує: у прямокутному трикутнику квадрат гіпотенузи дорівнює сумі квадратів катетів: a^2 + b^2 = c^2.",
    "Швидкість світла у вакуумі становить близько 300 000 кілометрів на секунду.",
    "Ймовірність випадкової події обчислюється як відношення сприятливих випадків до загальної кількості.",
    "Знайди похідну функції f(x) = x^3 + 4x^2 - 7x + 10.",
    "Інтеграл від швидкості за часом дає пройдену відстань.",
    
    # Програмування та алгоритми
    "Напиши функцію на мові програмування Python для знаходження факторіала числа.",
    "def factorial(n: int) -> int:\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
    "Алгоритм бінарного пошуку знаходить елемент у відсортованому масиві за логарифмічний час O(log n).",
    "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
    "Структури даних: масиви, зв'язані списки, стеки, черги, дерева, графи та хеш-таблиці.",
    "Об'єктно-орієнтоване програмування базується на інкапсуляції, успадкуванні та поліморфізмі.",
    "Напиши SQL-запит для отримання списку користувачів, які зареєструвалися за останній місяць.",
    "SELECT id, name, email, created_at FROM users WHERE created_at >= NOW() - INTERVAL '30 days' ORDER BY created_at DESC;",
    "База даних, індекси, первинний ключ, зовнішній ключ, транзакції та рівні ізоляції ACID.",
    "Git — це розподілена система контролю версій для командної розробки програмного забезпечення.",
    
    # Наука, комп'ютери та технології
    "Центральний процесор (CPU) виконує інструкції програми, а графічний процесор (GPU) оптимізований для масивних паралельних обчислень.",
    "Оперативна пам'ять RAM забезпечує швидкий доступ до даних, а SSD забезпечує енергонезалежне постійне зберігання.",
    "Протокол TCP гарантує доставку пакетів, тоді як UDP забезпечує мінімальну затримку без підтвердження.",
    "Модель трансформера використовує механізм уваги (self-attention) для аналізу контексту в послідовності токенів.",
    "Кеш ключів та значень (KV-Cache) дозволяє прискорити генерацію тексту, уникаючи повторних обчислень для попередніх токенів.",
    "Навчання нейронної мережі відбувається за допомогою алгоритму зворотного поширення помилки та градієнтного спуску.",
    
    # Окремі специфічні українські слова та літери
    "і ї є ґ І Ї Є Ґ об'єкт під'їзд зв'язок пам'ять м'яч розв'язання комп'ютер бур'ян",
    "користувач асистент запитання відповідь пояснення приклад задача завдання обчислення",
    "програма функція змінна значення результат помилка виконання перевірка умова цикл",
    "перший другий третій четвертий п'ятий шостий сьомий восьмий дев'ятий десятий",
    "бути мати робити знати думати казати бачити хотіти могти йти працювати вчитися",
]


def build_trilingual_tokenizer(
    output_dir: str | Path = "datasets/tokenizer",
    vocab_size: int = 16000,
) -> None:
    import sentencepiece as spm

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== ПОБУДОВА ТРИМОВНОГО BPE ТОКЕНІЗАТОРА (УКР + РУС + АНГЛ, {vocab_size} токенів) ===")

    corpus_fd, corpus_path = tempfile.mkstemp(suffix=".txt", text=True)
    total_lines = 0

    with open(corpus_fd, "w", encoding="utf-8") as out_f:
        # 1. Украинские тексты (мультиплицируем для достаточной частотности BPE)
        for _ in range(25):
            for line in UKRAINIAN_TEXTS:
                clean = line.strip().replace("\n", " ")
                if clean:
                    out_f.write(clean + "\n")
                    total_lines += 1

        # 2. Реальные датасеты из datasets/processed (русский, английский, код, математика)
        data_dir = Path("datasets/processed")
        if data_dir.exists():
            for text in read_texts([data_dir]):
                clean = text.strip().replace("\n", " ")
                if len(clean) > 5:
                    out_f.write(clean + "\n")
                    total_lines += 1

        # 3. Синтетический процедурный генератор (математика, код, диалоги)
        from nexus.synthetic import SyntheticDataGenerator
        gen = SyntheticDataGenerator(seed=42)
        print("Генерація процедурних зразків для повного покриття трьох мов...")
        for _ in range(15000):
            sample = gen.generate_sample().replace("\n", " ")
            out_f.write(sample + "\n")
            total_lines += 1

    print(f"Підготовлено {total_lines} речень для навчання BPE.")

    model_prefix = str(out_dir / "spm_trilingual")

    print(f"Запуск SentencePiece BPE (розмір словника: {vocab_size}, Byte-fallback: True)...")
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

    final_model_path = out_dir / "tokenizer.model"
    temp_model_path = Path(model_prefix + ".model")
    if temp_model_path.exists():
        if final_model_path.exists():
            final_model_path.unlink()
        temp_model_path.rename(final_model_path)

    temp_vocab = Path(model_prefix + ".vocab")
    if temp_vocab.exists():
        temp_vocab.unlink()
    try:
        os.remove(corpus_path)
    except Exception:
        pass

    sp = spm.SentencePieceProcessor(model_file=str(final_model_path))
    actual_size = sp.get_piece_size()
    print(f"Токенізатор успішно навчений! Фактичний розмір словника: {actual_size}")

    # Build vocab.json
    vocab: dict[str, int] = {}
    for i in range(actual_size):
        piece = sp.id_to_piece(i)
        vocab[piece] = i

    vocab_json_path = out_dir / "vocab.json"
    with vocab_json_path.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    config = {
        "tokenizer_type": "BPE",
        "languages": ["uk", "ru", "en"],
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
    with (out_dir / "tokenizer_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"Збережено:")
    print(f"  - {final_model_path} ({final_model_path.stat().st_size / 1024:.1f} KB)")
    print(f"  - {vocab_json_path} ({vocab_json_path.stat().st_size / 1024:.1f} KB)")

    # Тестирование на 3-х языках
    print("\n=== ПЕРЕВІРКА ТОКЕНІЗАЦІЇ НА ТРЬОХ МОВАХ (УКР, РУС, АНГЛ) ===")
    test_cases = [
        ("УКР", "Привіт! Як справи? Розв'яжи рівняння 3x + 12 = 45."),
        ("РУС", "Привет! Как дела? Реши уравнение 3x + 12 = 45."),
        ("АНГЛ", "Hello! How are you? Solve equation 3x + 12 = 45."),
        ("КОД", "def factorial(n): return 1 if n <= 1 else n * factorial(n - 1)")
    ]

    for lang_tag, text in test_cases:
        ids = sp.encode(text)
        decoded = sp.decode(ids)
        tokens = [sp.id_to_piece(tid) for tid in ids[:10]]
        print(f"[{lang_tag}] Текст: '{text[:40]}...' ({len(text)} симв.) -> {len(ids)} токенів")
        print(f"      Перші токени: {tokens}")
        assert decoded == text, f"Помилка декодування для {lang_tag}: {decoded} != {text}"

    # Проверка специфических украинских символов
    ukr_special_chars = ["і", "ї", "є", "ґ", "І", "Ї", "Є", "Ґ", "’", "об'єкт", "зв'язок"]
    for ch in ukr_special_chars:
        ch_ids = sp.encode(ch)
        ch_dec = sp.decode(ch_ids)
        assert ch_dec == ch, f"Помилка декодування символу '{ch}': {ch_dec}"
    print("Усі специфічні українські символи (і, ї, є, ґ, апостроф) декодуються бездоганно!")
    print("Тримовний токенізатор повністю готовий до роботи!")


if __name__ == "__main__":
    build_trilingual_tokenizer(vocab_size=16000)
