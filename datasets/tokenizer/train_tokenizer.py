import sys
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors

# Настройка кодировки вывода для корректного отображения спецсимволов и байтов в терминале
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# =============================================================================
# 1. КОНСТАНТЫ
# =============================================================================

# Целевой размер словаря для русско-английского корпуса (16k токенов)
VOCAB_SIZE = 16000

# Минимальная частота встречаемости пары символов для объединения в BPE
MIN_FREQUENCY = 2

# Специальные токены модели в строгом порядке:
# <pad> = 0, <bos> = 1, <eos> = 2, <unk> = 3
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

# Папка для сохранения итогового словаря (папка текущего скрипта)
OUTPUT_DIR = Path(__file__).parent


# =============================================================================
# 2. ГЕНЕРАТОР ТЕКСТА (ЗАГЛУШКА)
# =============================================================================

def text_iterator():
    """
    Генератор обучающих текстов на русском и английском языках.
    Служит потоковым источником данных для метода train_from_iterator.
    """
    # -------------------------------------------------------------------------
    # TODO: замени на свой генератор
    # Пример подключения внешнего генератора:
    #   from my_dataset_generator import get_infinite_stream
    #   for text in get_infinite_stream():
    #       yield text
    # -------------------------------------------------------------------------

    # Демонстрационная заглушка: 7 примеров текстов вперемешку на русском и английском
    stub_samples = [
        "Привет! Я разрабатываю новую языковую модель для общения и решения задач.",
        "Hello! I am developing a new language model for conversation and problem solving.",
        "Машинное обучение и нейросети требуют эффективной побайтовой BPE токенизации.",
        "Byte-Pair Encoding (BPE) splits text into frequent subwords and characters.",
        "def solve_equation(a: float, b: float) -> float:\n    return -b / a",
        "Искусственный интеллект способен писать качественный код и решать сложные уравнения.",
        "Multilingual models handle Russian and English seamlessly using shared subword representations."
    ]

    # Цикл while True позволяет забирать сколько угодно текста.
    # Для заглушки задано ограничение повторений, чтобы обучение корректно завершилось.
    # При подключении своего генератора настройте остановку по размеру корпуса или условию.
    max_samples = 20000
    yielded_count = 0

    while True:
        for sample in stub_samples:
            yield sample
            yielded_count += 1
            if yielded_count >= max_samples:
                return


# =============================================================================
# 3. ОСНОВНАЯ ФУНКЦИЯ ОБУЧЕНИЯ
# =============================================================================

def train() -> None:
    """
    Создает, обучает и сохраняет BPE-токенизатор на базе HuggingFace tokenizers.
    """
    print(f"Инициализация BPE-токенизатора (целевой размер словаря: {VOCAB_SIZE})...")

    # 1. Создание модели BPE с указанием токена для неизвестных символов
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # 2. Побайтовая предварительная токенизация (ByteLevel).
    # Обеспечивает возможность представить абсолютно любой символ (включая эмодзи и спецсимволы)
    # в виде байтов, благодаря чему токенизатор никогда не падает с ошибкой на неизвестных словах.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 3. Пост-процессинг для корректного декодирования байтовых смещений
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # 4. Настройка BpeTrainer со специальными токенами и целевым размером словаря
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    # 5. Обучение токенизатора напрямую из итератора строк
    print("Запуск обучения BPE по потоку текстов...")
    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)

    # 6. Сохранение обученного токенизатора в файл vocab.json
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUT_DIR / "vocab.json"
    tokenizer.save(str(save_path))

    # 7. Вывод результатов и диагностической информации
    vocab_size = tokenizer.get_vocab_size()
    print("\n" + "=" * 60)
    print("Обучение токенизатора успешно завершено!")
    print(f"Путь к файлу: {save_path.resolve()}")
    print(f"Фактический размер словаря: {vocab_size} токенов")
    print("=" * 60)

    # Получение и вывод первых 20 токенов словаря, отсортированных по их ID
    print("\nПервые 20 токенов словаря (ID и строковое представление):")
    vocab = tokenizer.get_vocab()
    sorted_tokens = sorted(vocab.items(), key=lambda item: item[1])

    for token_str, token_id in sorted_tokens[:20]:
        print(f"  ID {token_id:2d}: {repr(token_str)}")

    print("\nСпецтокены проверены:")
    for token in SPECIAL_TOKENS:
        token_id = tokenizer.token_to_id(token)
        print(f"  {token} -> ID {token_id}")


# =============================================================================
# 4. ТОЧКА ВХОДА
# =============================================================================

if __name__ == "__main__":
    train()


# =============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ СОХРАНЁННОГО ТОКЕНИЗАТОРА:
# =============================================================================
# from tokenizers import Tokenizer
#
# tok = Tokenizer.from_file("vocab.json")
# encoded = tok.encode("Привет, hello world")
# print(encoded.tokens)
# print(encoded.ids)
