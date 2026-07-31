1. Общая информация
Цель проекта.
Что такое NexusAI.
Основные принципы.
Почему создается собственная модель.
2. Цели проекта
NexusAI-100M.
NexusAI-300M.
NexusAI-1B.
NexusAI-3B.
NexusAI-6B.
3. Архитектура проекта

Полная структура директорий.
NexusAI/
│
├── train.py                  # Обучение
├── inference.py              # Генерация текста
├── config.py                 # Конфигурация
├── requirements.txt
├── README.md
│
├── nexus/
│   ├── __init__.py
│   ├── model.py              # Полная модель
│   ├── transformer.py        # Transformer Block
│   ├── attention.py          # Multi-Head Attention
│   ├── feedforward.py
│   ├── embedding.py
│   ├── tokenizer.py
│   ├── dataset.py
│   ├── trainer.py
│   ├── optimizer.py
│   ├── scheduler.py
│   ├── loss.py               # Функция потерь
│   ├── generation.py         # Генерация текста
│   └── utils.py
│
├── datasets/
│   ├── raw/                  # Исходные датасеты
│   ├── processed/            # Обработанные
│   └── tokenizer/            # Словарь токенизатора
│
├── checkpoints/
│
├── trained_models/
│   ├── NexusAI-100M/
│   ├── NexusAI-300M/
│   ├── NexusAI-1B/
│   ├── NexusAI-3B/
│   └── NexusAI-6B/
│
├── logs/
│
└── configs/
    ├── 100M.yaml
    ├── 300M.yaml
    ├── 1B.yaml
    ├── 3B.yaml
    └── 6B.yaml
4. Архитектура модели
Decoder-only Transformer.
RMSNorm или LayerNorm (решим после анализа).
Rotary Positional Embeddings (RoPE) или классические позиционные эмбеддинги.
Multi Head Attention.
Feed Forward.
Causal Mask.
5. Токенизатор
Требования.
Формат словаря.
Размер словаря.
Поддержка русского, украинского и английского.
6. Датасеты
Где брать.
Очистка.
Фильтрация.
Нормализация.
Формат хранения.
7. Обучение
Optimizer.
Scheduler.
Batch.
Gradient Accumulation.
Mixed Precision.
Checkpoints.
8. Конфигурации моделей

Отдельные параметры для:

100M
300M
1B
3B
6B
9. Требования к каждому файлу проекта

Например:

config.py

Что должен делать.

attention.py

Что должен делать.

trainer.py

Что должен делать.

И так для каждого файла.

10. Правила написания кода

Например:

Не писать "магический код".
Каждая функция документируется.
Один файл — одна задача.
Максимальная читаемость.
Стиль кода PEP 8.
Типизация (typing).
Докстринги.
Юнит-тесты для критичных компонентов.
11. План разработки

Примерно так:

v0.0.1 — структура проекта.
v0.0.2 — config.
v0.0.3 — Embedding.
v0.0.4 — Attention.
v0.0.5 — FeedForward.
v0.0.6 — Transformer Block.
v0.0.7 — Полная модель.
v0.0.8 — Trainer.
v0.0.9 — Tokenizer.
v0.1.0 — Первая тренировка.
12. Критерии готовности

Что считается завершенным для каждого этапа и какие тесты нужно пройти перед переходом к следующему.

13. Будущие версии

План масштабирования до 300M, 1B, 3B и 6B без изменения общей архитектуры проекта.