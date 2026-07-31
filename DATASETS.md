# Датасеты

`scripts/prepare_datasets.py` скачивает выбранные наборы через `datasets.load_dataset` и сохраняет их в JSONL через `Dataset.to_json(..., force_ascii=False)`.

Используемые источники:

- `evilfreelancer/golang-en-ru` — код Go, русский/английский, MIT.
- `lighteval/QazUNTv2` — математика, конфигурации `en` и `ru`, CC BY 4.0.
- `SonexaAI/ru_eng-dataset` — короткие диалоги, русский/английский, CC BY 4.0.

Скачать все наборы:

```bash
python scripts/prepare_datasets.py --output datasets/processed
```

Для теста или ограничения объёма:

```bash
python scripts/prepare_datasets.py --output datasets/processed --max-rows 1000
python scripts/prepare_datasets.py --only code conversation
```

Результаты:

```text
datasets/processed/code.jsonl
datasets/processed/math.jsonl
datasets/processed/conversation.jsonl
```

В каждом объекте есть нормализованное поле `text`, которое можно передать в `train.py` вместе с каталогом `datasets/processed`.
