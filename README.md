# NexusAI

NexusAI is a decoder-only transformer project designed for multilingual text generation with a scalable architecture from 100M to 6B parameters.

## Project structure

- `config.py` - общий загрузчик конфигураций и пресеты моделей.
- `train.py` - точка входа для обучения, поддерживает несколько каталогов данных.
- `inference.py` - запуск генерации текста через готовую модель.
- `requirements.txt` - зависимости проекта.
- `README.md` - описание проекта.
- `nexus/` - пакет модели и вспомогательных модулей.
  - `model.py` - модель с позиционными эмбеддингами и decoder-only архитектурой.
  - `transformer.py` - трансформер-блок.
  - `attention.py` - самовнимание с causal mask.
  - `feedforward.py` - FFN-блок.
  - `embedding.py` - эмбеддинги токенов.
  - `tokenizer.py` - простой токенизатор для прототипа.
  - `dataset.py` - датасет для обучения.
  - `data_loader.py` - построение DataLoader.
  - `trainer.py` - цикл обучения с optimizer/scheduler.
  - `optimizer.py` - сборка оптимизатора.
  - `scheduler.py` - линейный `warmup` и decay.
  - `loss.py` - функция потерь.
  - `generation.py` - простая генерация.
  - `utils.py` - вспомогательные утилиты.
- `datasets/` - корневой каталог для данных.
  - `raw/` - сырые тексты для обучения.
  - `processed/` - подготовленные тексты.
  - `tokenizer/` - файлы словаря.
- `checkpoints/` - сохранение чекпоинтов.
- `trained_models/` - отдельные директории для каждого размера модели.
- `logs/` - логи обучения.
- `configs/` - YAML-конфиги для каждого размера модели.

## Как использовать

1. Поместите тексты в одну или несколько папок внутри `datasets/raw/`.
2. Запустите обучение:

```bash
python train.py --config configs/100M.yaml --data datasets/raw --max-length 128
```

3. После обучения модель будет сохраняться в `checkpoints/`.

4. Запустите генерацию:

```bash
python inference.py --checkpoint checkpoints/latest.pt --prompt "Привет" --max-new-tokens 64
```

## На что обратить внимание

- `train.py` поддерживает сканирование нескольких файлов в каталоге `datasets/raw/`.
- Конфиги в `configs/` задают параметры для 100M, 300M, 1B, 3B и 6B.
- Содержимое папок `datasets/`, `checkpoints/`, `trained_models/` и `logs/` подготовлено для дальнейшего использования.

## Дальнейшие шаги

- добавить реальный токенизатор BPE/WordPiece;
- расширить датасет на мультиязычные тексты (русский, украинский, английский);
- добавить поддержку mixed precision и распределённого обучения.
