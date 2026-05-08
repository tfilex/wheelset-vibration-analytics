# Структура проекта `wheelset-vibration-analytics`

Проект решает две прикладные задачи вибродиагностики подшипников:
1. **Классификация дефектов** (датасет CWRU).
2. **Прогнозирование остаточного ресурса (RUL)** (датасет XJTU-SY).

Ниже — актуальная структура и назначение ключевых файлов.

## Данные (`data/`)

- `data/raw/CWRU/` — классы состояния подшипника в `.mat` файлах.
- `data/raw/XJTU-SY/` — run-to-failure данные в `.csv` файлах, сгруппированные по режимам и подшипникам.
- `data/processed/` — директория для производных наборов (если используются оффлайн-преобразования).
- `data/cache/cwt_scalograms/` — дисковый кэш CWT-скалограмм для RUL pipeline.
- `data/cache/cnn_features/` — дисковый кэш CNN-признаков для ускоренного RUL pipeline с замороженным encoder.

## Исходный код (`src/`)

### `src/classification/` — классификация дефектов CWRU

- `data_loader.py` — `CWRUDataset`: чтение `.mat`, нарезка окон, расчет STFT-спектрограмм.
- `model.py` — фабрика CNN-моделей (`SUPPORTED_MODELS`) с адаптацией под одноканальные входы.
- `train.py` — основной NAS+HPO pipeline (Optuna + MLflow), финальное обучение, тест, графики, экспорт моделей.
- `visualization/plot_spectrograms.py` — генерация индивидуальных и сводных графиков сигналов/спектров.

### `src/prediction/` — прогнозирование RUL XJTU-SY

- `config.py` — единая конфигурация путей и обучения.
  - Текущие значения для NAS/HPO: `N_TRIALS=50`, `EPOCHS=50`.
  - Для Optuna разрешены только `["lstm", "gru", "transformer"]`.
  - `CNN_CHECKPOINT_PATH=models/cnn/best_resnet18.pth` — checkpoint классификационного ResNet-18, используемый как инициализация CNN-энкодера для RUL.
- `data_loader.py` — `RULDataset`: чтение последовательностей `.csv`, CWT для двух каналов, формирование target RUL.
- `model.py` — `UniversalHybridRULNet`: CNN-энкодер + temporal-блок (`lstm`, `gru`, `tcn`, `transformer`, `mamba`).
  - Умеет загружать backbone из классификационного checkpoint `models/cnn/best_resnet18.pth`.
  - Поддерживает `fine_tune=True`: CNN размораживается и дообучается под задачу RUL.
  - `TemporalOnlyRULNet` — temporal-only вариант для быстрого обучения по заранее посчитанным CNN-фичам.
- `utils.py` — общие утилиты обучения/оценки и визуализации (`MSE/MAE`, learning curves, residuals, RUL plot).
- `train.py` — Optuna NAS+HPO для RUL, затем финальное обучение лучшей конфигурации и сохранение итоговой модели.
  - Пространство поиска `seq_length`: `[10, 20, 30, 50]`.
  - Batch size подбирается с учетом `seq_length`, чтобы снизить риск CUDA OOM при fine-tuning CNN.
- `train_boosting.py` — бейзлайн на `CatBoostRegressor` поверх CNN-фичей.
- `train_three_models.py` — отдельный сценарий без Optuna: последовательно обучает и сохраняет три модели:
  - `lstm`
  - `gru`
  - `transformer`
- `train_three_models_2.py` — расширенный сценарий RUL:
  - объединяет несколько подшипников и режимов XJTU-SY;
  - поддерживает профили `fast`, `balanced`, `full`;
  - умеет запускать отдельные temporal-модели через `--temporal-types`;
  - поддерживает два режима обучения:
    - `--feature-cache` (по умолчанию): быстрое обучение temporal-блока по кэшированным CNN-фичам;
    - `--no-feature-cache`: полный forward CNN+temporal с fine-tuning CNN под RUL;
  - логирует метрики качества, графики Optuna, learning curves, residuals, RUL prediction и скорость инференса.

## Модели и артефакты

- `models/` — сохраненные модели:
  - классификация: `models/cnn/best_<архитектура>.pth`, `models/cnn/best_<архитектура>.onnx`;
  - текущая инициализация RUL CNN: `models/cnn/best_resnet18.pth`;
  - прогноз RUL (Optuna-финал): `best_rul_model.pth`;
  - прогноз RUL (три фиксированные модели):  
    `best_rul_lstm.pth`, `best_rul_gru.pth`, `best_rul_transformer.pth`.
  - прогноз RUL v2: `models/preds_2/best_rul_<temporal>_ws<window>_v2*.pth`.
- `reports/figures/` — итоговые графики для анализа и отчетов.
  - `reports/figures/summary/train_three_models_2/<profile>/ws<window>/` — графики расширенного RUL pipeline.
- `mlflow.db`, `mlruns/`, `mlartifacts/` — трекинг экспериментов через MLflow.
- `run_mlflow.sh` — быстрый запуск локального UI MLflow.

## Рабочие сценарии запуска

- Классификация (NAS/HPO):  
  `uv run python src/classification/train.py`
- RUL с Optuna (только `lstm/gru/transformer`, 50x50):  
  `uv run python src/prediction/train.py`
- RUL без Optuna, последовательное обучение трех моделей:  
  `uv run python src/prediction/train_three_models.py`
- RUL v2, быстрый дневной прогон по кэшированным CNN-фичам:  
  `uv run python src/prediction/train_three_models_2.py --profile balanced --n-trials 30`
- RUL v2, только Transformer в стандартный MLflow experiment:  
  `uv run python src/prediction/train_three_models_2.py --profile balanced --n-trials 30 --temporal-types transformer --experiment-name XJTU_SY_RUL_ThreeModels_v2`
- RUL v2, fine-tuning CNN под RUL без feature-cache:  
  `uv run python src/prediction/train_three_models_2.py --profile balanced --n-trials 30 --temporal-types lstm gru transformer --no-feature-cache --num-workers 0`
- RUL бейзлайн на бустинге:  
  `uv run python src/prediction/train_boosting.py`

## Текущая RUL-архитектура

Для прогнозирования RUL используется гибридная схема:

1. CSV-сигналы XJTU-SY преобразуются в CWT-скалограммы по двум каналам вибрации.
2. CNN-энкодер ResNet-18 инициализируется из классификационной модели `models/cnn/best_resnet18.pth`, обученной на задаче дефектов.
3. Последовательность CNN-признаков подается в temporal-блок (`LSTM`, `GRU` или `Transformer`).
4. Линейная регрессионная голова предсказывает нормализованный RUL в диапазоне `[0, 1]`.

Есть два режима:

- **Быстрый режим** (`--feature-cache`, включен по умолчанию): CNN-признаки один раз кэшируются в `data/cache/cnn_features/`, после чего обучается только temporal-блок. Это удобно для HPO и сравнения LSTM/GRU/Transformer.
- **Fine-tuning режим** (`--no-feature-cache`): CNN участвует в backpropagation и адаптируется под RUL. Этот режим медленнее, но нужен для проверки гипотезы, что замороженные признаки дают константное предсказание.

## Конфигурация окружения и служебные директории

- `pyproject.toml`, `uv.lock` — управление зависимостями через `uv`.
- `requirements.txt` — альтернативный список зависимостей (если нужен pip-совместимый формат).
- `scratch_scripts/`, `scratch/` — экспериментальные черновые скрипты.
- `.gitignore` — исключает виртуальные окружения, временные/системные файлы и тяжелые автогенерируемые артефакты.
