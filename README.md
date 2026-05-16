# wheelset-vibration-analytics

Проект для магистерской диссертации по теме предиктивной диагностики буксовых узлов. Репозиторий объединяет два ML-направления и демонстрационный веб-интерфейс:

- классификация дефектов подшипников по вибросигналам CWRU;
- прогнозирование остаточного ресурса RUL по данным XJTU-SY;
- Streamlit-демо для показа результатов и сценариев диагностики.

## Основные задачи

1. **Классификация дефектов CWRU**
   - вход: вибросигналы подшипников в `.mat`;
   - преобразование: оконная нарезка, STFT-спектрограммы;
   - модель: CNN/ResNet-18;
   - результат: класс состояния подшипника.

2. **Прогноз остаточного ресурса XJTU-SY**
   - вход: run-to-failure вибросигналы в `.csv`;
   - преобразование: CWT-скалограммы и последовательности признаков;
   - модель: гибридная архитектура CNN + temporal-блок;
   - temporal-варианты: LSTM, GRU, Transformer, TCN и экспериментальные расширения;
   - результат: нормированный RUL в диапазоне `[0, 1]`.

3. **Веб-демо**
   - Streamlit-интерфейс для защиты и демонстрации;
   - интерактивные графики Plotly;
   - реальный инференс сохраненных моделей из `models/`.

## Структура проекта

```text
.
├── app.py                         # Streamlit UI: навигация и экраны демо
├── Dockerfile                     # Контейнер для запуска Streamlit-демо
├── requirements.txt               # Минимальные зависимости веб-демо
├── pyproject.toml                 # Полные зависимости ML-проекта для uv
├── uv.lock                        # Lock-файл uv
├── run_mlflow.sh                  # Локальный запуск MLflow UI
├── project_struct.md              # Подробная карта исследовательского кода
├── data/
│   ├── raw/CWRU/                  # Исходные .mat файлы CWRU
│   ├── raw/XJTU-SY/               # Исходные .csv файлы XJTU-SY
│   ├── processed/                 # Производные датасеты
│   └── cache/                     # Кэш CWT-скалограмм и CNN-фичей
├── models/
│   ├── cnn/                       # ResNet/CNN чекпоинты и ONNX-экспорт
│   ├── pred_0/                    # Базовые RUL-модели
│   ├── preds_2_frozen/            # RUL v2 с feature-cache
│   ├── preds_2_unfrozen/          # RUL v2 с дообучаемым CNN
│   ├── preds_3/                   # Transformer/LSTM/GRU эксперименты
│   ├── preds_3_frozen/            # RUL v3 с замороженным CNN
│   ├── preds_3_rnn/               # RNN/attention эксперименты
│   ├── preds_4_tcn/               # TCN эксперименты, не подключены к демо
│   └── preds_5_odd/               # PatchTST/Conformer эксперименты, не подключены к демо
├── reports/                       # Графики, логи и отчетные артефакты
├── src/
│   ├── classification/            # Классификация дефектов CWRU
│   ├── prediction/                # Прогнозирование RUL XJTU-SY
│   ├── demo/                      # Константы и вспомогательные данные демо
│   └── visualization/             # Plotly-графики для демо
├── scripts/                       # Сценарии запуска длительных экспериментов
├── scratch/                       # Черновые материалы
└── scratch_scripts/               # Черновые скрипты
```

## Веб-демо Streamlit

Демо состоит из трех экранов:

- **Классификация дефектов (CWRU)**: выбор checkpoint, выбор тестового сигнала, реальный инференс ResNet-18, график вибросигнала и карта важности модели.
- **Прогноз ресурса RUL (XJTU-SY)**: выбор checkpoint, реальный инференс CNN+temporal по CSV-окнам XJTU-SY и динамическая отрисовка True RUL / Pred RUL.
- **Бортовой модуль (Дашборд)**: имитация интерфейса машиниста с метриками, gauge chart и таблицей последних проверок.

Архитектура демо:

- `app.py` - UI-слой и навигация;
- `src/demo/mock_data.py` - константы демо и история проверок;
- `src/prediction/demo_inference.py` - загрузка реальных checkpoint и функции инференса;
- `src/visualization/plots.py` - построение интерактивных графиков Plotly.

В `src/prediction/demo_inference.py` функции загрузки моделей обернуты в `@st.cache_resource`, поэтому PyTorch-веса загружаются один раз на процесс Streamlit.
В интерфейсе есть `selectbox` для выбора checkpoint классификации и checkpoint прогноза RUL.

Используемые модели в веб-демо:

- классификация CWRU: `models/demo_best/cwru_classifier.pth`, если файл есть;
- прогноз RUL XJTU-SY: совместимые checkpoint из `models/demo_best/`, `models/pred_0/`, `models/preds_2_unfrozen/`, `models/preds_3/`, `models/preds_3_frozen/`, `models/preds_3_rnn/`;
- fallback для классификации: `models/cnn/best_resnet18.pth`;
- fallback для RUL: `models/pred_0/best_rul_lstm.pth`.

RUL-модели сортируются по `test_mse`, поэтому лучший совместимый checkpoint появляется первым в списке. Поддерживаются базовые `lstm/gru/tcn/transformer`, а также новые головы `bilstm`, `bigru`, `lstm_attn`, `gru_attn`, `transformer_improved`.

`models/preds_4_tcn/` и `models/preds_5_odd/` оставлены как экспериментальные артефакты, но не включены в автоматический выбор веб-демо: их лучшие checkpoint уступают текущему `transformer_improved` по `test_mse`, `test_mae` и `test_r2`, а подключение потребовало бы отдельных inference-голов без выигрыша в качестве.

Можно положить лучшие checkpoint в `models/demo_best/` без изменения кода. Также поддерживаются переменные окружения:

```bash
export DEMO_MODELS_DIR=/path/to/models
export CWRU_CLASSIFIER_CHECKPOINT=/path/to/cwru_classifier.pth
export XJTU_RUL_CHECKPOINT=/path/to/xjtu_rul.pth
```

## Быстрый запуск демо через Docker

```bash
docker build -t bearing-diagnostics-demo .
docker run --rm -p 8501:8501 bearing-diagnostics-demo
```

После запуска открыть:

```text
http://localhost:8501
```

## Запуск демо без Docker

```bash
python3 -m venv .streamlit-venv
source .streamlit-venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

Файл `requirements.txt` предназначен именно для веб-демо:

```text
streamlit
plotly
pandas
numpy
torch
torchvision
scipy
PyWavelets
```

## Полное ML-окружение

Для обучения моделей и запуска исследовательских pipeline используется `uv` и зависимости из `pyproject.toml`.

```bash
uv sync
```

Если `uv` не установлен:

```bash
python -m pip install uv
uv sync
```

Ключевые зависимости полного ML-контура:

- `torch`, `torchvision`, `torchaudio`;
- `scipy`, `scikit-learn`, `PyWavelets`;
- `mlflow`;
- `optuna`;
- `shap`;
- `catboost`;
- `plotly`, `matplotlib`, `seaborn`;
- `numpy`, `pandas`.

## Данные

Ожидаемая структура исходных данных:

```text
data/raw/
├── CWRU/
│   ├── 3_IR_021/
│   ├── 4_Ball_007/
│   └── ...
└── XJTU-SY/
    ├── 35Hz12kN/
    ├── 37.5Hz11kN/
    └── 40Hz10kN/
```

Кэшируемые производные данные:

- `data/cache/cwt_scalograms/` - CWT-скалограммы;
- `data/cache/cnn_features/` - CNN-признаки для ускоренного RUL-пайплайна.

## Обучение моделей

### Классификация CWRU

```bash
uv run python src/classification/train.py
```

Pipeline включает:

- загрузку CWRU `.mat`;
- формирование STFT-спектрограмм;
- подбор и обучение CNN-модели;
- логирование экспериментов в MLflow;
- сохранение чекпоинтов в `models/cnn/`.

### Прогнозирование RUL XJTU-SY

Базовый запуск Optuna/NAS + HPO:

```bash
uv run python src/prediction/train.py
```

Обучение трех фиксированных temporal-моделей:

```bash
uv run python src/prediction/train_three_models.py
```

Гибридный RUL pipeline v2:

```bash
uv run python src/prediction/train_rul_hybrid_v2.py --profile balanced --n-trials 30
```

Гибридный RUL pipeline v3:

```bash
uv run python src/prediction/train_rul_hybrid_v3.py --profile balanced --n-trials 30 --epochs 40 --temporal-types lstm gru transformer --num-workers 0
```

RUL baseline на CatBoost:

```bash
uv run python src/prediction/train_boosting.py
```

## MLflow

Эксперименты логируются в локальный MLflow:

- `mlflow.db`;
- `mlruns/`;
- `mlartifacts/`.

Запуск UI:

```bash
./run_mlflow.sh
```

Обычно интерфейс доступен по адресу:

```text
http://localhost:5000
```

## Модели и артефакты

Основные директории:

- `models/cnn/` - классификационные CNN/ResNet-18 модели;
- `models/pred_0/` - базовые RUL-модели;
- `models/preds_2_frozen/` - RUL v2 с замороженным CNN и feature-cache;
- `models/preds_2_unfrozen/` - RUL v2 с дообучаемым CNN;
- `models/preds_3/` и `models/preds_3_frozen/` - Transformer/LSTM/GRU варианты;
- `models/preds_3_rnn/` - RNN/attention варианты;
- `models/preds_4_tcn/` - TCN варианты, экспериментально;
- `models/preds_5_odd/` - PatchTST/Conformer варианты, экспериментально;
- `reports/figures/` - графики обучения, residuals, RUL prediction и отчеты.

## Как заменить модели в демо

Точки интеграции находятся в `src/prediction/demo_inference.py`.

1. Положить лучший классификатор в `models/demo_best/cwru_classifier.pth`.
2. Положить лучшую RUL-модель в одну из поддерживаемых папок или в `models/demo_best/xjtu_rul.pth`.
3. Для новых RUL-архитектур добавить inference-head в `src/prediction/demo_inference.py`.
4. Если меняется препроцессинг, обновить `_build_cwru_input()` или RUL dataset/pipeline в `predict_rul_series()`.

Такой подход сохраняет Streamlit как тонкий UI-слой, а ML-логику держит внутри `src/prediction`.

## Dockerfile

Контейнер предназначен для запуска веб-демо:

```dockerfile
FROM python:3.10-slim
```

Он копирует:

- `requirements.txt`;
- `app.py`;
- `src/`.
- `models/demo_best/`.
- fallback checkpoint из `models/cnn/` и `models/pred_0/`;
- совместимые лучшие RUL checkpoint из `models/preds_2_unfrozen/`, `models/preds_3/`, `models/preds_3_frozen/`, `models/preds_3_rnn/`;
- `data/raw/` для реального CWRU/XJTU-SY инференса.

Запуск внутри контейнера:

```bash
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

## Примечания

- `requirements.txt` предназначен для запуска веб-демо с реальными checkpoint.
- Для исследовательских запусков использовать `uv sync` и команды `uv run ...`.
- Веб-демо использует реальные сохраненные модели, а дашборд остается демонстрационным сценарием интерфейса.
- Подробная карта исследовательских скриптов находится в `project_struct.md`.
