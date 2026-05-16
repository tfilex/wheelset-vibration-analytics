# Demo Best Models

Положи сюда checkpoint-файлы, которые должны использоваться Streamlit-демо.

Ожидаемые имена по умолчанию:

- `cwru_classifier.pth` — классификатор дефектов CWRU.
- `xjtu_rul.pth` — модель прогноза RUL XJTU-SY.

Если файлы отсутствуют, демо автоматически использует fallback-модели:

- `models/cnn/best_resnet18.pth`;
- `models/pred_0/best_rul_lstm.pth`.

Можно переопределить директорию через переменную окружения:

```bash
export DEMO_MODELS_DIR=/path/to/models
```

Можно переопределить конкретные файлы:

```bash
export CWRU_CLASSIFIER_CHECKPOINT=/path/to/cwru_classifier.pth
export XJTU_RUL_CHECKPOINT=/path/to/xjtu_rul.pth
```
