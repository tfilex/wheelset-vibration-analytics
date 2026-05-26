# Demo Best Models

Этот каталог содержит checkpoint-файлы, которые используются Streamlit-демо в режиме `demo` / `Prod`.

## Активные модели

| Задача | Default checkpoint | Причина выбора |
|---|---|---|
| Классификация CWRU | `classification/cwru_classifier.pth` | Отобранный ResNet-18 классификатор для demo-инференса CWRU. |
| RUL XJTU-SY | `rul/best_rul_transformer_improved_ws1024_v3rnn_train_rul_hybrid_v3_rnn_profilebalanced_trials30_epochs10_featurecache_on.pth` | Выбранный защитный default из семейства `v3_rnn`: ImprovedTransformer, `finetune_cnn`, MSE `0.011698`, MAE `0.251594`, R2 `-0.357861`. |

RUL default закреплен в `src/prediction/demo_inference.py` через `DEFAULT_RUL_CHECKPOINT`. Старое имя `rul/xjtu_rul.pth` поддерживается как совместимый alias, если файл присутствует, но не является обязательным для текущего demo-режима.

## Режимы каталога

В режиме `demo` веб-демо показывает активные default checkpoint и не зависит от fallback-моделей из исследовательских папок. В режиме `experimental` дополнительно видны совместимые checkpoint из `models/demo_best/rul/`, `models/pred_0/`, `models/preds_2_unfrozen/`, `models/preds_3/`, `models/preds_3_frozen/` и `models/preds_3_rnn/`.

В Docker-образе режим задается переменными окружения:

```bash
MODEL_CATALOG_MODE=demo
MODEL_CATALOG_LOCKED=1
```

Для просмотра всех экспериментальных checkpoint:

```bash
MODEL_CATALOG_MODE=experimental
MODEL_CATALOG_LOCKED=1
```

Если активные файлы отсутствуют, демо использует fallback-модели:

- `models/cnn/best_resnet18.pth`;
- `models/pred_0/best_rul_lstm.pth`.

Можно переопределить директорию через переменную окружения:

```bash
export DEMO_MODELS_DIR=/path/to/models
```

Можно переопределить конкретные файлы:

```bash
export CWRU_CLASSIFIER_CHECKPOINT=/path/to/cwru_classifier.pth
export XJTU_RUL_CHECKPOINT=/path/to/rul_checkpoint.pth
```
