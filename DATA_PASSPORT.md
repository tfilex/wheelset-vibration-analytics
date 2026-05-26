# DATA_PASSPORT

Паспорт фиксирует, какие данные реально используются в проекте `wheelset-vibration-analytics`, как они преобразуются и где проходят границы применимости.

## Краткая сводка

| Датасет | Задача | Реальный вход | Использование в проекте |
|---|---|---|---|
| CWRU Bearing Data Center | Классификация дефектов подшипника | `.mat`, канал `*_DE_time` | STFT-спектрограммы для CNN/ResNet-18 и Streamlit demo. |
| XJTU-SY Bearing Datasets | Прогноз RUL | `.csv`, два канала вибрации `horizontal/vertical` | CWT-скалограммы и последовательности для CNN + temporal RUL моделей. |

## CWRU

| Поле | Значение |
|---|---|
| Локальный путь | `data/raw/CWRU/` |
| Формат | MATLAB `.mat` |
| Используемый канал | Drive End vibration, ключи вида `*_DE_time` |
| Классы demo | `Норма`, `IR 0.007`, `OR 0.007` из `src/prediction/demo_inference.py` |
| Полная схема классов | 10 классов: Normal, IR/Ball/OR с размерами дефекта 0.007, 0.014, 0.021 |
| Окно сигнала | 1024 отсчета |
| Преобразование | STFT, `nperseg=256`, `noverlap=128`, далее `abs(zxx) ** 2` |
| Demo-модель | `models/demo_best/classification/cwru_classifier.pth` |

Текущий training split в `src/classification/train.py`: стратифицированное разбиение окон 70 / 15 / 15 через `train_test_split`. Это удобно для исследовательского прототипа, но не является file-grouped split: окна одного исходного `.mat` могут попасть в разные части. Для промышленного отчета это нужно явно отмечать как ограничение или заменить grouped split по исходному файлу/режиму нагрузки.

## XJTU-SY

| Поле | Значение |
|---|---|
| Локальный путь | `data/raw/XJTU-SY/` |
| Режимы | `35Hz12kN`, `37.5Hz11kN`, `40Hz10kN` |
| Bearing-директории | 15 run-to-failure директорий, по 5 bearings на режим |
| Формат файла | CSV, две числовые колонки вибрации |
| Используемые каналы | Horizontal и vertical vibration signals |
| Окно сигнала | 1024 отсчета для активного demo checkpoint |
| Преобразование | CWT Mexican hat (`mexh`), scales `1..32`, два канала складываются в тензор `(2, scales, window)` |
| Последовательность | `seq_length=10` для v3 RNN demo checkpoint |
| Целевая переменная | Нормированный RUL `[0, 1]`; в v3 RNN pipeline используется `rul_clip=0.8` |
| Demo-модель | `models/demo_best/rul/best_rul_transformer_improved_ws1024_v3rnn_train_rul_hybrid_v3_rnn_profilebalanced_trials30_epochs10_featurecache_on.pth` |

Основной split для семейства `train_rul_hybrid_v3_rnn.py`: во всех трех режимах suffix bearings `1,2,4` идут в train, suffix `5` идет в validation, suffix `3` идет в test. Для validation/test используется `val_test_stride=1`, для train по умолчанию `seq_stride=2`.

В Streamlit demo RUL показ строится по подготовленным bearing-директориям из `XJTU_BEARING_DIRS`: `Bearing1_3`, `Bearing1_4`, `Bearing2_5`, `Bearing3_3`.

## Проверка наличия данных

```bash
find data/raw/CWRU -name "*.mat" | wc -l
find data/raw/XJTU-SY -name "*.csv" | wc -l
uv run python -c "from src.prediction.demo_inference import discover_rul_models; print(discover_rul_models('demo'))"
uv run pytest tests/test_demo_inference.py -q
```

## Ограничения применимости

- Данные CWRU и XJTU-SY являются benchmark-датасетами подшипников, а не промышленными измерениями буксового узла колесной пары вагона.
- В текущем коде нет реального трехосевого входа `ax, ay, az` и нет привязки входного сигнала к скорости движения как измеряемому полю.
- Нет датасета дефектов поверхности катания колеса: ползун, выщербина, овальность, профильный износ.
- Веб-демо использует подготовленные файлы из локального `data/raw/`, а не пользовательскую загрузку произвольного файла.
- RUL в километрах является демонстрационной постобработкой Health Index, а не физически валидированной моделью пробега.
- Интерактивная карта важности является gradient attribution, не SHAP. SHAP используется только в отдельных исследовательских/offline материалах.
