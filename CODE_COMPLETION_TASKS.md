# CODE_COMPLETION_TASKS

Документ для агента-разработчика, который будет доводить код, данные и демонстрационный контур до состояния, пригодного для защиты ВКР.

Основано на анализе репозитория `/home/ish/rudn/VKR`, README, исходного кода, тестов, Docker/Makefile, внешнего текста ВКР `C:/rudn/2 курс/VKR/диплом/ЗФИмд-01-24_Дудолин_ВКР_версия_23.docx`, индивидуального задания `C:/rudn/2 курс/VKR/требования/Задание Дудолин.pdf` и методических требований `C:/rudn/2 курс/VKR/требования/Требования к ВКР.pdf`.

## 1. Краткое описание проекта

Проект `wheelset-vibration-analytics` является исследовательским ML-прототипом для предиктивной вибродиагностики буксовых подшипников и приближенной демонстрации диагностики колесных пар вагонов.

Решаемые задачи:

- классификация дефектов подшипников по вибросигналам CWRU;
- прогнозирование остаточного ресурса RUL по run-to-failure данным XJTU-SY;
- расчет Health Index, диагностического статуса и демонстрационного остаточного пробега;
- визуальная демонстрация через Streamlit и CLI-сценарий.

Основные компоненты уже есть:

- `app.py` - Streamlit-интерфейс с тремя страницами;
- `src/classification/` - загрузка CWRU, STFT, CNN-классификаторы, обучение с Optuna/MLflow;
- `src/prediction/` - XJTU-SY, CWT-скалограммы, CNN+LSTM/GRU/TCN/Transformer/Mamba-fallback, обучающие пайплайны;
- `src/prediction/demo_inference.py` - загрузка checkpoint и реальный инференс для демо;
- `src/utils/health_index.py` и `src/utils/thresholds.py` - HI, статусы, пробег, скоростные пороги;
- `src/features/` - статистические признаки и kurtogram;
- `console_diagnostics/run.py` - оффлайн RUL/HI диагностика без Streamlit;
- `tests/` - smoke/unit-тесты;
- `Dockerfile`, `Makefile`, `pyproject.toml`, `requirements.txt` - запуск, зависимости, контейнер.

Стек технологий:

- Python 3.10+ / 3.12 в текущем окружении;
- PyTorch, torchvision;
- NumPy, pandas, SciPy, PyWavelets;
- scikit-learn, hmmlearn, CatBoost;
- Optuna, MLflow;
- SHAP как зависимость исследовательского контура;
- Streamlit, Plotly;
- Docker;
- uv, pytest, pytest-cov.

После завершения должен получиться воспроизводимый демонстрационный модуль: по инструкции запускается Streamlit, доступны выбранные стабильные модели классификации и RUL, основной сценарий показывает класс дефекта, график сигнала, карту важности, прогноз RUL, HI, статус и остаточный пробег. В README и дипломе должно быть честно указано, что это исследовательский прототип на CWRU/XJTU-SY, а не промышленная система постоянного трехосевого мониторинга колесной пары.

## 2. Текущее состояние реализации

### Frontend / demo UI

**Код:**

- `app.py`
- `src/visualization/plots.py`
- `src/demo/mock_data.py`

**Что работает:**

- Streamlit запускает три раздела: классификация CWRU, прогноз RUL XJTU-SY, дашборд;
- есть выбор checkpoint через `discover_classification_models()` и `discover_rul_models()`;
- есть графики исходного сигнала, heatmap важности, True/Pred RUL, HI, gauge;
- ошибки загрузки моделей и инференса показываются через `st.error`.

**Что работает частично:**

- страница классификации использует выбор из предопределенных сигналов `SIGNAL_TYPES`, а не загрузку пользовательского файла;
- карта важности строится как gradient * input в `_build_attribution()`, а не как SHAP;
- дашборд использует демонстрационные константы из `src/demo/mock_data.py`;
- остаточный пробег является инженерной демонстрационной шкалой, а не подтвержденной моделью километража.

**Что отсутствует:**

- `st.file_uploader`;
- поддержка пользовательского CWRU/XJTU/X-Y-Z файла;
- вероятностное распределение по всем 10 классам в UI;
- доверительный интервал RUL на базе ансамбля;
- класс `InferenceService`, заявленный в дипломном тексте;
- persistent storage истории проверок.

**Смотреть в первую очередь:**

- `app.py`
- `src/prediction/demo_inference.py`
- `src/visualization/plots.py`
- `src/demo/mock_data.py`

### Backend / inference layer

**Код:**

- `src/prediction/demo_inference.py`
- `console_diagnostics/run.py`

**Что работает:**

- lazy/cached загрузка моделей через `@st.cache_resource`;
- fallback-пути для классификации и RUL;
- поддержка нескольких RUL temporal-типов: `lstm`, `gru`, `tcn`, `transformer`, `bilstm`, `bigru`, attention-варианты, `transformer_improved`;
- `predict_rul_series()` строит последовательность точек RUL по XJTU-SY CSV;
- CLI-сценарий сохраняет CSV и PNG результатов.

**Что работает частично:**

- стабильный demo-режим по README ожидает `models/demo_best/rul/xjtu_rul.pth`, но в текущем git status этот файл удален; при этом в `models/demo_best/rul/` уже есть другие RUL-checkpoint;
- `models/preds_4_tcn/` и `models/preds_5_odd/` не включены в автоматический выбор веб-демо;
- функции инференса жестко привязаны к предопределенным CWRU/XJTU файлам.

**Что отсутствует:**

- единый сервисный класс или API-слой;
- проверка схемы пользовательских файлов;
- пакетная обработка произвольной поездки;
- API endpoint или backend-сервис отдельно от Streamlit.

### База данных

**Код:** отсутствует.

**Что есть вместо БД:**

- локальные CSV в `results/`;
- MLflow SQLite `mlflow.db`;
- артефакты в `mlruns/`, `mlartifacts/`;
- демонстрационная история в `src/demo/mock_data.py`.

**Что отсутствует:**

- схема БД;
- хранение пользователей/сессий/истории диагностик;
- миграции;
- механизм аудита решений.

Для защиты БД не обязательна, если в дипломе не заявлять ее наличие.

### API

**Код:** отдельного API нет.

**Что есть:**

- внутренние Python-функции `classify_signal()`, `predict_rul_series()`, `compute_hi()`, `get_status()`.

**Что отсутствует:**

- REST/gRPC API;
- OpenAPI;
- contract tests.

Для защиты достаточно описать как локальный программный модуль, если не заявлять API.

### ML / алгоритмы

**Код:**

- `src/classification/model.py`
- `src/classification/data_loader.py`
- `src/classification/train.py`
- `src/prediction/model.py`
- `src/prediction/data_loader.py`
- `src/prediction/train.py`
- `src/prediction/train_rul_hybrid_v2.py`
- `src/prediction/train_rul_hybrid_v3.py`
- `src/prediction/train_rul_hybrid_v3_rnn.py`
- `src/prediction/train_rul_hybrid_v4_tcn.py`
- `src/prediction/train_rul_hybrid_v5_odd.py`
- `src/models/hmm_baseline.py`
- `src/evaluation/roc_analysis.py`

**Что работает:**

- CWRU: чтение `.mat`, извлечение `_DE_time`, оконная нарезка, STFT, CNN-фабрика;
- классификационные архитектуры: ResNet-18, SqueezeNet, MobileNetV3, EfficientNet, ShuffleNet, ConvNeXt, EfficientNetV2, RegNet;
- RUL: чтение двухканальных CSV XJTU-SY, CWT, последовательности, нормированный RUL;
- гибридная модель `UniversalHybridRULNet`: CNN encoder + temporal block + regression head;
- temporal blocks: LSTM, GRU, TCN, Transformer, Mamba через GRU fallback;
- экспериментальные RUL-heads в demo inference: BiLSTM, BiGRU, attention heads, improved transformer;
- HMM baseline на статистических признаках.

**Что работает частично:**

- в `src/prediction/model.py` Mamba описана как поддерживаемая, но без `mamba-ssm` работает как GRU fallback;
- CWRU train split в `src/classification/train.py` делит окна случайно, а не по исходным `.mat` файлам, что создает риск leakage;
- скорость в `UniversalHybridRULNet.forward()` есть как optional вход, но реальные датасеты и demo mostly используют `speed=None`, то есть нулевую скорость;
- `src/features/stat_features.py` и `src/features/kurtogram.py` есть, но не являются основным путем в Streamlit.

**Что отсутствует:**

- датасет и классы дефектов именно колесной пары: ползун, выщербина, овальность, профиль колеса;
- трехосевой вход X/Y/Z;
- industrial online stream inference;
- доказанная модель “RUL в километрах” для вагона;
- подтверждение гипотезы 2000-3000 км до предаварийного состояния.

### Парсинг и обработка данных

**Код:**

- `src/classification/data_loader.py`
- `src/prediction/data_loader.py`
- `src/prediction/demo_inference.py`
- `src/features/stat_features.py`
- `src/features/kurtogram.py`

**Что работает:**

- CWRU `.mat` -> STFT;
- XJTU-SY `.csv` -> CWT-scalogram;
- статистические признаки RMS, peak/crest factor, kurtosis, variance;
- fast-kurtogram style поиск импульсной полосы.

**Что отсутствует:**

- валидатор пользовательских файлов;
- обработка трехкомпонентного `x,y,z` сигнала;
- привязка к скорости движения;
- нормализованный data passport.

### Авторизация

Не реализована и для текущего демо не требуется. Не заявлять в дипломе.

### Хранение файлов

**Что есть:**

- `data/raw/CWRU/` - 10 `.mat` файлов по классам;
- `data/raw/XJTU-SY/` - 15 bearing-директорий с CSV;
- `data/cache/` - кэш CWT/CNN features;
- `models/` - большое количество checkpoint;
- `models/demo_best/` - каталог выбранных demo-моделей;
- `results/`, `figures/`, `reports/` - результаты и графики.

**Проблемы:**

- `data/`, `reports/`, `models/demo_best/**/*.pth` в `.gitignore`, кроме двух исключений;
- в git статусе `models/demo_best/rul/xjtu_rul.pth` удален;
- в `models/demo_best/rul/` есть другие модели, но README ожидает конкретное имя `xjtu_rul.pth`.

### Тесты

**Код:**

- `tests/test_prediction_model.py`
- `tests/test_classification_model.py`
- `tests/test_rul_dataset.py`
- `tests/test_health_index.py`
- `tests/test_thresholds.py`
- `tests/test_stat_features.py`
- `tests/test_kurtogram.py`
- `tests/test_roc_analysis.py`
- `tests/test_visualization.py`
- `tests/test_demo_helpers.py`
- `tests/test_vkr_materials.py`

**Текущий результат:**

```text
uv run pytest --cov=src --cov=console_diagnostics --cov-report=term-missing -q
26 passed, 1 warning
TOTAL coverage: 45%
```

**Покрыто хорошо:**

- thresholds: 100%;
- visualization plots: 100%;
- health_index: 93%;
- prediction data loader: 91%;
- stat_features: 92%.

**Покрыто плохо:**

- `console_diagnostics/run.py`: 0%;
- `src/models/hmm_baseline.py`: 0%;
- `src/prediction/model.py`: 38%;
- `src/classification/model.py`: 41%;
- `src/evaluation/roc_analysis.py`: 48%.

### Документация

**Есть:**

- `README.md`;
- `project_struct.md`;
- `console_diagnostics/README.md`;
- `models/demo_best/README.md`;
- русскоязычные документы доработки ВКР в корне.

**Частично:**

- README достаточно подробный для запуска, но требует уточнения demo-best RUL checkpoint;
- не хватает data passport;
- не хватает таблицы “требование задания - реализация - ограничение”;
- не хватает честного списка limitations.

### Конфигурация и деплой

**Есть:**

- `pyproject.toml`, `uv.lock`;
- `requirements.txt`;
- `Makefile`;
- `Dockerfile`;
- `.dockerignore`;
- `run_mlflow.sh`;
- переменные `MODEL_CATALOG_MODE`, `MODEL_CATALOG_LOCKED`, `DEMO_MODELS_DIR`, `CWRU_CLASSIFIER_CHECKPOINT`, `XJTU_RUL_CHECKPOINT`.

**Отсутствует:**

- `.env.example`;
- GitHub Actions / CI;
- Helm/Kubernetes;
- Prometheus/Grafana;
- drift monitoring;
- автоматическая проверка Docker image в CI.

## 3. Что нужно дописать в коде

### TASK-001: Закрепить стабильную RUL-модель для demo-каталога

**Приоритет:** critical  
**Модуль:** models / demo / docs  
**Файлы:**

- `models/demo_best/rul/`
- `models/demo_best/README.md`
- `README.md`
- `src/prediction/demo_inference.py`

**Проблема:**  
README и код ожидают `models/demo_best/rul/xjtu_rul.pth`, но в текущем git статусе этот файл удален. При этом в `models/demo_best/rul/` уже есть другие RUL-checkpoint, то есть проблема не в отсутствии моделей, а в отсутствии стабильного default-артефакта/алиаса.

**Что сделать:**

1. Выбрать лучшую модель из `models/demo_best/rul/*.pth` по метрикам checkpoint (`test_mse`, `test_mae`, `test_r2`) и совместимости с `load_cnn_lstm_rul_model()`.
2. Скопировать или переименовать выбранный checkpoint в `models/demo_best/rul/xjtu_rul.pth`, либо изменить README и default path на фактическое имя.
3. Проверить `discover_rul_models("demo")`: первым должен быть стабильный demo checkpoint.
4. Обновить `models/demo_best/README.md` с указанием выбранной модели и причины выбора.

**Ожидаемый результат:**  
Demo-режим не зависит от fallback `models/pred_0/best_rul_lstm.pth`, а использует явно выбранную модель из `models/demo_best/rul/`.

**Критерий готовности:**  

```bash
uv run python -c "from src.prediction.demo_inference import discover_rul_models; print(discover_rul_models('demo'))"
```

Вывод содержит `models/demo_best/rul/xjtu_rul.pth` или документированный новый default.

**Зависимости:**  
Нет.

### TASK-002: Добавить интеграционный smoke-тест demo inference

**Приоритет:** critical  
**Модуль:** tests / demo  
**Файлы:**

- `tests/test_demo_inference.py` (создать)
- `src/prediction/demo_inference.py`

**Проблема:**  
Текущие тесты не проверяют реальную загрузку demo-checkpoint и сквозной вызов `classify_signal()` / `predict_rul_series()`.

**Что сделать:**

1. Добавить тест `discover_classification_models("demo")` и `discover_rul_models("demo")`.
2. Проверить, что выбранные checkpoint существуют.
3. Добавить опциональный slow-тест реального инференса для одного CWRU-сигнала и одного bearing с малым `output_steps`.
4. Пометить тяжелый тест маркером `@pytest.mark.slow` или пропускать при отсутствии файлов.

**Ожидаемый результат:**  
Перед защитой можно быстро проверить, что демонстрационный сценарий не сломан.

**Критерий готовности:**  
`uv run pytest tests/test_demo_inference.py -q` проходит локально.

**Зависимости:**  
TASK-001.

### TASK-003: Реализовать или убрать из диплома загрузку пользовательского файла

**Приоритет:** high  
**Модуль:** frontend / data  
**Файлы:**

- `app.py`
- `src/prediction/demo_inference.py`
- `README.md`

**Проблема:**  
В дипломе описано, что пользователь загружает виброзапись, но в `app.py` используются `selectbox` с предустановленными сигналами.

**Что сделать:**

Вариант A, предпочтительный:

1. Добавить `st.file_uploader` для `.mat` CWRU на странице классификации.
2. Реализовать функцию `classify_uploaded_mat(file)` или обобщить `_build_cwru_input()`.
3. Добавить валидацию: наличие `*_DE_time`, минимальная длина 1024, понятная ошибка.
4. Добавить тест на валидацию.

Вариант B, быстрый:

1. Оставить selectbox.
2. В дипломе и README написать “выбор тестового сигнала из подготовленной выборки”.

**Ожидаемый результат:**  
Текст и код не противоречат друг другу.

**Критерий готовности:**  
Либо UI реально принимает файл, либо в дипломе удалены фразы про загрузку файла.

**Зависимости:**  
Нет.

### TASK-004: Заменить псевдо-SHAP формулировки или подключить настоящий SHAP

**Приоритет:** high  
**Модуль:** explainability / docs / frontend  
**Файлы:**

- `src/prediction/demo_inference.py`
- `src/visualization/plots.py`
- `app.py`
- `scratch_scripts/scratch_shap.py`
- дипломный текст

**Проблема:**  
В `_build_attribution()` используется gradient * input, но диплом называет карту SHAP/GradientExplainer.

**Что сделать:**

Вариант A:

1. Реализовать SHAP GradientExplainer для классификатора.
2. Закэшировать background batch.
3. Ограничить вычисления для demo, чтобы не сломать скорость.

Вариант B:

1. Переименовать UI и дипломные формулировки в “градиентная карта важности”.
2. SHAP оставить как исследовательский материал из `scratch_scripts/make_figure_22_cwru_shap_pretty.py`, если есть готовые рисунки.

**Ожидаемый результат:**  
Не будет завышенного утверждения про SHAP в интерактивном demo.

**Критерий готовности:**  
В коде и тексте одинаковый термин: либо SHAP реально считается, либо написано “gradient attribution”.

**Зависимости:**  
TASK-002 желательно.

### TASK-005: Исправить риск утечки данных в CWRU train/test split

**Приоритет:** high  
**Модуль:** ML / classification  
**Файлы:**

- `src/classification/data_loader.py`
- `src/classification/train.py`
- `experiments/run_roc_analysis.py`
- `scratch_scripts/make_cwru_resnet18_file_grouped_confusion_matrix.py`

**Проблема:**  
`CWRUDataset` нарезает окна из `.mat` файлов, а `load_and_split_data()` в `src/classification/train.py` делит отдельные окна. Окна одного исходного файла могут попасть и в train, и в test.

**Что сделать:**

1. Сохранять `source_file` для каждого окна в `CWRUDataset`.
2. Реализовать grouped split по исходным `.mat` файлам или по режимам нагрузки.
3. Обновить ROC/test scripts.
4. Сравнить метрики window-random split vs file-grouped split.
5. В дипломе честно указать протокол.

**Ожидаемый результат:**  
Метрики классификации будут менее уязвимы к претензии “модель видела тот же сигнал”.

**Критерий готовности:**  
Есть тест/скрипт, подтверждающий отсутствие пересечения `source_file` между train/val/test.

**Зависимости:**  
Нет.

### TASK-006: Поддержать пользовательский двухканальный XJTU CSV или ZIP-папку

**Приоритет:** medium  
**Модуль:** frontend / data / prediction  
**Файлы:**

- `app.py`
- `src/prediction/demo_inference.py`
- `src/prediction/data_loader.py`

**Проблема:**  
RUL demo принимает только предопределенные bearing-директории из `XJTU_BEARING_DIRS`.

**Что сделать:**

1. Добавить режим “Тестовая выборка” / “Загрузка файлов”.
2. Поддержать ZIP с CSV или список CSV.
3. Проверять две числовые колонки.
4. Ограничить размер файла.
5. Возвращать понятные ошибки.

**Ожидаемый результат:**  
Демо ближе к заявленному “загрузить вибролог и получить прогноз”.

**Критерий готовности:**  
Можно загрузить небольшой набор CSV и получить RUL-график.

**Зависимости:**  
TASK-001.

### TASK-007: Документировать или реализовать ensemble confidence interval

**Приоритет:** high  
**Модуль:** prediction / docs  
**Файлы:**

- `src/prediction/demo_inference.py`
- `app.py`
- дипломный текст

**Проблема:**  
Диплом утверждает, что прогноз RUL имеет доверительный интервал на основе ансамбля. В `app.py` выводится только один `pred_rul`, а неопределенность в км рассчитывается эвристически через slope.

**Что сделать:**

Вариант A:

1. Выбрать 3-5 совместимых RUL checkpoint.
2. Добавить `predict_rul_ensemble()`.
3. Считать mean/std или quantile interval.
4. Отрисовать interval band в `build_rul_figure()`.

Вариант B:

1. Убрать слово “ансамбль” из диплома.
2. Описать `sigma_km` как эвристическую оценку неопределенности по тренду HI.

**Ожидаемый результат:**  
Интервал RUL не будет недоказанным утверждением.

**Критерий готовности:**  
В UI либо есть interval band от нескольких моделей, либо текст не обещает ансамбль.

**Зависимости:**  
TASK-001.

### TASK-008: Поднять покрытие тестов до приемлемого уровня

**Приоритет:** high  
**Модуль:** tests  
**Файлы:**

- `tests/`
- `src/prediction/model.py`
- `src/classification/model.py`
- `src/models/hmm_baseline.py`
- `console_diagnostics/run.py`

**Проблема:**  
Фактическое покрытие 45%, методические требования говорят про 80% для тестового отчета.

**Что сделать:**

1. Добавить тесты всех веток `get_model()` для классификационных backbone на dummy input.
2. Добавить тесты temporal blocks: `lstm`, `gru`, `tcn`, `transformer`, `mamba` fallback.
3. Добавить unit-тесты `HMMRULPredictor` на синтетических последовательностях.
4. Добавить тест CLI `console_diagnostics/run.py` через monkeypatch `predict_rul_series`.
5. Добавить coverage target в README/Makefile, например `make coverage`.

**Ожидаемый результат:**  
Покрытие растет хотя бы до 65-75% быстро; 80% - целевой максимум при наличии времени.

**Критерий готовности:**  
`uv run pytest --cov=src --cov=console_diagnostics --cov-report=term-missing -q` показывает целевой процент, согласованный с текстом ВКР.

**Зависимости:**  
TASK-002 желательно.

### TASK-009: Добавить data passport

**Приоритет:** high  
**Модуль:** data / docs  
**Файлы:**

- `DATA_PASSPORT.md` (создать)
- `README.md`
- дипломный текст

**Проблема:**  
Методические требования требуют паспорт датасета: источник, структура, split, ограничения, reproducibility.

**Что сделать:**

1. Описать CWRU: 10 `.mat` файлов, классы, канал DE, STFT, window size 1024.
2. Описать XJTU-SY: 15 bearing-директорий, режимы 35Hz12kN/37.5Hz11kN/40Hz10kN, CSV с horizontal/vertical channels.
3. Указать train/val/test split для каждого pipeline.
4. Указать ограничения: не колесная пара вагона, нет X/Y/Z, нет скорости в реальном входе.
5. Указать команды проверки наличия данных.

**Ожидаемый результат:**  
Комиссия видит, откуда данные, что именно обучалось и где границы применимости.

**Критерий готовности:**  
`DATA_PASSPORT.md` существует и содержит таблицы по CWRU/XJTU-SY.

**Зависимости:**  
Нет.

### TASK-010: Добавить `.env.example`

**Приоритет:** medium  
**Модуль:** config / docs  
**Файлы:**

- `.env.example` (создать)
- `README.md`

**Проблема:**  
Переменные окружения описаны в README фрагментарно, файла-примера нет.

**Что сделать:**

Добавить:

```env
MODEL_CATALOG_MODE=demo
MODEL_CATALOG_LOCKED=1
DEMO_MODELS_DIR=models/demo_best
CWRU_CLASSIFIER_CHECKPOINT=models/demo_best/classification/cwru_classifier.pth
XJTU_RUL_CHECKPOINT=models/demo_best/rul/xjtu_rul.pth
```

**Ожидаемый результат:**  
Запуск demo-контура проще воспроизвести.

**Критерий готовности:**  
README ссылается на `.env.example`.

**Зависимости:**  
TASK-001.

### TASK-011: Добавить CI для тестов

**Приоритет:** medium  
**Модуль:** CI / tests  
**Файлы:**

- `.github/workflows/tests.yml` (создать)
- `README.md`

**Проблема:**  
Методические требования упоминают CI/CD. В репозитории нет `.github` или другого CI.

**Что сделать:**

1. Добавить workflow на Python 3.10/3.12.
2. Установить uv.
3. Выполнить `uv sync --dev`.
4. Запустить `uv run pytest -q`.
5. Не требовать тяжелых данных/моделей в CI, если они не коммитятся.

**Ожидаемый результат:**  
Есть минимальное подтверждение автоматической проверки.

**Критерий готовности:**  
Workflow запускается локально/на GitHub без тяжелого обучения.

**Зависимости:**  
Нет.

### TASK-012: Уточнить Docker-сборку и размер образа

**Приоритет:** medium  
**Модуль:** deploy  
**Файлы:**

- `Dockerfile`
- `.dockerignore`
- `README.md`

**Проблема:**  
Диплом говорит про Docker и вес образа. Dockerfile копирует `data/raw` и много моделей, но нет автоматического теста сборки. Если `xjtu_rul.pth` отсутствует, demo-mode внутри Docker будет fallback.

**Что сделать:**

1. Проверить `make docker-build`.
2. Проверить `make docker-run-demo`.
3. Уточнить в README, какие модели и данные попадают в образ.
4. При необходимости сделать lightweight demo image без всех экспериментальных моделей.

**Ожидаемый результат:**  
Docker-сценарий воспроизводим.

**Критерий готовности:**  
Контейнер стартует, Streamlit открывается на `http://localhost:8501`.

**Зависимости:**  
TASK-001.

### TASK-013: Добавить поддержку трехосевого промышленного формата или явно зафиксировать ограничение

**Приоритет:** high  
**Модуль:** data / prediction / docs  
**Файлы:**

- `src/prediction/data_loader.py`
- `src/prediction/demo_inference.py`
- `README.md`
- `DATA_PASSPORT.md`
- дипломный текст

**Проблема:**  
Индивидуальное задание требует входной трехкомпонентный вектор ускорения X/Y/Z и опционально скорость. Текущий код использует CWRU DE channel и XJTU-SY horizontal/vertical channels.

**Что сделать:**

Вариант A:

1. Спроектировать loader для CSV с колонками `time, ax, ay, az, speed_kmh`.
2. Добавить preprocessing трех каналов.
3. Добавить mock/fixture для промышленного формата.
4. Описать, что модель промышленного формата пока работает в compatibility/demo режиме.

Вариант B:

1. Не реализовывать трехосевой вход.
2. В README и дипломе явно написать: “в работе реализован двухканальный исследовательский прототип; промышленный X/Y/Z-вход относится к направлению внедрения”.

**Ожидаемый результат:**  
Нет скрытого несоответствия с заданием.

**Критерий готовности:**  
Есть либо loader/test для X/Y/Z, либо limitation table.

**Зависимости:**  
TASK-009.

### TASK-014: Добавить в README раздел Known limitations

**Приоритет:** high  
**Модуль:** docs  
**Файлы:**

- `README.md`

**Проблема:**  
Текущий README хорошо описывает проект, но не фиксирует ограничения относительно задания.

**Что сделать:**

Добавить список:

- прототип не является промышленным online monitoring;
- нет реального трехосевого датчика колесной пары;
- нет датасета дефектов поверхности катания колеса;
- CWRU/XJTU-SY - открытые benchmark-датасеты подшипников;
- RUL в км является демонстрационной постобработкой;
- SHAP/ensemble только если реально реализованы.

**Ожидаемый результат:**  
README станет честным и защитным.

**Критерий готовности:**  
Раздел присутствует и совпадает с дипломным текстом.

**Зависимости:**  
TASK-003, TASK-004, TASK-007, TASK-013.

### TASK-015: Добавить пользовательский сценарий защиты

**Приоритет:** medium  
**Модуль:** docs / demo  
**Файлы:**

- `DEMO_SCENARIO.md` (создать)
- `README.md`

**Проблема:**  
Нет отдельного пошагового сценария показа для комиссии.

**Что сделать:**

1. Описать команды запуска.
2. Описать, какой checkpoint выбрать.
3. Описать сценарий CWRU: “Норма”, “IR”, “OR”.
4. Описать сценарий RUL: `Bearing1_3`.
5. Добавить резервный CLI-сценарий `python -m console_diagnostics.run`.
6. Добавить скриншоты или список, какие скриншоты подготовить.

**Ожидаемый результат:**  
На защите можно показать проект без импровизации.

**Критерий готовности:**  
Другой человек запускает demo по документу.

**Зависимости:**  
TASK-001, TASK-002.

## 4. Какие данные нужно собрать

### DATA-001: Стабильные demo-checkpoint

**Какие данные нужны:**  
Выбранный checkpoint классификации и выбранный checkpoint RUL.

**Зачем нужны:**  
Для воспроизводимого demo-режима и Docker-сборки.

**Где получить:**  
Уже есть:

- `models/demo_best/classification/cwru_classifier.pth`;
- несколько файлов `models/demo_best/rul/*.pth`;
- fallback `models/pred_0/best_rul_lstm.pth`.

**Формат:**  
PyTorch `.pth` dict со `state_dict`, `temporal_type`, `seq_length`, метриками.

**Куда положить:**  

- `models/demo_best/classification/cwru_classifier.pth`;
- `models/demo_best/rul/xjtu_rul.pth` или документированный новый default.

**Как проверить:**  
`discover_classification_models("demo")`, `discover_rul_models("demo")`, `load_*_model()`.

**Можно ли mock:**  
Нет для защиты. Mock допустим только в unit-тестах.

### DATA-002: Паспорт CWRU

**Какие данные нужны:**  
Классы, файлы, частота, канал, window size, STFT параметры, split.

**Зачем нужны:**  
Для диплома и воспроизводимости.

**Где получить:**  
`data/raw/CWRU/`, `src/classification/data_loader.py`, `src/classification/train.py`.

**Формат:**  
Markdown-таблица в `DATA_PASSPORT.md`.

**Куда положить:**  
`DATA_PASSPORT.md`, кратко продублировать в README и диплом.

**Как проверить:**  
Скриптом/ручной таблицей: 10 классов, 10 `.mat`.

**Можно ли mock:**  
Нет, это описание реального используемого датасета.

### DATA-003: Паспорт XJTU-SY

**Какие данные нужны:**  
15 bearing-директорий по режимам:

- `35Hz12kN/Bearing1_1 ... Bearing1_5`;
- `37.5Hz11kN/Bearing2_1 ... Bearing2_5`;
- `40Hz10kN/Bearing3_1 ... Bearing3_5`.

**Зачем нужны:**  
Для описания RUL-эксперимента и ограничений.

**Где получить:**  
`data/raw/XJTU-SY/`, `data/raw/XJTU-SY/Introduction_to_XJTU-SY_Bearing_Dataset.pdf`.

**Формат:**  
Markdown-таблица: режим, bearing, роль train/val/test, число CSV, каналы.

**Куда положить:**  
`DATA_PASSPORT.md`.

**Как проверить:**  
Проверить наличие CSV в каждой bearing-директории.

**Можно ли mock:**  
Для тестов да; для дипломных результатов нет.

### DATA-004: Промышленный трехосевой формат X/Y/Z

**Какие данные нужны:**  
Пример CSV с колонками `timestamp, ax, ay, az, speed_kmh`.

**Зачем нужны:**  
Чтобы закрыть требование индивидуального задания по трехкомпонентному датчику.

**Где получить:**  
Если реальных данных нет, создать синтетический fixture и явно обозначить как synthetic/mock.

**Формат:**  
CSV, UTF-8, числовые колонки, частота дискретизации в metadata.

**Куда положить:**  

- `data/examples/three_axis_sample.csv` для small fixture;
- `DATA_PASSPORT.md` с пометкой synthetic.

**Как проверить:**  
Тест loader-а на shape `(N, 3)` и наличие скорости.

**Можно ли mock:**  
Да, но только для демонстрации формата, не для метрик качества.

### DATA-005: Скриншоты и demo-видео

**Какие данные нужны:**  
Скриншоты трех страниц Streamlit и видео 2-4 минуты.

**Зачем нужны:**  
Методические требования к пакету защиты.

**Где получить:**  
После запуска `make demo`.

**Формат:**  
PNG/JPG для скриншотов, MP4 для видео.

**Куда положить:**  
Лучше вне git или в `reports/demo/`, если размер приемлем.

**Как проверить:**  
Открыть файлы и убедиться, что видны checkpoint, результат и графики.

**Можно ли mock:**  
Нет, должны быть скриншоты реального demo.

## 5. Недостающие зависимости и настройки

Проверено:

- `pyproject.toml` содержит полный ML-набор: torch, torchvision, mlflow, optuna, shap, catboost, plotly, streamlit, hmmlearn;
- `requirements.txt` содержит demo-набор плюс pytest/pytest-cov;
- `Makefile` содержит install/test/smoke/check/demo/docker;
- `Dockerfile` есть;
- `.env.example` отсутствует;
- CI отсутствует;
- Helm/Kubernetes отсутствуют;
- Prometheus/Grafana/drift alerts отсутствуют;
- миграции отсутствуют, потому что БД приложения нет;
- внешних API-ключей не требуется.

Что добавить:

- `.env.example`;
- `make coverage`;
- `DATA_PASSPORT.md`;
- `DEMO_SCENARIO.md`;
- `.github/workflows/tests.yml`, если репозиторий будет на GitHub;
- README-раздел “Known limitations”.

## 6. Минимальный план дописывания кода

### Что исправить первым

1. TASK-001: выбрать и закрепить стабильный RUL checkpoint для demo.
2. TASK-002: добавить demo inference smoke-test.
3. TASK-003/TASK-004/TASK-007: привести UI и дипломные заявления к одному фактическому состоянию.
4. TASK-014: добавить limitations в README.

### Что нужно для минимально рабочей версии

1. `make check` проходит.
2. `make demo` запускает Streamlit.
3. В demo доступны classifier и RUL checkpoint.
4. CWRU и RUL сценарии работают без ручной правки путей.
5. README описывает реальные ограничения.

### Что нужно для демонстрации на защите

1. `DEMO_SCENARIO.md`.
2. Стабильный Docker или локальный запуск.
3. Скриншоты трех страниц.
4. CLI fallback: `uv run python -m console_diagnostics.run --bearing Bearing1_3`.
5. Таблица “требование задания - реализация - ограничение”.

### Что можно отложить

- полноценный REST API;
- БД истории диагностик;
- Helm/Kubernetes;
- Prometheus/Grafana;
- реальный online stream;
- полноценный трехосевой industrial loader, если это честно указано как limitation.

### Какие задачи опасно оставлять незавершенными

- отсутствие стабильного demo checkpoint;
- расхождение “SHAP/ensemble/upload” между дипломом и кодом;
- отсутствие честного ограничения по CWRU/XJTU-SY вместо колесной пары;
- отсутствие тестового отчета;
- незапускаемый Docker/demo.

## 7. Команды для проверки

### Установка зависимостей

```bash
uv sync --dev
```

Если uv не установлен:

```bash
python -m pip install uv
uv sync --dev
```

### Запуск проекта

```bash
make demo
```

Или напрямую:

```bash
uv run --with-requirements requirements.txt streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

### Запуск тестов

```bash
make test
```

```bash
uv run pytest -q
```

### Coverage

```bash
uv run pytest --cov=src --cov=console_diagnostics --cov-report=term-missing -q
```

Текущий проверенный результат: `26 passed, 1 warning, TOTAL coverage 45%`.

### Smoke-check моделей

```bash
make smoke
```

Эквивалент:

```bash
uv run python src/classification/model.py
uv run python src/prediction/model.py
```

### Docker

```bash
make docker-build
make docker-run-demo
```

### MLflow

```bash
make mlflow
```

### Генерация материалов для ВКР

```bash
make vkr-materials
```

### Команды, которые стоит добавить

```make
coverage:
	uv run pytest --cov=src --cov=console_diagnostics --cov-report=term-missing -q
```

```make
demo-cli:
	uv run python -m console_diagnostics.run --bearing Bearing1_3
```
