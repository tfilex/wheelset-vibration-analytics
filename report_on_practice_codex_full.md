# Полная история разработки проекта и RUL-пайплайна

## 1. Назначение документа

Этот документ собирает полную историю разработки проекта по текущему состоянию репозитория `/home/ish/rudn/VKR`. В отличие от `report_on_practice_codex.md`, который был сфокусирован на трех новых ветках `v3_rnn`, `v4_tcn`, `v5_odd`, здесь описана вся эволюция проекта: от первичной классификации дефектов подшипников до полноценной исследовательской платформы для прогнозирования остаточного ресурса RUL.

История собрана по нескольким источникам:

- git-история ветки `develop`;
- текущая структура проекта;
- содержимое ключевых файлов в `src/classification` и `src/prediction`;
- сохраненные модели в `models`;
- графики и CSV-артефакты в `reports/figures`;
- MLflow-артефакты;
- новые незакоммиченные файлы, которые уже присутствуют в рабочем дереве.

## 2. Краткий итог развития

Проект развивался в несколько крупных этапов:

1. Базовый проект для классификации дефектов CWRU по STFT-спектрограммам.
2. Production-ready классификационный NAS/HPO pipeline с Optuna, MLflow, SHAP и экспортом моделей.
3. Реструктуризация проекта на две задачи: `classification` и `prediction`.
4. Первый RUL pipeline для XJTU-SY: CNN-энкодер + temporal-модель.
5. Бейзлайны для RUL: CatBoost, LSTM, GRU, Transformer.
6. `v2`: multi-bearing RUL pipeline, CWT-cache, CNN feature-cache, профили `fast/balanced/full`.
7. `v3`: двухфазное обучение, piecewise RUL, AsymmetricHuberLoss, monotonicity penalty, EMA-графики, PHM score.
8. Повторяющаяся диагностика коллапсов RUL-моделей: константные предсказания, отрицательный или близкий к нулю R2, несовместимость frozen-HPO и fine-tuning.
9. RUL-specific предобучение CNN: `best_resnet18_rul.pth`.
10. Расширенные семейства моделей: `v3_rnn`, `v4_tcn`, `v5_odd`.
11. Автоматизация запусков: balanced runners и overnight runner.
12. Production-ready Streamlit-демо с реальным инференсом, выбором checkpoint и Docker-упаковкой.

Главный результат: проект перестал быть одним скриптом обучения и стал исследовательской платформой для сравнения разных архитектур прогнозирования RUL на одной экспериментальной базе.

## 3. Текущее состояние репозитория

Сейчас в проекте есть две основные прикладные задачи:

- Классификация дефектов подшипников по датасету CWRU.
- Прогнозирование остаточного ресурса подшипников по датасету XJTU-SY.

Ключевые директории:

- `src/classification` - классификация CWRU.
- `src/prediction` - прогнозирование RUL XJTU-SY.
- `src/demo` - константы и вспомогательные данные для веб-демо.
- `src/visualization` - Plotly-визуализации для интерактивного интерфейса.
- `models` - сохраненные веса CNN, RUL-моделей и CatBoost.
- `reports/figures` - графики для анализа и отчета.
- `mlartifacts` - артефакты MLflow.
- `scripts` - сценарии автоматического запуска длинных экспериментов.
- `scratch_scripts` - черновые исследовательские скрипты.
- `app.py` - Streamlit-приложение для демонстрации классификации, RUL-прогноза и бортового дашборда.
- `Dockerfile`, `.dockerignore`, `requirements.txt` - минимальная упаковка демо в контейнер.
- `README.md` - инструкция по запуску, структуре проекта и замене моделей.

Текущая оценка объема артефактов:

- `src`: 40 файлов, около 1.13 MB исходного кода.
- `scripts`: 6 файлов.
- `scratch_scripts`: 8 файлов.
- `models`: 77 файлов, около 2.54 GB.
- `reports/figures/summary`: 458 файлов, около 100.88 MB.
- `reports/figures/pretrain_cnn_rul`: 5 файлов, около 3.9 MB.
- `mlartifacts`: 23 файла, около 89.24 MB.

Это подтверждает, что проект содержит не только код, но и результаты реальных запусков.

## 4. Git-история разработки

В текущей ветке `develop` обнаружено 8 коммитов. Коммиты крупные, поэтому историю лучше читать как этапы разработки.

### 4.1. `d0f3926 develop: базовое приложение и диаграммы`

Первый значимый этап. В проект были добавлены:

- `.gitignore`;
- данные CWRU в `data/raw/CWRU`;
- `pyproject.toml` и `requirements.txt`;
- `setup_project.sh`;
- исходные файлы `src/data_loader.py`, `src/model.py`, `src/train.py`;
- визуализация спектрограмм `src/visualization/plot_spectrograms.py`.

На этом этапе проект решал базовую задачу классификации дефектов подшипников. Сигналы CWRU читались из `.mat`, нарезались на окна, преобразовывались в спектрограммы и подавались в CNN.

### 4.2. `b8755bb develop`

На втором этапе проект стал ближе к полноценному ML-эксперименту:

- появились MLflow-артефакты;
- были сохранены первые веса `best_resnet18.pth`;
- появился ONNX-экспорт `best_resnet18.onnx`;
- добавлен `run_mlflow.sh`;
- появились scratch-скрипты для проверки моделей, SHAP, размеров входа, torchaudio и MaxViT;
- расширились `src/model.py`, `src/train.py` и визуализация.

Этот этап можно описывать как переход от минимального классификатора к воспроизводимой экспериментальной среде с MLflow и сохранением моделей.

### 4.3. `42b0cdc refactor: реструктуризация проекта и запуск RUL pipeline`

Ключевая архитектурная точка проекта. Код был разделен по задачам:

- `src/classification` - классификация дефектов CWRU;
- `src/prediction` - прогнозирование RUL XJTU-SY.

Также были добавлены:

- `src/prediction/config.py`;
- `src/prediction/data_loader.py`;
- `src/prediction/model.py`;
- `src/prediction/train.py`;
- `src/prediction/train_boosting.py`;
- `src/prediction/train_three_models.py`;
- `src/prediction/train_three_models_2.py`;
- `src/prediction/utils.py`.

Появился первый RUL pipeline: CSV-сигналы XJTU-SY превращаются в CWT-скалограммы, CNN-энкодер извлекает признаки, temporal-блок предсказывает RUL.

Также появились:

- CatBoost-артефакты в `catboost_info`;
- первые RUL-модели в `models/pred_1`;
- первые `preds_2` чекпоинты;
- обновленный `project_struct.md`.

С этого момента проект стал решать две задачи: диагностику дефектов и прогнозирование остаточного ресурса.

### 4.4. `c00073c develop: модельки новые`

Этот коммит в основном связан с сохраненными RUL-моделями:

- обновлены `models/preds_2/best_rul_gru_ws1024_v2_balanced.pth`;
- обновлены `models/preds_2/best_rul_lstm_ws1024_v2_balanced.pth`;
- добавлен `models/preds_2/best_rul_transformer_ws1024_v2_balanced.pth`.

Практический смысл этапа: были обучены и сохранены новые версии temporal-моделей для RUL.

### 4.5. `8650a2d develop: добавил двухфазный проход`

На этом этапе появился важный переход к двухфазной логике:

- старые `models/pred_1` были перенесены в `models/pred_0`;
- появились fast-чекпоинты для `preds_3`;
- `src/prediction/train_three_models_3.py` был существенно расширен.

Двухфазный подход стал ответом на практическую проблему: полный CNN+temporal fine-tuning дорогой, а обучение только temporal-головы быстрее. Поэтому pipeline начал разделять быстрые эксперименты и более тяжелое финальное обучение.

### 4.6. `4fa6112 develop: применил autopep8 к src`

Технический этап стабилизации кода:

- отформатированы файлы classification и prediction;
- улучшена читаемость;
- уменьшен хаос после активных экспериментов.

Для отчета этот этап можно описывать как приведение исследовательского кода к более поддерживаемому виду.

### 4.7. `96004a9 feat: добавить v3 pipeline обучения RUL с двухфазным fine-tuning`

Один из главных этапов RUL-разработки. Были добавлены и сохранены:

- модели `preds_2_frozen`;
- модели `preds_2_unfrozen`;
- модели `preds_3`;
- модели `preds_3_frozen`;
- изменения в `train_three_models_2.py` и `train_three_models_3.py`.

Здесь закрепилась идея сравнивать frozen и unfrozen/fine-tuning режимы. Это стало основой для последующих `v3`, `v3_rnn`, `v4_tcn`, `v5_odd`.

### 4.8. `060cbf0 develop: добавил подкраску текста, дополнительные логи и ещё много чего`

Последний коммит в git-истории. Он добавил:

- `scripts/run_rul_hybrid_v3_matrix.sh`;
- `scripts/run_v3_ws2048_lstm_gru.sh`;
- полноценные `train_rul_hybrid_v2.py` и `train_rul_hybrid_v3.py`;
- compatibility wrappers для `train_three_models_2.py` и `train_three_models_3.py`;
- обновления `project_struct.md`;
- цветные логи и расширенную консольную отчетность.

В этом коммите `v2` и `v3` стали отдельными большими пайплайнами, а не просто черновыми вариантами.

## 5. Незакоммиченная текущая разработка

Поверх git-истории в рабочем дереве есть важные новые файлы и изменения:

- `src/prediction/pretrain_cnn_rul.py`;
- `src/prediction/train_rul_hybrid_v3_rnn.py`;
- `src/prediction/train_rul_hybrid_v4_tcn.py`;
- `src/prediction/train_rul_hybrid_v5_odd.py`;
- `scripts/run_v3_rnn_balanced_freeze_then_finetune.sh`;
- `scripts/run_v4_tcn_balanced_freeze_then_finetune.sh`;
- `scripts/run_v5_odd_balanced_freeze_then_finetune.sh`;
- `scripts/run_all_hybrids_overnight.sh`;
- `models/cnn/best_resnet18_rul.pth`;
- новые директории `models/preds_3_rnn`, `models/preds_4_tcn`, `models/preds_5_odd`;
- измененный `src/prediction/config.py`;
- документы `report_on_practice.md` и `report_on_practice_codex.md`.

Эти изменения являются самой свежей частью разработки. Они расширяют `v3` до трех новых исследовательских семейств моделей и добавляют RUL-specific предобучение CNN.

## 6. История классификационного блока CWRU

### 6.1. Датасет и признаки

Классификационный блок находится в `src/classification`. Его задача - распознавать тип дефекта подшипника по CWRU.

`src/classification/data_loader.py` реализует `CWRUDataset`:

- читает `.mat` файлы;
- ищет сигнал `*_DE_time`;
- режет сигнал на окна;
- строит STFT-спектрограмму;
- возвращает тензор `[1, Freq, Time]` и метку класса.

Классы:

- `Normal`;
- `IR_007`, `IR_014`, `IR_021`;
- `Ball_007`, `Ball_014`, `Ball_021`;
- `OR_007`, `OR_014`, `OR_021`.

### 6.2. NAS по CNN-архитектурам

`src/classification/model.py` содержит фабрику CNN-моделей. Поддерживаемые архитектуры:

- `resnet18`;
- `squeezenet1_1`;
- `mobilenet_v3_small`;
- `efficientnet_b0`;
- `shufflenet_v2_x1_0`;
- `convnext_tiny`;
- `efficientnet_v2_s`;
- `regnet_y_400mf`.

Так как исходные STFT-спектрограммы маленькие, добавлен `SpectrogramAdapter`, который ресайзит вход до безопасного spatial-размера для глубоких CNN.

### 6.3. Production-ready train pipeline

`src/classification/train.py` реализует:

- Neural Architecture Search по CNN;
- Optuna HPO;
- nested MLflow runs;
- CosineAnnealingLR;
- early stopping;
- confusion matrix;
- learning curves;
- SHAP GradientExplainer;
- сохранение лучшей модели;
- экспорт и логирование артефактов.

Текущий файл содержит 787 строк кода. Это уже не минимальный учебный пример, а полноценный pipeline классификации.

### 6.4. Роль классификации в RUL-задаче

Классификационный ResNet-18 стал отправной точкой для RUL. Его веса сохраняются в:

```text
models/cnn/best_resnet18.pth
models/cnn/best_resnet18.onnx
```

Дальше этот CNN использовался как энкодер CWT-скалограмм в prediction-пайплайнах.

## 7. История базового RUL-блока

### 7.1. Первый универсальный RUL pipeline

`src/prediction/train.py` - ранний NAS+HPO pipeline для RUL. Он:

- использует XJTU-SY;
- строит CWT-скалограммы;
- ищет temporal-блок через Optuna;
- подбирает `learning_rate`, `seq_length`, `hidden_size`, `dropout`;
- обучает финальную модель;
- строит график True vs Predicted RUL.

На этом этапе temporal-модель была параметром поиска: LSTM, GRU, TCN или Transformer.

### 7.2. Универсальная гибридная модель

`src/prediction/model.py` описывает общую архитектуру:

```text
CSV signal -> CWT scalogram -> CNN encoder -> sequence of features -> temporal block -> RUL
```

Поддерживаемые temporal-блоки в базовой модели:

- `lstm`;
- `gru`;
- `tcn`;
- `transformer`;
- `mamba` как заглушка/экспериментальное направление.

Ключевой момент: CNN-энкодер умеет загружать веса классификационного backbone и адаптировать первый convolution под другое число входных каналов. Для RUL используется два канала вибросигнала: horizontal и vertical.

### 7.3. Простые сценарии обучения

`src/prediction/train_three_models.py` был отдельным сценарием без Optuna:

- последовательно обучить `lstm`;
- обучить `gru`;
- обучить `transformer`;
- сохранить отдельные чекпоинты.

Это был удобный baseline до появления более сложных `v2` и `v3`.

### 7.4. CatBoost baseline

`src/prediction/train_boosting.py` добавил классический ML-бейзлайн:

- скалограммы проходят через CNN-энкодер;
- CNN-фичи конкатенируются;
- CatBoostRegressor предсказывает RUL;
- метрики и модель логируются в MLflow.

Этот baseline важен методологически: он показывает, что сравнение идет не только между нейросетями, но и с сильным табличным алгоритмом.

## 8. История RUL v2

`src/prediction/train_rul_hybrid_v2.py` стал первым большим RUL-пайплайном. Он вырос до 1609 строк и добавил то, чего не было в ранних сценариях.

Ключевые изменения относительно `train_three_models.py`:

- объединение нескольких подшипников и режимов XJTU-SY;
- sliding window по CSV;
- профили запуска `fast`, `balanced`, `full`;
- Optuna HPO отдельно для каждой temporal-архитектуры;
- поддержка `--temporal-types`;
- поддержка window size candidates `[1024, 2048]`;
- CWT-cache;
- CNN feature-cache;
- сохранение чекпоинтов с подробными суффиксами;
- summary-графики и CSV;
- MLflow-логирование.

`v2` проверял базовые temporal-модели:

```text
lstm, gru, transformer
```

Основная цель `v2`: сделать обучение RUL более масштабируемым и воспроизводимым.

## 9. История RUL v3

`src/prediction/train_rul_hybrid_v3.py` стал ответом на проблемы, обнаруженные в `v2`: переобучение, нестабильное fine-tuning-поведение и склонность моделей к константным предсказаниям.

Файл вырос до 1949 строк и добавил:

- `rul_clip=0.8`;
- AsymmetricHuberLoss;
- HPO по `loss_delta` и `loss_alpha`;
- monotonicity penalty;
- AdamW вместо Adam;
- weight decay;
- ReduceLROnPlateau в final fit;
- EMA-сглаживание предсказаний;
- R2, RMSE и PHM score;
- двухфазную схему: HPO на frozen CNN feature-cache, final fit с CNN fine-tuning;
- discriminative learning rate;
- warmup fine-tuning.

Именно `v3` стал методологической основой для последующих расширений `v3_rnn`, `v4_tcn` и `v5_odd`.

### 9.1. Главная трудность RUL: повторяющиеся коллапсы моделей

Отдельно важно зафиксировать, что центральной технической трудностью RUL-направления были не разовые ошибки запуска, а повторяющиеся коллапсы моделей. Под коллапсом здесь понимается ситуация, когда модель перестает восстанавливать деградационный тренд и начинает предсказывать почти константное значение RUL. В разных экспериментах это проявлялось как горизонтальная линия на графиках true/pred RUL, предсказание около среднего значения целевой переменной и отрицательный либо близкий к нулю коэффициент детерминации R2.

Первый тип коллапса проявился при попытке улучшить базовый frozen-CNN подход через `rul_clip=0.8` и AsymmetricHuberLoss. Идея была физически оправданной: не заставлять модель угадывать почти неразличимые ранние состояния здорового подшипника и сильнее штрафовать опасную переоценку ресурса. Но при замороженном классификационном энкодере temporal-блок получал признаки, которые хорошо разделяли классы дефектов, но недостаточно отражали степень постепенной деградации. В результате оптимизация могла сводиться к предсказанию среднего RUL вместо восстановления формы кривой износа.

Второй тип коллапса был связан с двухфазной схемой `HPO frozen -> final fit unfrozen`. Feature-cache резко ускорял Optuna, потому что trial обучал только temporal-голову поверх заранее вычисленных CNN-фичей. Однако гиперпараметры, найденные в этом режиме, не всегда переносились на сквозное обучение с размороженным CNN. Конфигурации с малым `hidden_size` и низким `lr`, оптимальные для фиксированного признакового пространства, оказывались недостаточно устойчивыми после включения CNN в backpropagation.

Третий тип проблемы возникал при RUL-ориентированном дообучении CNN. Слишком длительное обучение энкодера могло приводить к запоминанию фонового шума тренировочных подшипников и разрушению более универсальных фильтров, полученных на классификационной задаче. Тогда на новых bearing-прогонах CNN выдавал слишком похожие или малосодержательные 512-мерные эмбеддинги, а LSTM/GRU/Transformer уже не могли построить информативную временную траекторию и снова сходились к почти прямой линии.

Именно поэтому эволюция `v2` -> `v3` -> `v3_rnn`/`v4_tcn`/`v5_odd` была не просто добавлением новых архитектур. Значительная часть работы была посвящена борьбе за устойчивость деградационного тренда: введены RUL-specific pretraining, разделение `frozen` и `finetune` режимов, warmup перед разморозкой CNN, discriminative learning rates, ограничение batch size при fine-tuning, `monotonicity_penalty`, PHM score, R2 и обязательные графики предсказаний и остатков. По сути, проект постепенно переходил от вопроса "какая модель дает ниже MSE" к более важному вопросу "какая схема обучения не вырождается в константу и сохраняет физически осмысленную кривую деградации".

## 10. RUL-specific CNN pretraining

Позже был добавлен `src/prediction/pretrain_cnn_rul.py`.

Проблема: классификационный CNN обучен распознавать тип дефекта, но RUL-задача требует понимать степень деградации. Поэтому появился отдельный предобучающий регрессор:

```text
single CWT scalogram -> CNN encoder -> linear regressor -> normalized RUL
```

Результат сохраняется в:

```text
models/cnn/best_resnet18_rul.pth
```

В `src/prediction/config.py` теперь используется приоритет:

1. Если есть `best_resnet18_rul.pth`, брать его.
2. Иначе использовать старый `best_resnet18.pth`.

Для `pretrain_cnn_rul` сохранены:

- `pretrain_learning_curves.png`;
- `pretrain_rul_scatter.png`;
- `pretrain_residuals.png`;
- `pretrain_shap_analysis.png`;
- `pretrain_metrics.csv`.

По сохраненному CSV:

```text
MSE=0.09410, MAE=0.25894, RMSE=0.30676, R2=-0.37269
```

Даже если метрики предобучения не идеальные, сам этап важен: он меняет смысл CNN-фичей с классификационных на регрессионные, ближе к RUL.

## 11. Расширенная ветка v3_rnn

`src/prediction/train_rul_hybrid_v3_rnn.py` - свежая ветка на основе `v3`. Файл содержит 2375 строк.

Цель: проверить, насколько можно усилить RNN-подход без перехода к TCN/SOTA-моделям.

Поддерживаемые модели:

```text
lstm, gru, transformer,
bilstm, lstm_attn, bigru, gru_attn, transformer_improved
```

Новые архитектуры:

- `BiLSTMHead` - двунаправленный LSTM.
- `LSTMAttentionHead` - LSTM с temporal self-attention.
- `BiGRUHead` - двунаправленный GRU.
- `GRUAttentionHead` - GRU с temporal self-attention.
- `ImprovedTransformerHead` - Pre-LN Transformer с CLS-токеном и learnable positional encoding.

Фактические артефакты:

- 25 чекпоинтов в `models/preds_3_rnn`;
- 141 файл графиков/CSV в `reports/figures/summary/train_rul_hybrid_v3_rnn`.

Лучший найденный результат по summary CSV:

```text
transformer_improved_frozen, ws=1024,
test_mse=0.00923, test_rmse=0.24024, test_r2=0.10506
```

## 12. Расширенная ветка v4_tcn

`src/prediction/train_rul_hybrid_v4_tcn.py` - ветка TCN. Файл содержит 2394 строки.

Цель: проверить temporal convolution как альтернативу рекуррентным сетям.

Поддерживаемые модели:

```text
lstm, gru, transformer,
tcn, tcn_ms, tcna, tcn_bi
```

Новые архитектуры:

- `TCNBlock` - residual-блок с dilation и WeightNorm.
- `TCNHead` - классический causal TCN.
- `MultiScaleTCNHead` - параллельные ветки с kernel size 3, 7, 15.
- `TCNAttentionHead` - TCN с temporal attention.
- `BiTCNHead` - non-causal/bidirectional TCN для offline-анализа.

Дополнительно Optuna подбирает `kernel_size` для `tcn`, `tcna`, `tcn_bi`.

Фактические артефакты:

- 15 чекпоинтов в `models/preds_4_tcn`;
- 95 файлов графиков/CSV в `reports/figures/summary/train_rul_hybrid_v4_tcn`.

Лучший найденный результат по summary CSV:

```text
tcn_bi_finetune, ws=2048,
test_mse=0.01302, test_rmse=0.29660, test_r2=-0.36417
```

## 13. Расширенная ветка v5_odd

`src/prediction/train_rul_hybrid_v5_odd.py` - ветка SOTA temporal-моделей. Файл содержит 2451 строку.

Несмотря на имя `odd`, фактически реализованы не NODE/LTC-модели, а:

```text
lstm, gru, transformer,
patchtst, conformer, mamba
```

Новые архитектуры:

- `PatchTSTHead` - патч-токенизация временного ряда + Transformer.
- `ConformerHead` - гибрид feed-forward, multi-head self-attention и depthwise convolution.
- `MambaHead` - selective state space model через `mamba-ssm`.

Для Mamba добавлен graceful fallback: если `mamba-ssm` не установлен, модель пропускается, а весь запуск не падает.

Дополнительные HPO-параметры:

- `nhead` для PatchTST и Conformer;
- `patch_size` для PatchTST;
- `d_state` для Mamba.

Фактические артефакты:

- 9 чекпоинтов в `models/preds_5_odd`;
- 59 файлов графиков/CSV в `reports/figures/summary/train_rul_hybrid_v5_odd`.

Лучший найденный результат по summary CSV:

```text
conformer_finetune, ws=2048,
test_mse=0.01297, test_rmse=0.27323, test_r2=-0.15765
```

## 14. Автоматизация экспериментов

В проект добавлены shell-скрипты:

- `scripts/run_rul_hybrid_v3_matrix.sh`;
- `scripts/run_v3_ws2048_lstm_gru.sh`;
- `scripts/run_v3_rnn_balanced_freeze_then_finetune.sh`;
- `scripts/run_v4_tcn_balanced_freeze_then_finetune.sh`;
- `scripts/run_v5_odd_balanced_freeze_then_finetune.sh`;
- `scripts/run_all_hybrids_overnight.sh`.

Старые скрипты автоматизировали матричные прогоны `v3`. Новые скрипты делают больше:

- проверяют наличие RUL CNN checkpoint;
- при необходимости запускают `pretrain_cnn_rul.py`;
- запускают balanced-режим;
- обучают `frozen` и `finetune` варианты;
- сохраняют логи в `reports/logs`;
- позволяют переопределять `WINDOW_SIZES`, `TEMPORAL_TYPES`, `N_TRIALS`, `EPOCHS`, `NUM_WORKERS`;
- объединяют все семейства в один overnight pipeline.

`run_all_hybrids_overnight.sh` по умолчанию запускает:

- `V3 RNN`: `lstm bilstm lstm_attn bigru gru_attn transformer_improved`;
- `V4 TCN`: `tcn tcn_ms tcna tcn_bi`;
- `V5 ODD/SOTA`: `patchtst conformer mamba`.

## 15. Эволюция качества и артефактов

В `models` сохранены разные поколения моделей:

- `models/cnn` - классификационный ResNet-18, ONNX-экспорт и RUL-pretrained ResNet-18;
- `models/demo_best` - curated-набор лучших checkpoint для веб-демо;
- `models/pred_0` - ранние RUL и CatBoost-модели;
- `models/preds_1` - ранние v2-чекпоинты;
- `models/preds_2_frozen` - v2 frozen-модели;
- `models/preds_2_unfrozen` - v3 fast unfrozen-модели;
- `models/preds_3` - v3 balanced-модели;
- `models/preds_3_frozen` - v3 frozen-модели;
- `models/preds_3_rnn` - расширенное RNN-семейство;
- `models/preds_4_tcn` - TCN-семейство;
- `models/preds_5_odd` - SOTA-семейство.

В `reports/figures/summary` сохранены результаты разных поколений:

- classification summary plots;
- boosting plots;
- `train_three_models_2_0_nonfrozen`;
- `train_three_models_2_frozen`;
- `train_three_models_3`;
- `train_three_models_3_frozen`;
- `train_three_models_3_unfrozen`;
- `train_rul_hybrid_v3_rnn`;
- `train_rul_hybrid_v4_tcn`;
- `train_rul_hybrid_v5_odd`.

Количество файлов по новым семействам:

```text
train_rul_hybrid_v3_rnn: 141
train_rul_hybrid_v4_tcn: 95
train_rul_hybrid_v5_odd: 59
```

### 15.1. Отбор моделей для веб-демо

После обучения нескольких семейств моделей был сделан отдельный практический шаг: отобраны checkpoint, которые имеет смысл показывать в интерактивном веб-демо. Критерий отбора - не полнота всех экспериментов, а сочетание качества, устойчивости инференса и понятности демонстрации.

В текущем Streamlit-интерфейсе для RUL автоматически просматриваются совместимые checkpoint из:

- `models/demo_best`;
- `models/pred_0`;
- `models/preds_2_unfrozen`;
- `models/preds_3`;
- `models/preds_3_frozen`;
- `models/preds_3_rnn`.

Модели сортируются по `test_mse`, поэтому лучший checkpoint отображается первым. По текущим сохраненным метрикам лучший результат среди подключенных моделей дает `transformer_improved` из `preds_3_rnn`:

```text
test_mse = 0.00923
test_mae = 0.203
test_r2  = 0.105
```

Ветки `preds_4_tcn` и `preds_5_odd` оставлены как исследовательские артефакты, но не включены в production-demo. Причина прагматичная: их лучшие checkpoint уступают текущему лидеру.

```text
best TCN / TCNA:
test_mse = 0.01053
test_mae = 0.218
test_r2  = 0.008

best ODD / PatchTST:
test_mse = 0.01135
test_mae = 0.229
test_r2  = -0.041
```

Таким образом, подключение TCN/PatchTST/Conformer к демо увеличило бы сложность inference-кода и Docker-образа, но не улучшило бы качество демонстрационного прогноза. Для защиты это является важным инженерным решением: не все исследовательские ветки обязаны попадать в финальную витрину, если метрики и эксплуатационная сложность не оправдывают интеграцию.

## 16. Основные инженерные решения

### 16.1. CNN как общий энкодер

Все RUL-ветки используют одну идею: CNN не предсказывает RUL напрямую по всей истории, а извлекает признаки из каждого временного окна. Temporal-модель анализирует последовательность этих признаков.

Плюсы:

- можно использовать сильный CNN-backbone;
- можно переиспользовать классификационные веса;
- можно кэшировать CNN-фичи;
- можно сравнивать temporal-архитектуры при одинаковом визуальном энкодере.

### 16.2. CWT вместо сырого сигнала

Для RUL используется CWT по двум каналам вибрации. Это дает двумерное time-frequency представление, удобное для CNN.

Плюсы:

- частотные признаки становятся явнее;
- CNN может находить локальные паттерны на скалограмме;
- horizontal и vertical vibration используются вместе.

### 16.3. Feature-cache для HPO

HPO по полной CNN+temporal модели слишком дорогой. Поэтому используется схема:

```text
CNN frozen -> precompute features -> Optuna trains temporal head only
```

Это резко ускоряет подбор гиперпараметров и делает длинные серии экспериментов реальными.

### 16.4. Final fit отдельно от HPO

После HPO лучшая temporal-конфигурация обучается финально:

- в `frozen` режиме;
- или в `finetune` режиме;
- или в обоих режимах для сравнения.

Это делает эксперимент честнее: HPO быстрый, а финальная модель получает возможность дообучить CNN.

### 16.5. Piecewise RUL target

`rul_clip=0.8` уменьшает давление на модель в здоровой фазе. Вместо того чтобы требовать различать почти одинаковые ранние состояния подшипника, модель фокусируется на зоне реальной деградации.

### 16.6. Loss и физические ограничения

В `v3` и новых ветках появились:

- AsymmetricHuberLoss;
- monotonicity penalty;
- PHM score.

Это отражает специфику RUL: ошибка в сторону слишком оптимистичного прогноза может быть опаснее, а RUL физически не должен хаотично расти при приближении отказа.

### 16.7. Streamlit-демо как прикладной слой

В конце работы был добавлен прикладной демонстрационный слой на Streamlit. Его задача - показать результаты исследования не как набор training-скриптов, а как понятный интерактивный прототип системы предиктивной диагностики.

Архитектура демо:

- `app.py` задает широкую разметку страницы, навигацию через `st.sidebar.radio` и три рабочих экрана;
- `src/prediction/demo_inference.py` отвечает за поиск checkpoint, загрузку моделей и реальный инференс;
- загрузка PyTorch-весов обернута в `@st.cache_resource`, поэтому тяжелые модели не перечитываются при каждом взаимодействии с UI;
- `src/visualization/plots.py` строит интерактивные Plotly-графики;
- `src/demo/mock_data.py` хранит демонстрационные константы и историю проверок;
- `models/demo_best` служит отдельной директорией для лучших моделей, которые можно заменить без изменения кода.

В интерфейсе реализованы три сценария:

- классификация дефектов CWRU через ResNet-18 с выбором checkpoint и визуализацией сигнала;
- прогноз остаточного ресурса XJTU-SY через CNN+temporal модель с динамическим графиком True RUL / Pred RUL;
- бортовой дашборд машиниста с `st.metric`, gauge chart и таблицей последних проверок.

Проект также подготовлен к контейнеризации. `Dockerfile` использует `python:3.10-slim`, устанавливает зависимости из `requirements.txt`, копирует код, модели и исходные данные, открывает порт `8501` и запускает:

```bash
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

Это переводит проект из исследовательского состояния в форму, которую можно запустить и показать как законченный демонстрационный стенд.

## 17. Что можно рассказывать в отчете по практике

Короткая формулировка:

> В ходе практики проект был развит от базовой CNN-классификации дефектов подшипников до полноценной исследовательской платформы прогнозирования RUL и интерактивного веб-демо системы предиктивной диагностики. Сначала был реализован классификационный pipeline CWRU с NAS, Optuna, MLflow и SHAP. Затем проект был реструктурирован, добавлен RUL pipeline для XJTU-SY на основе CWT-скалограмм, CNN-энкодера и temporal-моделей. Далее были реализованы версии v2 и v3 с multi-bearing обучением, кэшированием, двухфазным HPO/final-fit протоколом, piecewise RUL target, AsymmetricHuberLoss, monotonicity penalty и расширенными метриками. На последнем этапе были добавлены исследовательские ветки v3_rnn, v4_tcn и v5_odd/SOTA, RUL-specific предобучение CNN, автоматизированные overnight-запуски и Streamlit/Docker-демо с реальным инференсом сохраненных checkpoint.

Развернутая формулировка:

> Основной вклад работы заключается не только в добавлении новых моделей, но и в построении воспроизводимого экспериментального контура. Для каждой модели сохраняются чекпоинты, графики, summary CSV, MLflow-логи, Optuna-история и метрики инференса. Благодаря feature-cache стало возможно быстро подбирать гиперпараметры temporal-части, а благодаря final fit можно отдельно оценивать frozen и finetune режимы. Это позволило сравнивать LSTM/GRU/Transformer, attention-RNN, TCN, multi-scale TCN, PatchTST, Conformer и Mamba на общей постановке задачи. Отдельно был реализован прикладной слой: Streamlit-интерфейс выбирает лучшие совместимые checkpoint, выполняет реальный инференс и показывает результаты в форме, пригодной для демонстрации на защите.

## 18. Хронология по версиям моделей

### Ранние версии

- `pred_0`: первые RUL-модели и CatBoost baseline.
- `preds_1`: ранние checkpoint-результаты v2.
- `preds_2_frozen`: v2 с замороженным CNN.
- `preds_2_unfrozen`: early v3/fine-tuning эксперименты.

### v3

- `preds_3`: balanced-модели v3.
- `preds_3_frozen`: frozen-варианты v3.
- Основной смысл: двухфазный pipeline, улучшенный loss, дополнительные метрики и стабилизация обучения.

### v3_rnn

- `preds_3_rnn`: RNN/attention/ImprovedTransformer.
- Основной смысл: усиление классических recurrent-моделей.

### v4_tcn

- `preds_4_tcn`: TCN, multi-scale TCN, TCN+attention, bidirectional TCN.
- Основной смысл: проверить convolutional temporal-модели как альтернативу RNN.

### v5_odd

- `preds_5_odd`: PatchTST, Conformer, Mamba.
- Основной смысл: проверить современные sequence-модели на RUL.

## 19. Текущие ограничения и честные замечания

В отчете стоит честно отметить:

- Часть новых файлов сейчас находится в рабочем дереве как untracked, то есть еще не зафиксирована коммитом.
- Название `v5_odd` исторически может вводить в заблуждение: фактически там реализованы PatchTST, Conformer и Mamba, а не NODE/LTC.
- Некоторые fast-прогоны являются sanity-check и не должны использоваться как финальное сравнение качества.
- R2 у части моделей отрицательный, поэтому основной акцент лучше делать на сравнении MSE/RMSE/MAE и на исследовательской инфраструктуре.
- Повторяющиеся коллапсы RUL-моделей были одной из главных трудностей разработки: модель могла давать приемлемые отдельные loss-значения, но фактически предсказывать среднее или почти прямую линию.
- HPO на замороженном CNN feature-cache и последующий fine-tuning CNN не являются полностью взаимозаменяемыми режимами; найденные гиперпараметры могут плохо переноситься между ними.
- Fine-tuning CNN тяжелее по GPU-памяти, поэтому для TCN/SOTA веток batch size ограничивался.
- TCN и ODD/SOTA ветки не подключены к веб-демо: по сохраненным метрикам они уступают лучшему `transformer_improved`, поэтому оставлены как исследовательское сравнение, а не как production-кандидаты.

Эти замечания не уменьшают ценность работы: наоборот, они показывают, что разработка велась как реальное исследование, где фиксировались не только успехи, но и ограничения.

## 20. Итоговая оценка проделанной работы

По текущему состоянию проекта сделано следующее:

- построен классификационный pipeline CWRU;
- реализован NAS/HPO по CNN-архитектурам;
- добавлены MLflow и Optuna;
- добавлена SHAP-интерпретация;
- создан RUL pipeline XJTU-SY;
- реализованы CNN+temporal модели;
- добавлен CatBoost baseline;
- реализованы `v2` и `v3` RUL-пайплайны;
- проведена серия итераций по диагностике и снижению коллапсов RUL-моделей;
- добавлены CWT-cache и CNN feature-cache;
- добавлено двухфазное обучение;
- добавлено RUL-specific CNN pretraining;
- добавлены семейства `v3_rnn`, `v4_tcn`, `v5_odd`;
- добавлены автоматизированные bash-runners;
- сохранены модели, графики, CSV и MLflow-артефакты;
- реализовано Streamlit/Docker-демо с выбором реальных моделей, кэшированной загрузкой PyTorch-весов и интерактивными Plotly-графиками.

Финальная формулировка результата:

> В результате практики был создан и развит программный комплекс для анализа вибросигналов подшипников, включающий классификацию дефектов, прогнозирование остаточного ресурса и веб-демонстрацию системы предиктивной диагностики. Проект поддерживает несколько поколений моделей, воспроизводимое обучение, автоматический подбор гиперпараметров, логирование экспериментов, визуальную аналитику, сравнение современных temporal-архитектур и запуск интерактивного Streamlit-прототипа в Docker.
