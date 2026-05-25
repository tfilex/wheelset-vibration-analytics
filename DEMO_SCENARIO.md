# DEMO_SCENARIO

Пошаговый сценарий для показа проекта на защите. Он рассчитан на воспроизводимый demo-режим с локальными checkpoint и подготовленными CWRU/XJTU-SY данными.

## 1. Предварительная проверка

```bash
uv run pytest tests/test_demo_inference.py -q
uv run python -c "from src.prediction.demo_inference import discover_rul_models; print(discover_rul_models('demo')[0])"
```

Ожидаемый RUL checkpoint в demo-режиме:

```text
models/demo_best/rul/best_rul_transformer_improved_ws1024_v3rnn_train_rul_hybrid_v3_rnn_profilebalanced_trials30_epochs10_featurecache_on.pth
```

## 2. Запуск Streamlit

```bash
make demo PORT=8501
```

Открыть в браузере:

```text
http://localhost:8501
```

Если нужен Docker-сценарий:

```bash
make docker-build
make docker-run-demo PORT=8501
```

## 3. Экран классификации CWRU

1. В боковой панели оставить режим моделей `Prod`.
2. Открыть раздел `Классификация дефектов (CWRU)`.
3. Выбрать checkpoint `models/demo_best/classification/cwru_classifier.pth`.
4. Последовательно показать подготовленные сигналы:
   - `Норма`;
   - `Дефект внутреннего кольца`;
   - `Дефект внешнего кольца`.
5. Нажать `Выполнить диагностику` и прокомментировать результат: класс, confidence, исходный сигнал и карту важности.

Формулировка для защиты: интерактивное demo использует подготовленные CWRU `.mat` из `data/raw/CWRU/`; пользовательский upload произвольного файла в текущей версии не реализован. Карта важности является gradient * input attribution, не SHAP.

## 4. Экран RUL XJTU-SY

1. Открыть раздел `Прогноз ресурса RUL (XJTU-SY)`.
2. Оставить выбранной demo-модель ImprovedTransformer `...transformer_improved...epochs10...pth`.
3. Выбрать `Bearing1_3`.
4. Нажать `Запустить симуляцию деградации`.
5. Показать график `True RUL / Pred RUL`, затем блок Health Index, статус и демонстрационный остаточный ресурс в километрах.

Формулировка для защиты: RUL-график строится одной закрепленной моделью, ансамблевый доверительный интервал в UI не реализован. Километры являются демонстрационной инженерной шкалой на базе Health Index и slope.

## 5. Экран дашборда

1. Открыть `Бортовой модуль (Дашборд)`.
2. Показать gauge chart, текущий статус и историю проверок.
3. Уточнить, что это имитация интерфейса, а история берется из `src/demo/mock_data.py`; persistent storage и авторизация не реализованы.

## 6. Резервный CLI-сценарий

Если Streamlit недоступен, показать оффлайн-диагностику:

```bash
uv run python -m console_diagnostics.run --bearing Bearing1_3 --output-steps 20 --anchor-points 4 --color never
```

CLI сохраняет CSV и PNG в локальные каталоги `results/` и `figures/` и печатает checkpoint, Health Index, статус и RUL display.

## 7. Что подготовить как скриншоты

- CWRU `Норма`: сигнал, predicted class, confidence, карта важности.
- CWRU `Дефект внешнего кольца`: сигнал, predicted class, confidence, карта важности.
- XJTU-SY `Bearing1_3`: график RUL и блок Health Index.
- Дашборд: gauge chart и история проверок.
