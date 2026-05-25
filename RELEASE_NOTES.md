# Release v0.2.1

Дата: 2026-05-25

## Что изменилось

- Закреплен новый demo checkpoint для RUL-контура: `ImprovedTransformer` из семейства `v3_rnn` в режиме `finetune_cnn`.
- Добавлены интеграционные smoke-тесты demo inference в `tests/test_demo_inference.py`:
  - проверка demo-каталога моделей;
  - проверка выбора pinned RUL checkpoint;
  - короткий CWRU inference;
  - короткий XJTU-SY RUL inference.
- Добавлен pytest marker `slow` для интеграционных проверок, которые грузят локальные checkpoint и sample data.
- README дополнен иллюстрациями, таблицей топ-5 RUL-моделей, ссылками на защитные документы и разделом ограничений.
- Добавлены рабочие документы для подготовки к защите: `DATA_PASSPORT.md`, `DEMO_SCENARIO.md`, `CODE_COMPLETION_TASKS.md`, `IMPLEMENTATION_DESCRIPTION_TODO.md`, `FINAL_PROJECT_REVIEW.md`.

## Проверки

Последняя локальная проверка:

```text
uv run pytest -q
30 passed, 2 warnings

uv run pytest --cov=src --cov=console_diagnostics --cov-report=term-missing -q
30 passed, 2 warnings
TOTAL coverage: 56%
```

## Ограничения релиза

- RUL-контур остается исследовательским прототипом: значения `R²` для топ-5 моделей отрицательные.
- Веб-демо использует подготовленные CWRU/XJTU-SY данные через `selectbox`; загрузка пользовательских файлов не реализована.
- Ансамблевый доверительный интервал и промышленная трехосевая X/Y/Z-валидация не входят в этот релиз.
