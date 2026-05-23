# Console RUL Diagnostics

Консольный запуск RUL/Health Index диагностики без Streamlit-интерфейса.

## Запуск

```bash
uv run python console_diagnostics/run.py --bearing Bearing1_3
```

Старый путь тоже работает как обёртка:

```bash
uv run python experiments/run_offline_rul_diagnostics.py --bearing Bearing1_3
```

Цветной вывод:

```bash
uv run python console_diagnostics/run.py --bearing Bearing1_3 --color always
```

## Какие данные используются

Скрипт использует тестовые bearing-директории XJTU-SY из `src.prediction.demo_inference`:

- `Bearing1_3` -> `data/raw/XJTU-SY/35Hz12kN/Bearing1_3`
- `Bearing1_4` -> `data/raw/XJTU-SY/35Hz12kN/Bearing1_4`
- `Bearing2_5` -> `data/raw/XJTU-SY/37.5Hz11kN/Bearing2_5`
- `Bearing3_3` -> `data/raw/XJTU-SY/40Hz10kN/Bearing3_3`

По умолчанию используется сохранённая демо-модель:

```text
models/demo_best/rul/xjtu_rul.pth
```

## Что сохраняется

Для выбранного подшипника создаются:

- CSV с `true_rul`, `pred_rul`, `hi`, статусом и RUL в километрах:
  `results/offline_rul_diagnostics_<bearing>.csv`
- PNG-график RUL/HI:
  `figures/offline_rul_hi_<bearing>.png`
