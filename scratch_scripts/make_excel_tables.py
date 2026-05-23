import pandas as pd
import os

os.makedirs("/home/ish/rudn/VKR/reports/tables_for_vkr", exist_ok=True)

data_2_1 = [
    ["v3_rnn", "Transformer Improved", "Frozen", 1024, 0.0092, 0.2402, 0.1051, "Анализ долгосрочных зависимостей через механизм внимания"],
    ["v3_rnn", "BiLSTM", "Finetune", 1024, 0.0103, 0.2555, -0.0121, "Двунаправленное моделирование временных рядов"],
    ["v4_tcn", "BiTCN", "Finetune", 2048, 0.0130, 0.2966, -0.3642, "Сверточный анализ последовательностей с большим полем восприятия"],
    ["v5_odd", "Conformer", "Finetune", 2048, 0.0130, 0.2732, -0.1577, "Гибридное извлечение локальных и глобальных признаков (SOTA)"]
]
cols_2_1 = ["Семейство", "Модель", "Режим", "Окно", "Test MSE", "Test RMSE", "Test R2", "Назначение модели"]

data_2_2 = [
    ["Исходный код", "Python-модули пайплайнов, демо и утилиты (.py)", 40, "1.4 MB"],
    ["Shell скрипты", "Сценарии автоматизации и запуска обучения (.sh)", 10, "40 KB"],
    ["Чекпоинты", "Сохраненные веса моделей и экспорты (.pth, .onnx)", 80, "2.6 GB"],
    ["Графики", "Визуализации обучения, RUL-прогнозов и важности (.png)", 507, "124 MB"],
    ["CSV файлы", "Сводные метрики и результаты экспериментов (.csv)", 20, "<1 MB"],
    ["MLflow артефакты", "Данные трекинга экспериментов и логи", 23, "90 MB"]
]
cols_2_2 = ["Категория артефактов", "Описание / Тип файлов", "Кол-во (ориент.)", "Общий объем"]

df1 = pd.DataFrame(data_2_1, columns=cols_2_1)
df2 = pd.DataFrame(data_2_2, columns=cols_2_2)

# Try with default engine first, if openpyxl is not found it will fail
try:
    df1.to_excel("/home/ish/rudn/VKR/reports/tables_for_vkr/table_2_1_best_models.xlsx", index=False)
    df2.to_excel("/home/ish/rudn/VKR/reports/tables_for_vkr/table_2_2_artifacts.xlsx", index=False)
    print("Excel files created.")
except Exception as e:
    print(f"Error: {e}")
    print("Falling back to CSV...")
    df1.to_csv("/home/ish/rudn/VKR/reports/tables_for_vkr/table_2_1_best_models.csv", index=False)
    df2.to_csv("/home/ish/rudn/VKR/reports/tables_for_vkr/table_2_2_artifacts.csv", index=False)
    print("CSV files created instead.")
