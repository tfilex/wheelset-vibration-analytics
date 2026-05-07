#!/bin/bash

# Находим абсолютный путь к папке, где лежит этот скрипт (корень проекта)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Устанавливаем URI к локальной базе данных SQLite
DB_URI="sqlite:///$PROJECT_ROOT/mlflow.db"

echo "[INFO] Запуск MLflow UI с базой: $DB_URI"
echo "[INFO] MLflow будет доступен по адресу http://127.0.0.1:5000"

# Запускаем mlflow, используя uv, чтобы подцепить окружение
cd "$PROJECT_ROOT" && uv run mlflow ui --backend-store-uri "$DB_URI" --host 127.0.0.1 --port 5000
