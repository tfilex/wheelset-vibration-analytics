#!/bin/bash

# Создание структуры директорий
mkdir -p data/raw
mkdir -p data/processed
mkdir -p src

# Создание пустых файлов исходного кода
touch src/data_loader.py
touch src/preprocess.py
touch src/model.py
touch src/train.py

echo "Базовая структура проекта успешно создана!"
