import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Добавляем директорию src в пути импорта для загрузки соседних модулей
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import CWRUDataset
from model import LightweightCNN

def train():
    # Гиперпараметры и настройки
    EPOCHS = 20
    BATCH_SIZE = 32
    LR = 0.001
    DATA_DIR = 'data/raw' # предполагаем запуск из корня проекта
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    # 1. Загрузка и разбиение датасета
    print("Загрузка датасета...")
    dataset = CWRUDataset(data_dir=DATA_DIR)
    
    dataset_size = len(dataset)
    if dataset_size == 0:
        print("Ошибка: Датасет пуст. Убедитесь, что данные лежат в data/raw/")
        return
        
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Обучающая выборка: {len(train_dataset)} примеров")
    print(f"Валидационная выборка: {len(val_dataset)} примеров")

    # 2. Инициализация модели, функции потерь и оптимизатора
    model = LightweightCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 3. Настройка MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("CWRU_Baseline")

    # 4. Запуск эксперимента MLflow
    print("Начало обучения...")
    with mlflow.start_run(run_name="STFT_CNN_1HP"):
        
        # Логирование гиперпараметров
        mlflow.log_params({
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "optimizer": "Adam",
            "loss_function": "CrossEntropyLoss"
        })

        for epoch in range(EPOCHS):
            # Тренировочная фаза
            model.train()
            running_loss = 0.0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                
            train_loss = running_loss / train_size
            
            # Валидационная фаза
            model.eval()
            val_running_loss = 0.0
            correct = 0
            total = 0
            
            # Списки для построения матрицы ошибок на последней эпохе
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_running_loss += loss.item() * inputs.size(0)
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # Сохраняем предсказания и истинные метки (только для последней эпохи)
                    if epoch == EPOCHS - 1:
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        
            val_loss = val_running_loss / val_size
            val_accuracy = correct / total
            
            print(f"Эпоха [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
            
            # Логирование метрик каждую эпоху
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            }, step=epoch)

        # 5. Построение матрицы ошибок
        print("Построение и логирование матрицы ошибок...")
        cm = confusion_matrix(all_labels, all_preds)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.title('Confusion Matrix (Validation Data)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Сохранение графика в MLflow
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        # 6. Сохранение модели в MLflow
        print("Сохранение модели...")
        mlflow.pytorch.log_model(model, "model")
        
        print("Обучение завершено успешно!")

if __name__ == '__main__':
    train()
