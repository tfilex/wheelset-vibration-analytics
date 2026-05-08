"""
utils.py — Общие утилиты для модуля prediction (RUL).

Содержит функции, используемые в train.py и train_boosting.py,
чтобы избежать дублирования кода:
    - get_device: выбор устройства
    - train_one_epoch: одна эпоха обучения
    - evaluate: оценка модели (MSE, MAE, predictions)
    - plot_rul: генерация академического графика RUL
"""

import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error


# ========================== DEVICE =========================================


def get_device() -> torch.device:
    """Выбор вычислительного устройства (CUDA / CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Устройство: {device}")
    return device


# ========================== TRAINING HELPERS ===============================


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Одна эпоха обучения. Возвращает средний MSE Loss.

    Args:
        model: Модель для обучения.
        loader: DataLoader с обучающими данными.
        criterion: Функция потерь (MSELoss).
        optimizer: Оптимизатор.
        device: Устройство.

    Returns:
        Средний loss за эпоху.
    """
    model.train()
    running_loss, total = 0.0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        total += labels.size(0)

    return running_loss / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, List[float], List[float]]:
    """Оценка модели на валидационной/тестовой выборке.

    Args:
        model: Модель для оценки.
        loader: DataLoader с данными.
        criterion: Функция потерь.
        device: Устройство.

    Returns:
        Кортеж (avg_mse, mae, predictions, labels).
    """
    model.eval()
    running_loss, total = 0.0, 0
    all_preds: List[float] = []
    all_labels: List[float] = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            total += labels.size(0)

            all_preds.extend(outputs.cpu().numpy().flatten().tolist())
            all_labels.extend(labels.cpu().numpy().flatten().tolist())

    mae = mean_absolute_error(all_labels, all_preds)
    return running_loss / total, mae, all_preds, all_labels


# ========================== VISUALIZATION ==================================


def plot_rul(
    true_rul: List[float],
    pred_rul: List[float],
    save_path: str,
    model_name: str = "",
    pred_color: str = "#dc2626",
    pred_label: str = "Predicted RUL",
) -> None:
    """Генерация академического графика True vs Predicted RUL.

    Args:
        true_rul: Истинные значения RUL.
        pred_rul: Предсказанные значения RUL.
        save_path: Путь для сохранения.
        model_name: Название модели для заголовка.
        pred_color: Цвет линии предсказаний.
        pred_label: Подпись линии предсказаний в легенде.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })
    fig, ax = plt.subplots(figsize=(12, 6))

    x_axis = range(len(true_rul))
    ax.plot(x_axis, true_rul, label="True RUL", color="#2563eb",
            linewidth=2.5, alpha=0.9)
    ax.plot(x_axis, pred_rul, label=pred_label, color=pred_color,
            linestyle="--", linewidth=2, alpha=0.85)
    ax.fill_between(x_axis, true_rul, pred_rul, alpha=0.1, color=pred_color)

    ax.set_xlabel("Временные шаги (последовательности)")
    ax.set_ylabel("Нормализованный RUL")
    title = "Прогнозирование остаточного ресурса подшипника (RUL)"
    if model_name:
        title += f" — {model_name}"
    ax.set_title(title, fontweight="bold")
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] График RUL сохранён: {save_path}")


def plot_residuals(
    true_rul: List[float],
    pred_rul: List[float],
    save_path: str,
    model_name: str = "",
) -> None:
    """Генерация графика остатков (residuals) и распределения ошибок.

    Args:
        true_rul: Истинные значения.
        pred_rul: Предсказанные значения.
        save_path: Путь для сохранения.
        model_name: Название модели.
    """
    import seaborn as sns
    true_rul_arr = np.array(true_rul)
    pred_rul_arr = np.array(pred_rul)
    residuals = true_rul_arr - pred_rul_arr

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Scatter plot: True vs Predicted
    ax1.scatter(true_rul_arr, pred_rul_arr, alpha=0.5, color="#2563eb")
    ax1.plot([0, 1], [0, 1], '--', color="#dc2626", linewidth=2)
    ax1.set_xlabel("Истинный RUL")
    ax1.set_ylabel("Предсказанный RUL")
    ax1.set_title(f"True vs Predicted RUL ({model_name})")
    ax1.grid(True, alpha=0.3)

    # Histogram of residuals
    sns.histplot(residuals, kde=True, ax=ax2, color="#059669")
    ax2.set_xlabel("Ошибка (True - Pred)")
    ax2.set_ylabel("Частота")
    ax2.set_title("Распределение ошибок (Residuals)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] График Residuals сохранён: {save_path}")


def plot_learning_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: str,
    metric_name: str = "MSE",
) -> None:
    """Отрисовка кривых обучения (Loss vs Epochs)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, label=f"Train {metric_name}", color="#2563eb", linewidth=2)
    ax.plot(epochs, val_losses, label=f"Val {metric_name}", color="#dc2626", linewidth=2)
    
    ax.set_xlabel("Эпохи / Итерации")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Кривые обучения ({metric_name})", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Кривые обучения сохранены: {save_path}")
