"""
train.py — Production-ready NAS-пайплайн для классификации дефектов CWRU.

Реализует:
    - Neural Architecture Search: 5 torchvision-архитектур
    - Optuna HPO с nested MLflow runs
    - CosineAnnealingLR + early stopping для стабильного обучения
    - Сохранение лучшей модели на диск и в MLflow
    - Публикационные графики: confusion matrix, learning curves
    - SHAP-интерпретируемость (GradientExplainer)

Usage:
    uv run python src/train.py
"""

import copy
import os
import sys
import warnings
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import optuna
import seaborn as sns
import shap
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio.transforms as T
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader, Subset

# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import CWRUDataset  # noqa: E402
from model import get_model, SUPPORTED_MODELS  # noqa: E402
# ---------------------------------------------------------------------------

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="shap")

# Ускорение сверток (т.к. размер входа всегда 1x129x9)
torch.backends.cudnn.benchmark = True

# ========================== CONFIGURATION ==================================

RANDOM_SEED: int = 13
DATA_DIR: str = "data/raw/CWRU"
FIGURES_DIR: str = "reports/figures/summary"
MODELS_DIR: str = "models"

N_TRIALS: int = 24       # Было 30
HPO_EPOCHS: int = 10     # Уменьшили эпохи для HPO (было 25)
FINAL_EPOCHS: int = 25   # Эпохи для финального обучения
NUM_CLASSES: int = 10
PATIENCE: int = 5        # Early stopping patience

CLASS_NAMES: List[str] = [
    "Normal",
    "IR_007", "IR_014", "IR_021",
    "Ball_007", "Ball_014", "Ball_021",
    "OR_007", "OR_014", "OR_021",
]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLFLOW_TRACKING_URI: str = f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}"
MLFLOW_EXPERIMENT: str = "CWRU_NAS_HPO_D070526_3"


# ========================== UTILITY ========================================


def get_device() -> torch.device:
    """Выбор устройства."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Устройство: {device}")
    return device


def load_and_split_data(
    seed: int = RANDOM_SEED,
) -> Tuple[Subset, Subset, Subset]:
    """Загрузка CWRU и стратифицированное разбиение 70 / 15 / 15."""
    print("[INFO] Загрузка датасета CWRU...")
    dataset = CWRUDataset(data_dir=DATA_DIR)
    total = len(dataset)
    if total == 0:
        raise RuntimeError(f"Датасет пуст. Проверьте {DATA_DIR}/")

    labels = dataset.labels
    indices = list(range(total))

    train_idx, temp_idx, _, temp_labels = train_test_split(
        indices, labels, train_size=0.70, stratify=labels, random_state=seed
    )
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, temp_labels, test_size=0.5, stratify=temp_labels, random_state=seed
    )

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

    print(
        f"[INFO] Split: train={len(train_ds)}, "
        f"val={len(val_ds)}, test={len(test_ds)} (total={total})"
    )
    return train_ds, val_ds, test_ds


# ========================== TRAINING HELPERS ===============================


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Одна эпоха обучения. Возвращает (avg_loss, accuracy)."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    # Initialize scaler, handle cpu case
    scaler = torch.amp.GradScaler(device.type) if device.type == 'cuda' else None
    
    time_masking = T.TimeMasking(time_mask_param=2)
    freq_masking = T.FrequencyMasking(freq_mask_param=15)
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # SpecAugment (Time & Frequency Masking)
        inputs = time_masking(inputs)
        inputs = freq_masking(inputs)
        
        optimizer.zero_grad()
        
        # Mixed Precision
        if device.type == 'cuda':
            with torch.amp.autocast(device.type):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return running_loss / total, correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, List[int], List[int]]:
    """Оценка модели. Возвращает (avg_loss, accuracy, f1_macro, preds, labels)."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds: List[int] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            
    f1 = f1_score(all_labels, all_preds, average="macro")
    return running_loss / total, correct / total, f1, all_preds, all_labels


# ========================== OPTUNA OBJECTIVE ===============================


def objective(
    trial: optuna.Trial,
    train_ds: Subset,
    val_ds: Subset,
    device: torch.device,
) -> float:
    """Optuna objective: один trial = один nested MLflow run (Train/Val split)."""
    model_name = trial.suggest_categorical("model_name", SUPPORTED_MODELS)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    scheduler_name = trial.suggest_categorical("scheduler", ["CosineAnnealingLR", "ReduceLROnPlateau"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = get_model(model_name, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt_cls = optim.Adam if optimizer_name == "Adam" else optim.AdamW
    optimizer = opt_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    if scheduler_name == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=HPO_EPOCHS)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_val_f1 = 0.0

    with mlflow.start_run(
        run_name=f"trial_{trial.number:03d}_{model_name}", nested=True
    ):
        mlflow.log_params({
            "model_name": model_name,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "optimizer": optimizer_name,
            "scheduler": scheduler_name,
            "epochs": HPO_EPOCHS,
            "cv_folds": 1,
        })

        for epoch in range(HPO_EPOCHS):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc, val_f1, _, _ = evaluate(
                model, val_loader, criterion, device
            )
            
            if scheduler_name == "ReduceLROnPlateau":
                scheduler.step(val_f1)
            else:
                scheduler.step()

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                
            mlflow.log_metrics({
                "val_loss": val_loss,
                "val_f1": val_f1,
            }, step=epoch)

            # Pruning on the epoch level
            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mlflow.log_metric("best_val_f1", best_val_f1)

    print(
        f"  Trial {trial.number:03d} | {model_name:24s} | "
        f"lr={lr:.2e}, wd={weight_decay:.2e}, bs={batch_size}, opt={optimizer_name}, sched={scheduler_name} "
        f"→ val_f1={best_val_f1:.4f}"
    )
    return best_val_f1


# ========================== VISUALIZATION ==================================


def plot_confusion_matrix(
    labels: List[int],
    preds: List[int],
    class_names: List[str],
    save_path: str,
) -> plt.Figure:
    """Матрица ошибок в академическом стиле (dpi=300)."""
    cm = confusion_matrix(labels, preds, labels=range(len(class_names)))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        cm_pct = np.nan_to_num(cm_pct)

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)"

    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.titlesize": 14, "axes.labelsize": 12,
    })

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm, annot=annot, fmt="", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Количество образцов"}, ax=ax,
    )
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title("Матрица ошибок классификации дефектов подшипников (CWRU)")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Матрица ошибок: {save_path}")
    return fig


def plot_learning_curves(
    history: Dict[str, List[float]],
    save_path: str,
    model_name: str = "",
) -> plt.Figure:
    """Кривые обучения (Loss + Accuracy) в академическом стиле."""
    epochs_range = range(1, len(history["train_loss"]) + 1)

    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.titlesize": 14, "axes.labelsize": 12,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs_range, history["train_loss"], "o-", label="Train Loss",
             color="#2563eb", markersize=4, linewidth=1.5)
    ax1.plot(epochs_range, history["val_loss"], "s-", label="Val Loss",
             color="#dc2626", markersize=4, linewidth=1.5)
    ax1.set_xlabel("Эпоха")
    ax1.set_ylabel("Loss (CrossEntropy)")
    ax1.set_title("Кривые потерь")
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, history["train_acc"], "o-", label="Train Accuracy",
             color="#2563eb", markersize=4, linewidth=1.5)
    ax2.plot(epochs_range, history["val_acc"], "s-", label="Val Accuracy",
             color="#dc2626", markersize=4, linewidth=1.5)
    ax2.set_xlabel("Эпоха")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Кривые точности")
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.0, 1.05])

    title = "Кривые обучения финальной модели"
    if model_name:
        title += f" ({model_name})"
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Кривые обучения: {save_path}")
    return fig


# ========================== SHAP ANALYSIS ==================================


def analyze_with_shap(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    save_path: str,
    n_background: int = 50,
    n_explain: int = 8,
) -> None:
    """SHAP GradientExplainer для CNN-спектрограмм.

    Строит 2-рядную сетку: верхний ряд — спектрограммы, нижний — SHAP overlay.
    """
    model.eval()

    all_inputs, all_labels = [], []
    for inputs, labels in test_loader:
        all_inputs.append(inputs)
        all_labels.append(labels)
    all_inputs = torch.cat(all_inputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    n_bg = min(n_background, len(all_inputs))
    n_ex = min(n_explain, len(all_inputs))
    background = all_inputs[:n_bg].to(device)
    explain_samples = all_inputs[n_bg: n_bg + n_ex].to(device)
    explain_labels = all_labels[n_bg: n_bg + n_ex]

    print(f"[INFO] SHAP: background={n_bg}, explain={n_ex}")

    explainer = shap.GradientExplainer(model, background)
    shap_values_raw = explainer.shap_values(explain_samples)

    # Нормализация формата SHAP values
    if isinstance(shap_values_raw, list):
        shap_all = np.stack(shap_values_raw, axis=-1)
    else:
        shap_all = shap_values_raw
        if isinstance(shap_all, torch.Tensor):
            shap_all = shap_all.cpu().numpy()

    has_class_dim = (shap_all.ndim == 5)
    explain_np = explain_samples.cpu().numpy()

    plt.rcParams.update({"font.family": "serif", "font.size": 10})

    n_cols = n_ex
    fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    im = None
    for i in range(n_ex):
        true_label = explain_labels[i].item()
        spec = explain_np[i, 0]

        if has_class_dim:
            shap_map = shap_all[i, 0, :, :, true_label]
        else:
            shap_map = shap_all[i, 0]

        ax_top = axes[0, i]
        ax_top.imshow(spec, aspect="auto", origin="lower", cmap="viridis")
        ax_top.set_title(f"{CLASS_NAMES[true_label]}", fontsize=9)
        ax_top.set_ylabel("Частотный бин" if i == 0 else "")
        ax_top.set_xlabel("")
        ax_top.tick_params(labelsize=7)

        ax_bot = axes[1, i]
        ax_bot.imshow(spec, aspect="auto", origin="lower", cmap="gray",
                      alpha=0.4)
        im = ax_bot.imshow(
            shap_map, aspect="auto", origin="lower", cmap="jet", alpha=0.7,
        )
        ax_bot.set_ylabel("Частотный бин" if i == 0 else "")
        ax_bot.set_xlabel("Временной фрейм")
        ax_bot.tick_params(labelsize=7)

    fig.suptitle(
        "SHAP-анализ: вклад частотных полос в классификацию CNN",
        fontsize=13, fontweight="bold",
    )

    plt.tight_layout()
    fig.subplots_adjust(top=0.90, right=0.92)

    # Вертикальный colorbar справа (без наезда на графики)
    if im is not None:
        cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical", label="SHAP value")
        cbar.ax.tick_params(labelsize=8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] SHAP-анализ: {save_path}")


# ========================== MAIN PIPELINE ==================================


def main() -> None:
    """Основной пайплайн:

    1. Load & split data (70 / 15 / 15)
    2. Optuna NAS + HPO (Parent MLflow Run + nested trials)
    3. Final training с лучшими параметрами (train+val), CosineAnnealing,
       early stopping, сохранение лучшего чекпоинта
    4. Оценка на test set + публикационные графики + SHAP
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = get_device()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    train_ds, val_ds, test_ds = load_and_split_data(seed=RANDOM_SEED)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # ================================================================
    # PHASE 1: Optuna NAS + HPO
    # ================================================================
    print("\n" + "=" * 60)
    print("  PHASE 1: Neural Architecture Search + HPO (Optuna)")
    print("=" * 60)

    with mlflow.start_run(run_name="NAS_HPO_Study") as parent_run:
        mlflow.log_params({
            "n_trials": N_TRIALS,
            "epochs_per_trial": HPO_EPOCHS,
            "random_seed": RANDOM_SEED,
            "dataset": "CWRU",
            "num_classes": NUM_CLASSES,
            "architectures": str(SUPPORTED_MODELS),
        })

        study = optuna.create_study(
            direction="maximize",
            study_name="cwru_nas",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5, n_warmup_steps=0
            ),
        )
        study.optimize(
            lambda trial: objective(trial, train_ds, val_ds, device),
            n_trials=N_TRIALS,
            show_progress_bar=True,
        )

        best = study.best_trial
        bp = best.params
        print(f"\n[RESULT] Best trial #{best.number}: "
              f"val_f1={best.value:.4f}")
        print(f"[RESULT] Best params: {bp}")

        mlflow.log_params({f"best_{k}": v for k, v in bp.items()})
        mlflow.log_metric("best_val_f1", best.value)

        # ============================================================
        # PHASE 2: Final Training
        # ============================================================
        print("\n" + "=" * 60)
        print(f"  PHASE 2: Final Training — {bp['model_name']}")
        print("=" * 60)

        best_model_name = bp["model_name"]
        best_lr = bp["lr"]
        best_wd = bp["weight_decay"]
        best_bs = bp["batch_size"]
        best_opt = bp["optimizer"]
        best_scheduler = bp["scheduler"]

        # Для честного финального обучения создадим cv_ds
        cv_indices = train_ds.indices + val_ds.indices
        cv_ds = Subset(train_ds.dataset, cv_indices)
        
        # Используем cv_ds (train+val), чтобы сделать честный стратифицированный сплит
        labels_cv = [cv_ds.dataset.labels[i] for i in cv_ds.indices]
        retrain_idx, reval_idx, _, _ = train_test_split(
            range(len(cv_ds)), labels_cv, test_size=0.12, stratify=labels_cv, random_state=RANDOM_SEED + 1
        )
        
        retrain_ds = Subset(cv_ds, retrain_idx)
        reval_ds = Subset(cv_ds, reval_idx)

        retrain_loader = DataLoader(
            retrain_ds, batch_size=best_bs, shuffle=True, num_workers=4, pin_memory=True
        )
        reval_loader = DataLoader(
            reval_ds, batch_size=best_bs, shuffle=False, num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            test_ds, batch_size=best_bs, shuffle=False, num_workers=4, pin_memory=True
        )

        final_model = get_model(
            best_model_name, num_classes=NUM_CLASSES
        ).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        opt_cls = optim.Adam if best_opt == "Adam" else optim.AdamW
        final_optimizer = opt_cls(final_model.parameters(), lr=best_lr, weight_decay=best_wd)
        
        if best_scheduler == "CosineAnnealingLR":
            final_scheduler = optim.lr_scheduler.CosineAnnealingLR(final_optimizer, T_max=FINAL_EPOCHS)
        else:
            final_scheduler = optim.lr_scheduler.ReduceLROnPlateau(final_optimizer, mode='max', factor=0.5, patience=2)

        best_val_f1 = 0.0
        epochs_no_improve = 0
        best_state_dict = None

        print("\n[INFO] Обучение финальной модели...")
        history: Dict[str, List[float]] = {
            "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
            "val_f1": [],
        }


        with mlflow.start_run(
            run_name=f"Final_{best_model_name}", nested=True
        ):
            mlflow.log_params({
                "model_name": best_model_name,
                "lr": best_lr,
                "weight_decay": best_wd,
                "batch_size": best_bs,
                "optimizer": best_opt,
                "epochs": FINAL_EPOCHS,
                "patience": PATIENCE,
                "scheduler": best_scheduler,
                "train_samples": len(retrain_ds),
                "val_samples": len(reval_ds),
                "test_samples": len(test_ds),
            })

            for epoch in range(FINAL_EPOCHS):
                train_loss, train_acc = train_one_epoch(
                    final_model, retrain_loader, criterion,
                    final_optimizer, device,
                )
                val_loss, val_acc, val_f1, _, _ = evaluate(
                    final_model, reval_loader, criterion, device,
                )
                
                if best_scheduler == "ReduceLROnPlateau":
                    final_scheduler.step(val_f1)
                else:
                    final_scheduler.step()

                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["train_acc"].append(train_acc)
                history["val_acc"].append(val_acc)
                history["val_f1"].append(val_f1)
                
                current_lr = final_optimizer.param_groups[0]['lr']

                mlflow.log_metrics({
                    "final_train_loss": train_loss,
                    "final_train_accuracy": train_acc,
                    "final_val_loss": val_loss,
                    "final_val_accuracy": val_acc,
                    "final_val_f1": val_f1,
                    "lr": current_lr,
                }, step=epoch)

                # ---- Early stopping: сохраняем лучший чекпоинт ----
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_state_dict = copy.deepcopy(
                        final_model.state_dict()
                    )
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                print(
                    f"  Эпоха [{epoch + 1:02d}/{FINAL_EPOCHS}] | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val F1: {val_f1:.4f}"
                    + (" ★" if epochs_no_improve == 0 else "")
                )

                if epochs_no_improve >= PATIENCE:
                    print(
                        f"[INFO] Early stopping на эпохе {epoch + 1} "
                        f"(patience={PATIENCE})"
                    )
                    break

            # ---- Загружаем лучший чекпоинт ----
            if best_state_dict is not None:
                final_model.load_state_dict(best_state_dict)
                print(
                    f"[INFO] Загружен лучший чекпоинт "
                    f"(val_f1={best_val_f1:.4f})"
                )

            # ---- Сохранение модели на диск ----
            model_path = os.path.join(
                MODELS_DIR, f"best_{best_model_name}.pth"
            )
            torch.save({
                "model_name": best_model_name,
                "num_classes": NUM_CLASSES,
                "state_dict": final_model.state_dict(),
                "best_val_f1": best_val_f1,
                "hyperparams": bp,
            }, model_path)
            print(f"[INFO] Модель сохранена: {model_path}")

            # ---- Evaluate on held-out TEST set ----
            print("\n[INFO] Оценка на тестовой выборке...")
            test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
                final_model, test_loader, criterion, device,
            )
            print(
                f"[RESULT] Test Loss: {test_loss:.4f} | "
                f"Test Accuracy: {test_acc:.4f} | "
                f"Test F1: {test_f1:.4f}"
            )

            mlflow.log_metrics({
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "test_f1": test_f1,
            })

            # ============================================================
            # PHASE 3: Publication Figures
            # ============================================================
            print("\n" + "=" * 60)
            print("  PHASE 3: Generating Publication Figures")
            print("=" * 60)

            cm_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
            fig_cm = plot_confusion_matrix(
                test_labels, test_preds, CLASS_NAMES, cm_path,
            )
            mlflow.log_artifact(cm_path, artifact_path="figures")
            plt.close(fig_cm)

            lc_path = os.path.join(FIGURES_DIR, "learning_curves.png")
            fig_lc = plot_learning_curves(
                history, lc_path, model_name=best_model_name
            )
            mlflow.log_artifact(lc_path, artifact_path="figures")
            plt.close(fig_lc)

            shap_path = os.path.join(FIGURES_DIR, "shap_analysis.png")
            analyze_with_shap(final_model, test_loader, device, shap_path)
            mlflow.log_artifact(shap_path, artifact_path="figures")

            # ---- Log model to MLflow ----
            print("\n[INFO] Логирование модели в MLflow...")
            mlflow.pytorch.log_model(final_model, "final_model")
            mlflow.log_artifact(model_path, artifact_path="checkpoints")

            # ---- ONNX Export ----
            print("\n[INFO] Экспорт модели в ONNX...")
            dummy_input, _ = next(iter(test_loader))
            dummy_input = dummy_input[:1].to(device)
            
            onnx_path = os.path.join(MODELS_DIR, f"best_{best_model_name}.onnx")
            torch.onnx.export(
                final_model, 
                dummy_input, 
                onnx_path, 
                export_params=True, 
                opset_version=13, 
                do_constant_folding=True, 
                input_names=['input'], 
                output_names=['output']
            )
            mlflow.log_artifact(onnx_path, artifact_path="production_models")
            print(f"[INFO] ONNX-модель сохранена: {onnx_path}")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Best architecture      : {best_model_name}")
    print(f"  Best HPO val_f1        : {best.value:.4f}")
    print(f"  Final test_f1          : {test_f1:.4f}")
    print(f"  Final test_accuracy    : {test_acc:.4f}")
    print(f"  Model saved to         : {model_path}")
    print(f"  Figures saved to       : {FIGURES_DIR}/")
    print(f"  MLflow experiment      : {MLFLOW_EXPERIMENT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
