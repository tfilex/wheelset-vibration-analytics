"""
train_boosting.py — Бустинг-бейзлайн для прогнозирования RUL (XJTU-SY).

Подход:
    1. Загружаем данные через RULDataset (PyTorch DataLoader).
    2. Пропускаем скалограммы через замороженный CNN-энкодер → вектор фичей.
    3. Обучаем CatBoostRegressor на фичах.
    4. Логируем метрики (MSE, MAE) и модель в MLflow.

Usage:
    uv run python src/prediction/train_boosting.py
"""

from utils import get_device, plot_rul, plot_residuals, plot_learning_curves
from model import create_cnn_encoder
from data_loader import RULDataset
from config import (
    RANDOM_SEED, TRAIN_DIR, VAL_DIR, TEST_DIR,
    FIGURES_DIR, MODELS_DIR, MLFLOW_TRACKING_URI,
    CNN_BACKBONE, CNN_IN_CHANNELS,
)
import os
import sys
import warnings
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    _CATBOOST_AVAILABLE = True
except ImportError:
    _CATBOOST_AVAILABLE = False

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

SEQ_LENGTH: int = 5
MLFLOW_EXPERIMENT: str = "XJTU_SY_RUL_Boosting"


# ========================== FEATURE EXTRACTION ==============================


def extract_features(
    dataset: RULDataset,
    encoder: nn.Module,
    device: torch.device,
    batch_size: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """Извлекает признаки из скалограмм через CNN-энкодер.

    Для каждого сэмпла извлекает CNN-фичи из каждого шага
    и конкатенирует их в один плоский вектор.

    Args:
        dataset: RULDataset для обработки.
        encoder: Замороженный CNN-энкодер.
        device: Вычислительное устройство.
        batch_size: Размер батча.

    Returns:
        features: np.ndarray (n_samples, seq_length * encoder_dim).
        targets: np.ndarray (n_samples,).
    """
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=2)
    encoder.eval()

    all_features: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            batch_size_actual, seq_len, channels, height, width = images.size()

            cnn_in = images.view(batch_size_actual *
                                 seq_len, channels, height, width)
            feats = encoder(cnn_in)
            feats = feats.view(batch_size_actual, -1).cpu().numpy()

            all_features.append(feats)
            all_targets.append(labels.numpy().flatten())

    return np.vstack(all_features), np.concatenate(all_targets)


def plot_catboost_feature_importance(model, save_path: str, top_n: int = 20):
    """Отрисовка важности признаков CatBoost."""
    import pandas as pd
    import seaborn as sns

    feature_importance = model.get_feature_importance()
    feature_names = [f"feat_{i}" for i in range(len(feature_importance))]

    df = pd.DataFrame(
        {'importance': feature_importance, 'name': feature_names})
    df = df.sort_values(by='importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='name', data=df, palette='viridis')
    plt.title(f"Top {top_n} Feature Importance (CatBoost)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ========================== MAIN PIPELINE ==================================


def main() -> None:
    """Пайплайн: CNN feature extraction → CatBoost → MLflow."""
    if not _CATBOOST_AVAILABLE:
        print("[ERROR] CatBoost не установлен. Установите: uv add catboost")
        sys.exit(1)

    np.random.seed(RANDOM_SEED)
    device = get_device()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # --- CNN-энкодер ---
    encoder, enc_dim = create_cnn_encoder(
        backbone_name=CNN_BACKBONE,
        in_channels=CNN_IN_CHANNELS,
        pretrained=False, freeze=True,
    )
    encoder = encoder.to(device)
    print(f"[INFO] CNN-энкодер: {CNN_BACKBONE}, feature_dim={enc_dim}")

    # --- Извлечение фичей ---
    print("\n[INFO] Извлечение CNN-фичей из Train...")
    X_train, y_train = extract_features(RULDataset(
        TRAIN_DIR, seq_length=SEQ_LENGTH), encoder, device)
    print(f"  → X_train: {X_train.shape}, y_train: {y_train.shape}")

    print("[INFO] Извлечение CNN-фичей из Val...")
    X_val, y_val = extract_features(RULDataset(
        VAL_DIR, seq_length=SEQ_LENGTH), encoder, device)
    print(f"  → X_val: {X_val.shape}, y_val: {y_val.shape}")

    print("[INFO] Извлечение CNN-фичей из Test...")
    X_test, y_test = extract_features(RULDataset(
        TEST_DIR, seq_length=SEQ_LENGTH), encoder, device)
    print(f"  → X_test: {X_test.shape}, y_test: {y_test.shape}")

    # --- CatBoost ---
    print("\n" + "=" * 60)
    print("  Training CatBoostRegressor")
    print("=" * 60)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="CatBoost_CNN_features"):
        catboost_params = {
            "iterations": 1000, "learning_rate": 0.05, "depth": 6,
            "loss_function": "RMSE", "random_seed": RANDOM_SEED,
            "verbose": 100, "early_stopping_rounds": 50,
        }
        mlflow.log_params({
            "model_type": "CatBoostRegressor", "cnn_backbone": CNN_BACKBONE,
            "seq_length": SEQ_LENGTH, "feature_dim": X_train.shape[1],
            **{f"cb_{k}": v for k, v in catboost_params.items() if k != "verbose"},
        })

        model = CatBoostRegressor(**catboost_params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=100)

        y_pred_test = model.predict(X_test)
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)

        print(f"\n[RESULT] Test MSE: {test_mse:.6f}")
        print(f"[RESULT] Test MAE: {test_mae:.4f}")

        mlflow.log_metrics({"test_mse": test_mse, "test_mae": test_mae})

        plot_path = os.path.join(FIGURES_DIR, "rul_prediction_boosting.png")
        plot_rul(y_test, y_pred_test, plot_path,
                 model_name="CatBoost", pred_color="#059669",
                 pred_label="CatBoost Predicted RUL")
        mlflow.log_artifact(plot_path, artifact_path="figures")

        cb_model_path = os.path.join(MODELS_DIR, "catboost_rul_model.cbm")
        model.save_model(cb_model_path)
        mlflow.log_artifact(cb_model_path, artifact_path="checkpoints")
        print(f"[INFO] CatBoost модель сохранена: {cb_model_path}")

        # --- Дополнительная визуализация и интерпретируемость ---
        print("\n[INFO] Генерация дополнительных графиков...")

        # 1. Residuals plot
        res_path = os.path.join(FIGURES_DIR, "residuals_boosting.png")
        plot_residuals(y_test, y_pred_test, res_path, model_name="CatBoost")
        mlflow.log_artifact(res_path, artifact_path="figures")

        # 2. Learning Curves
        eval_metrics = model.get_evals_result()
        if "learn" in eval_metrics and "RMSE" in eval_metrics["learn"]:
            lc_path = os.path.join(FIGURES_DIR, "learning_curves_boosting.png")
            plot_learning_curves(
                eval_metrics["learn"]["RMSE"],
                eval_metrics["validation"]["RMSE"],
                lc_path, metric_name="RMSE"
            )
            mlflow.log_artifact(lc_path, artifact_path="figures")

        # 3. Feature Importance
        fi_path = os.path.join(FIGURES_DIR, "feature_importance_boosting.png")
        plot_catboost_feature_importance(model, fi_path)
        mlflow.log_artifact(fi_path, artifact_path="figures")

        # 4. SHAP (если доступен)
        if _SHAP_AVAILABLE:
            print("[INFO] Расчёт SHAP значений...")
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)

                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_test,
                                  show=False, max_display=20)
                shap_path = os.path.join(
                    FIGURES_DIR, "shap_summary_boosting.png")
                plt.savefig(shap_path, dpi=300, bbox_inches="tight")
                plt.close()
                mlflow.log_artifact(shap_path, artifact_path="figures")
                print(f"[INFO] SHAP график сохранён: {shap_path}")
            except Exception as e:
                print(f"[WARNING] Ошибка при расчёте SHAP: {e}")

    print("\n" + "=" * 60)
    print("  BOOSTING PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
