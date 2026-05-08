"""
train.py — NAS + HPO пайплайн для прогнозирования RUL (XJTU-SY).

Реализует:
    - Optuna Neural Architecture Search: автоматический выбор временного блока
      (LSTM / GRU / TCN / Transformer) как гиперпараметра
    - Optuna HPO: подбор learning_rate, seq_length, hidden_size, dropout
    - MLflow трекинг: логирование метрик, гиперпараметров и артефактов
    - Финальное обучение с лучшими параметрами + Early Stopping
    - Генерация графика "True vs Predicted RUL" для диссертации

Usage:
    uv run python src/prediction/train.py
"""

import os
import sys
import copy
import warnings

import matplotlib
import mlflow
import mlflow.pytorch
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    RANDOM_SEED, DATA_BASE_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
    FIGURES_DIR, MODELS_DIR, MLFLOW_TRACKING_URI,
    N_TRIALS, EPOCHS, PATIENCE,
    CNN_BACKBONE, CNN_IN_CHANNELS, CNN_FREEZE, CNN_CHECKPOINT_PATH,
    NAS_TEMPORAL_TYPES,
)
from data_loader import RULDataset
from model import UniversalHybridRULNet, create_cnn_encoder
from utils import get_device, train_one_epoch, evaluate, plot_rul, plot_residuals, plot_learning_curves

matplotlib.use("Agg")
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

MLFLOW_EXPERIMENT: str = "XJTU_SY_RUL_NAS"
SEQ_LENGTH_CANDIDATES = [10, 20, 30, 50]
FINE_TUNE_CNN = True


def _batch_size_candidates(seq_length: int) -> list[int]:
    """Keeps CNN fine-tuning memory bounded for longer temporal histories."""
    if seq_length >= 50:
        return [1, 2]
    if seq_length >= 30:
        return [1, 2, 4]
    if seq_length >= 20:
        return [2, 4, 8]
    return [4, 8, 16]


def _is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


# ========================== MODEL BUILDER ==================================


def _build_model(
    temporal_type: str,
    hidden_size: int,
    dropout: float,
    device: torch.device,
    fine_tune: bool = FINE_TUNE_CNN,
) -> nn.Module:
    """Создаёт модель UniversalHybridRULNet с заданными параметрами."""
    encoder, enc_dim = create_cnn_encoder(
        backbone_name=CNN_BACKBONE,
        in_channels=CNN_IN_CHANNELS,
        pretrained=False,
        freeze=not fine_tune,
        checkpoint_path=CNN_CHECKPOINT_PATH,
    )
    model = UniversalHybridRULNet(
        encoder=encoder,
        encoder_dim=enc_dim,
        temporal_type=temporal_type,
        hidden_size=hidden_size,
        dropout=dropout,
        num_temporal_layers=2,
        fine_tune=fine_tune,
    )
    return model.to(device)


# ========================== OPTUNA OBJECTIVE ===============================


def objective(trial: optuna.Trial, device: torch.device) -> float:
    """Optuna objective: NAS + HPO. Один trial = один nested MLflow run.

    Подбираемые гиперпараметры:
        - temporal_type: архитектура временного блока
        - lr: learning rate
        - seq_length: длина истории (окна)
        - hidden_size: размер скрытого состояния
        - dropout: вероятность dropout
    """
    temporal_type = trial.suggest_categorical("temporal_type", NAS_TEMPORAL_TYPES)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    seq_length = trial.suggest_categorical("seq_length", SEQ_LENGTH_CANDIDATES)
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    batch_size = trial.suggest_categorical(
        "batch_size", _batch_size_candidates(seq_length))

    try:
        train_ds = RULDataset(TRAIN_DIR, seq_length=seq_length)
        val_ds = RULDataset(VAL_DIR, seq_length=seq_length)
    except ValueError as e:
        print(f"[WARNING] Skipping trial: {e}")
        raise optuna.TrialPruned()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = _build_model(
        temporal_type, hidden_size, dropout, device, fine_tune=FINE_TUNE_CNN)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
    )

    best_val_mse = float("inf")

    with mlflow.start_run(
        run_name=f"trial_{trial.number:03d}_{temporal_type}", nested=True,
    ):
        mlflow.log_params({
            "temporal_type": temporal_type, "lr": lr,
            "seq_length": seq_length, "hidden_size": hidden_size,
            "dropout": dropout, "batch_size": batch_size,
            "cnn_backbone": CNN_BACKBONE,
            "cnn_checkpoint_path": CNN_CHECKPOINT_PATH,
            "fine_tune_cnn": FINE_TUNE_CNN,
        })

        try:
            for epoch in range(EPOCHS):
                train_loss = train_one_epoch(
                    model, train_loader, criterion, optimizer, device)
                val_loss, val_mae, _, _ = evaluate(
                    model, val_loader, criterion, device)

                if val_loss < best_val_mse:
                    best_val_mse = val_loss

                mlflow.log_metrics({
                    "train_mse": train_loss, "val_mse": val_loss, "val_mae": val_mae,
                }, step=epoch)

                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        except RuntimeError as e:
            if _is_cuda_oom(e):
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                mlflow.set_tag("pruned_reason", "cuda_oom")
                raise optuna.TrialPruned()
            raise

        mlflow.log_metric("best_val_mse", best_val_mse)

    print(
        f"  Trial {trial.number:03d} | {temporal_type:12s} | "
        f"lr={lr:.2e}, seq={seq_length}, batch={batch_size}, hid={hidden_size}, "
        f"drop={dropout:.2f} → val_mse={best_val_mse:.6f}"
    )
    return best_val_mse


def log_optuna_plots(study: optuna.Study, figures_dir: str) -> None:
    """Сохранение и логирование графиков Optuna (Matplotlib)."""
    import optuna.visualization.matplotlib as vis
    import matplotlib.pyplot as plt
    try:
        # Optimization History
        vis.plot_optimization_history(study)
        hist_path = os.path.join(figures_dir, "optuna_history.png")
        plt.tight_layout()
        plt.savefig(hist_path, dpi=300)
        plt.close()
        mlflow.log_artifact(hist_path, artifact_path="figures")
        
        # Param Importances
        vis.plot_param_importances(study)
        imp_path = os.path.join(figures_dir, "optuna_importances.png")
        plt.tight_layout()
        plt.savefig(imp_path, dpi=300)
        plt.close()
        mlflow.log_artifact(imp_path, artifact_path="figures")
        
        print("[INFO] Графики Optuna сохранены и залогированы.")
    except Exception as e:
        print(f"[WARNING] Не удалось сохранить графики Optuna: {e}")


# ========================== MAIN PIPELINE ==================================


def main() -> None:
    """Phase 1: Optuna NAS+HPO → Phase 2: Final training → Phase 3: Test + Plot."""
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = get_device()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # ================================================================
    # PHASE 1: Optuna NAS + HPO
    # ================================================================
    print("\n" + "=" * 60)
    print("  PHASE 1: Neural Architecture Search + HPO (Optuna)")
    print("=" * 60)

    with mlflow.start_run(run_name="RUL_NAS_HPO_Study") as parent_run:
        mlflow.log_params({
            "n_trials": N_TRIALS, "epochs_per_trial": EPOCHS,
            "random_seed": RANDOM_SEED, "dataset": "XJTU-SY",
            "cnn_backbone": CNN_BACKBONE,
            "cnn_checkpoint_path": CNN_CHECKPOINT_PATH,
            "fine_tune_cnn": FINE_TUNE_CNN,
            "seq_length_candidates": str(SEQ_LENGTH_CANDIDATES),
            "nas_candidates": str(NAS_TEMPORAL_TYPES),
        })

        study = optuna.create_study(
            direction="minimize", study_name="xjtu_rul_nas",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2),
        )
        study.optimize(
            lambda trial: objective(trial, device),
            n_trials=N_TRIALS, show_progress_bar=True,
        )

        best = study.best_trial
        bp = best.params
        print(f"\n[RESULT] Best trial #{best.number}: val_mse={best.value:.6f}")
        print(f"[RESULT] Best params: {bp}")

        mlflow.log_params({f"best_{k}": v for k, v in bp.items()})
        mlflow.log_metric("best_val_mse", best.value)

        # Логируем графики Optuna
        log_optuna_plots(study, FIGURES_DIR)

        # ============================================================
        # PHASE 2: Final Training
        # ============================================================
        best_temporal = bp["temporal_type"]
        best_seq = bp["seq_length"]
        best_lr = bp["lr"]
        best_hidden = bp["hidden_size"]
        best_dropout = bp["dropout"]
        best_batch_size = bp["batch_size"]

        print("\n" + "=" * 60)
        print(f"  PHASE 2: Final Training — CNN+{best_temporal.upper()}")
        print("=" * 60)

        final_model = _build_model(
            best_temporal,
            best_hidden,
            best_dropout,
            device,
            fine_tune=FINE_TUNE_CNN,
        )
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, final_model.parameters()), lr=best_lr,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        train_ds = RULDataset(TRAIN_DIR, seq_length=best_seq)
        val_ds = RULDataset(VAL_DIR, seq_length=best_seq)
        test_ds = RULDataset(TEST_DIR, seq_length=best_seq)

        train_loader = DataLoader(train_ds, batch_size=best_batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=best_batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=best_batch_size, shuffle=False)

        best_val_mse = float("inf")
        epochs_no_improve = 0
        best_state = None

        with mlflow.start_run(
            run_name=f"Final_CNN+{best_temporal.upper()}", nested=True,
        ):
            mlflow.log_params({
                "temporal_type": best_temporal, "lr": best_lr,
                "seq_length": best_seq, "hidden_size": best_hidden,
                "dropout": best_dropout, "batch_size": best_batch_size,
                "epochs": EPOCHS, "patience": PATIENCE,
                "cnn_checkpoint_path": CNN_CHECKPOINT_PATH,
                "fine_tune_cnn": FINE_TUNE_CNN,
            })

            print(f"\n[INFO] Обучение финальной модели на {EPOCHS} эпох...")
            train_hist, val_hist = [], []
            for epoch in range(EPOCHS):
                train_loss = train_one_epoch(
                    final_model, train_loader, criterion, optimizer, device,
                )
                val_loss, val_mae, _, _ = evaluate(
                    final_model, val_loader, criterion, device,
                )
                scheduler.step()

                train_hist.append(train_loss)
                val_hist.append(val_loss)

                mlflow.log_metrics({
                    "final_train_mse": train_loss,
                    "final_val_mse": val_loss,
                    "final_val_mae": val_mae,
                    "lr": optimizer.param_groups[0]["lr"],
                }, step=epoch)

                improved = val_loss < best_val_mse
                if improved:
                    best_val_mse = val_loss
                    best_state = copy.deepcopy(final_model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                print(
                    f"  Эпоха [{epoch + 1:02d}/{EPOCHS}] | "
                    f"Train MSE: {train_loss:.6f} | "
                    f"Val MSE: {val_loss:.6f} | "
                    f"Val MAE: {val_mae:.4f}"
                    + (" ★" if improved else "")
                )

                if epochs_no_improve >= PATIENCE:
                    print(f"[INFO] Early stopping на эпохе {epoch + 1}")
                    break

            # После обучения сохраняем кривые обучения
            lc_path = os.path.join(FIGURES_DIR, "learning_curves_nn.png")
            plot_learning_curves(train_hist, val_hist, lc_path, metric_name="MSE")
            mlflow.log_artifact(lc_path, artifact_path="figures")

            if best_state is not None:
                final_model.load_state_dict(best_state)

            # ============================================================
            # PHASE 3: Test + Visualization
            # ============================================================
            print("\n" + "=" * 60)
            print("  PHASE 3: Test Evaluation + Visualization")
            print("=" * 60)

            test_loss, test_mae, test_preds, test_labels = evaluate(
                final_model, test_loader, criterion, device,
            )
            print(f"[RESULT] Test MSE: {test_loss:.6f} | Test MAE: {test_mae:.4f}")

            mlflow.log_metrics({"test_mse": test_loss, "test_mae": test_mae})

            plot_path = os.path.join(FIGURES_DIR, "rul_prediction.png")
            plot_rul(test_labels, test_preds, plot_path,
                     model_name=f"CNN+{best_temporal.upper()}")
            mlflow.log_artifact(plot_path, artifact_path="figures")

            # Добавляем Residuals Plot
            res_path = os.path.join(FIGURES_DIR, "residuals_nn.png")
            plot_residuals(test_labels, test_preds, res_path, 
                           model_name=f"CNN+{best_temporal.upper()}")
            mlflow.log_artifact(res_path, artifact_path="figures")

            model_path = os.path.join(MODELS_DIR, "best_rul_model.pth")
            torch.save({
                "temporal_type": best_temporal, "cnn_backbone": CNN_BACKBONE,
                "hidden_size": best_hidden, "dropout": best_dropout,
                "state_dict": final_model.state_dict(),
                "best_val_mse": best_val_mse,
                "test_mse": test_loss, "test_mae": test_mae,
                "hyperparams": bp,
            }, model_path)
            mlflow.log_artifact(model_path, artifact_path="checkpoints")
            print(f"[INFO] Модель сохранена: {model_path}")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Best architecture : CNN+{best_temporal.upper()}")
    print(f"  Best HPO val_mse  : {best.value:.6f}")
    print(f"  Final test_mse    : {test_loss:.6f}")
    print(f"  Final test_mae    : {test_mae:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
