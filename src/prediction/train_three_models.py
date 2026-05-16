"""
train_three_models.py — обучение и сохранение трех temporal-вариантов RUL модели.

Обучает модели с temporal-блоками:
    - lstm
    - gru
    - transformer

Сценарий:
    1) Загружает train/val/test через RULDataset
    2) Обучает каждую модель с early stopping
    3) Оценивает на test
    4) Сохраняет отдельные чекпоинты в models/
"""

import copy
import os
import sys
from typing import Dict, List

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (  # noqa: E402
    CNN_BACKBONE,
    CNN_CHECKPOINT_PATH,
    CNN_FREEZE,
    CNN_IN_CHANNELS,
    EPOCHS,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    PATIENCE,
    RANDOM_SEED,
    TEST_DIR,
    TRAIN_DIR,
    VAL_DIR,
)
from data_loader import RULDataset  # noqa: E402
from model import UniversalHybridRULNet, create_cnn_encoder  # noqa: E402
from utils import evaluate, get_device, train_one_epoch  # noqa: E402

TEMPORAL_TYPES: List[str] = ["lstm", "gru", "transformer"]
MLFLOW_EXPERIMENT = "XJTU_SY_RUL_ThreeModels"


def build_model(
    temporal_type: str,
    hidden_size: int,
    dropout: float,
    device: torch.device,
) -> nn.Module:
    encoder, enc_dim = create_cnn_encoder(
        backbone_name=CNN_BACKBONE,
        in_channels=CNN_IN_CHANNELS,
        pretrained=False,
        freeze=False,
        checkpoint_path=CNN_CHECKPOINT_PATH,
    )
    model = UniversalHybridRULNet(
        encoder=encoder,
        encoder_dim=enc_dim,
        temporal_type=temporal_type,
        hidden_size=hidden_size,
        dropout=dropout,
        num_temporal_layers=2,
        fine_tune=True,
    )
    return model.to(device)


def train_single_model(
    temporal_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    lr: float,
    hidden_size: int,
    dropout: float,
) -> Dict[str, float]:
    model = build_model(temporal_type, hidden_size, dropout, device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_mse = float("inf")
    epochs_no_improve = 0
    best_state = None

    for epoch in range(EPOCHS):
        train_mse = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_mse, val_mae, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        mlflow.log_metrics(
            {
                "train_mse": train_mse,
                "val_mse": val_mse,
                "val_mae": val_mae,
                "lr": optimizer.param_groups[0]["lr"],
            },
            step=epoch,
        )

        improved = val_mse < best_val_mse
        if improved:
            best_val_mse = val_mse
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(
            f"  [{temporal_type}] Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"train_mse={train_mse:.6f} | val_mse={val_mse:.6f} | "
            f"val_mae={val_mae:.4f}"
            + (" ★" if improved else "")
        )

        if epochs_no_improve >= PATIENCE:
            print(f"  [{temporal_type}] Early stopping on epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_mse, test_mae, _, _ = evaluate(model, test_loader, criterion, device)
    print(f"  [{temporal_type}] Test MSE: {test_mse:.6f} | Test MAE: {test_mae:.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    ckpt_path = os.path.join(MODELS_DIR, f"best_rul_{temporal_type}.pth")
    torch.save(
        {
            "temporal_type": temporal_type,
            "cnn_backbone": CNN_BACKBONE,
            "hidden_size": hidden_size,
            "dropout": dropout,
            "lr": lr,
            "epochs": EPOCHS,
            "state_dict": model.state_dict(),
            "best_val_mse": best_val_mse,
            "test_mse": test_mse,
            "test_mae": test_mae,
        },
        ckpt_path,
    )
    mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")

    return {
        "best_val_mse": best_val_mse,
        "test_mse": test_mse,
        "test_mae": test_mae,
    }


def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = get_device()

    seq_length = 10
    batch_size = 16
    lr = 1e-3
    hidden_size = 64
    dropout = 0.2

    train_ds = RULDataset(TRAIN_DIR, seq_length=seq_length)
    val_ds = RULDataset(VAL_DIR, seq_length=seq_length)
    test_ds = RULDataset(TEST_DIR, seq_length=seq_length)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=2)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    print("\n" + "=" * 68)
    print("  Training three temporal models: lstm, gru, transformer")
    print("=" * 68)

    summary: Dict[str, Dict[str, float]] = {}
    for temporal_type in TEMPORAL_TYPES:
        with mlflow.start_run(run_name=f"Final_CNN+{temporal_type.upper()}"):
            mlflow.log_params(
                {
                    "temporal_type": temporal_type,
                    "cnn_backbone": CNN_BACKBONE,
                    "seq_length": seq_length,
                    "batch_size": batch_size,
                    "lr": lr,
                    "hidden_size": hidden_size,
                    "dropout": dropout,
                    "epochs": EPOCHS,
                    "patience": PATIENCE,
                }
            )
            summary[temporal_type] = train_single_model(
                temporal_type=temporal_type,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                lr=lr,
                hidden_size=hidden_size,
                dropout=dropout,
            )

    print("\n" + "=" * 68)
    print("  TRAINING COMPLETE")
    print("=" * 68)
    for temporal_type, metrics in summary.items():
        print(
            f"  {temporal_type:11s} | "
            f"best_val_mse={metrics['best_val_mse']:.6f} | "
            f"test_mse={metrics['test_mse']:.6f} | "
            f"test_mae={metrics['test_mae']:.4f}"
        )


if __name__ == "__main__":
    main()
