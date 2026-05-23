"""
pretrain_cnn_rul.py

Этот скрипт предобучает CNN (ResNet18) предсказывать RUL по одиночным CWT-скалограммам.
Это решает проблему mode collapse: классификационный CNN не понимает степень деградации,
поэтому мы заставляем его выучить continuous RUL (Temporal Distance Regression).

После этого обучения полученный чекпоинт `best_resnet18_rul.pth` можно использовать
для заморозки и извлечения CNN-фичей в гибридных RNN-моделях.
"""

import os
import re
import sys
import time
import argparse
import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Добавляем путь для импорта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prediction.model import create_cnn_encoder
from prediction.config import (
    DATA_BASE_DIR,
    MODELS_DIR,
    CNN_BACKBONE,
    CNN_IN_CHANNELS,
    CNN_CHECKPOINT_PATH, # Это базовый классификационный вес
)
from prediction.utils import get_device

CWT_SCALES = 32

# Папка для нового чекпоинта
NEW_CNN_CHECKPOINT = os.path.join(MODELS_DIR, "cnn", "best_resnet18_rul.pth")

# Функции кэширования из train_rul_hybrid_v3_rnn.py
def _cwt_cache_path(file_path: str, window_size: int, cwt_scales: int) -> str:
    parts = file_path.split(os.sep)
    bearing_name = f"{parts[-2]}__{parts[-1].replace('.csv', '')}"
    cache_dir = os.path.join("data", "cache", "cwt_scalograms")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{bearing_name}_ws{window_size}_sc{cwt_scales}.npy")

def _atomic_npy_save(path: str, arr: np.ndarray) -> None:
    tmp_path = path + f".tmp.{os.getpid()}"
    with open(tmp_path, "wb") as f:
        np.save(f, arr)
    os.replace(tmp_path, path)

class SingleFrameRULDataset(Dataset):
    """
    Датасет для предобучения: читает одиночные скалограммы и привязывает RUL.
    В 10 раз больше примеров, чем в MultiBearingRULDataset (т.к. не берем seq_len=10).
    """
    def __init__(self, bearing_dirs: list, window_size=1024, cwt_scales=32, rul_clip=1.0, use_cache=True):
        self.window_size = window_size
        self.cwt_widths = np.arange(1, cwt_scales + 1)
        self.cwt_scales = cwt_scales
        self.rul_clip = rul_clip
        self.use_cache = use_cache
        self.samples = []
        
        for b_dir in bearing_dirs:
            files = [f for f in os.listdir(b_dir) if f.endswith(".csv")]
            files.sort(key=lambda f: int(re.sub(r"\D", "", f)))
            paths = [os.path.join(b_dir, f) for f in files]
            
            total_files = len(paths)
            if total_files == 0: continue
            
            for i, p in enumerate(paths):
                rul = min(1.0 - (i / max(total_files - 1, 1)), self.rul_clip)
                self.samples.append((p, float(rul)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, rul = self.samples[idx]
        
        if self.use_cache:
            cache_path = _cwt_cache_path(file_path, self.window_size, self.cwt_scales)
            if os.path.exists(cache_path):
                scalogram = np.load(cache_path)
                return torch.tensor(scalogram, dtype=torch.float32), torch.tensor([rul], dtype=torch.float32)

        df = pd.read_csv(file_path)
        n = len(df)
        if n < self.window_size:
            h_sig = np.pad(df.iloc[:, 0].values, (0, self.window_size - n))
            v_sig = np.pad(df.iloc[:, 1].values, (0, self.window_size - n))
        else:
            h_sig = df.iloc[:self.window_size, 0].values.astype(np.float32)
            v_sig = df.iloc[:self.window_size, 1].values.astype(np.float32)

        cwt_h, _ = pywt.cwt(h_sig, self.cwt_widths, "mexh")
        cwt_v, _ = pywt.cwt(v_sig, self.cwt_widths, "mexh")

        def _normalize(arr):
            mu, std = arr.mean(), arr.std()
            return (arr - mu) / (std + 1e-8)

        scalogram = np.stack([_normalize(cwt_h), _normalize(cwt_v)], axis=0).astype(np.float32)
        
        if self.use_cache:
            _atomic_npy_save(cache_path, scalogram)
            
        return torch.tensor(scalogram, dtype=torch.float32), torch.tensor([rul], dtype=torch.float32)


class CNNRULRegressor(nn.Module):
    def __init__(self, backbone, in_channels, checkpoint_path):
        super().__init__()
        # Загружаем базовые веса из классификации (или ImageNet), размораживаем
        self.encoder, enc_dim = create_cnn_encoder(
            backbone, in_channels, pretrained=True, freeze=False, checkpoint_path=checkpoint_path
        )
        self.regressor = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(enc_dim, 1)
        )

    def forward(self, x):
        # x: (B, 2, 32, 1024)
        feats = self.encoder(x)
        return self.regressor(feats)


def plot_pretrain_results(train_losses, val_losses, true_rul, pred_rul, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    # Learning curves
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label="Train MSE", color="#2563eb")
    ax.plot(val_losses, label="Val MSE", color="#dc2626")
    ax.set_title("CNN RUL Pre-training: Learning Curves", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pretrain_learning_curves.png"), dpi=300)
    plt.close(fig)
    
    # Prediction scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(true_rul, pred_rul, alpha=0.3, color="#2563eb")
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2)
    ax.set_title("CNN RUL Pre-training: True vs Predicted RUL", fontweight="bold")
    ax.set_xlabel("True RUL")
    ax.set_ylabel("Predicted RUL")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pretrain_rul_scatter.png"), dpi=300)
    plt.close(fig)

def plot_residuals(true_rul, pred_rul, out_dir):
    residuals = np.array(true_rul) - np.array(pred_rul)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=30, color="#059669", alpha=0.8)
    ax.set_title("CNN RUL Pre-training: Residuals Distribution", fontweight="bold")
    ax.set_xlabel("Error (True - Pred)")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pretrain_residuals.png"), dpi=300)
    plt.close(fig)

def plot_shap_explanations(model, background_batch, test_batch, out_dir, device):
    """
    Generate and plot SHAP values for a few samples.
    """
    model.eval()
    background_batch = background_batch.to(device)
    test_batch = test_batch.to(device)
    
    # Use GradientExplainer
    explainer = shap.GradientExplainer(model, background_batch)
    shap_values = explainer.shap_values(test_batch)
    
    # Depending on SHAP version, shap_values might be a list or array
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
        
    if isinstance(shap_values, torch.Tensor):
        shap_values = shap_values.cpu().detach().numpy()
        
    test_batch_np = test_batch.cpu().detach().numpy()
    
    n_samples = min(3, test_batch_np.shape[0])
    fig, axes = plt.subplots(n_samples, 2, figsize=(12, 4 * n_samples))
    
    if n_samples == 1:
        axes = np.expand_dims(axes, 0)
        
    for i in range(n_samples):
        # We plot the horizontal channel (index 0)
        orig_img = test_batch_np[i, 0, :, :]
        shap_img = shap_values[i, 0, :, :]
        
        # Plot original CWT
        ax_orig = axes[i, 0]
        im = ax_orig.imshow(orig_img, aspect='auto', cmap='jet', origin='lower')
        ax_orig.set_title(f"Sample {i+1} Original CWT (Horizontal)", fontweight="bold")
        ax_orig.set_ylabel("Scales (Frequency)")
        if i == n_samples - 1:
            ax_orig.set_xlabel("Time Step")
            
        # Plot SHAP
        ax_shap = axes[i, 1]
        vmax = np.max(np.abs(shap_img))
        im_shap = ax_shap.imshow(shap_img, aspect='auto', cmap='coolwarm', origin='lower', vmin=-vmax, vmax=vmax)
        ax_shap.set_title(f"Sample {i+1} SHAP Values", fontweight="bold")
        if i == n_samples - 1:
            ax_shap.set_xlabel("Time Step")
        fig.colorbar(im_shap, ax=ax_shap, orientation='vertical')
        
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pretrain_shap_analysis.png"), dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    device = get_device()
    print(f"[INFO] Using device: {device}")

    # Bearings split
    train_bearings = ["35Hz12kN/Bearing1_1", "35Hz12kN/Bearing1_2", "37.5Hz11kN/Bearing2_1", "37.5Hz11kN/Bearing2_2", "40Hz10kN/Bearing3_1", "40Hz10kN/Bearing3_2"]
    val_bearings = ["35Hz12kN/Bearing1_3", "37.5Hz11kN/Bearing2_3"]
    
    train_dirs = [os.path.join("data/raw/XJTU-SY", b) for b in train_bearings]
    val_dirs = [os.path.join("data/raw/XJTU-SY", b) for b in val_bearings]

    print("[INFO] Creating datasets...")
    train_ds = SingleFrameRULDataset(train_dirs, cwt_scales=CWT_SCALES, rul_clip=0.8)
    val_ds = SingleFrameRULDataset(val_dirs, cwt_scales=CWT_SCALES, rul_clip=0.8)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"[INFO] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    model = CNNRULRegressor(CNN_BACKBONE, CNN_IN_CHANNELS, CNN_CHECKPOINT_PATH).to(device)
    
    # Discriminative learning rates to prevent catastrophic forgetting of pre-trained features
    optimizer = optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': 1e-5},
        {'params': model.regressor.parameters(), 'lr': args.lr}
    ], weight_decay=1e-2)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler(device.type) if device.type == "cuda" else None

    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    print("[INFO] Starting training...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            if scaler:
                with torch.amp.autocast(device.type):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
            running_loss += loss.item() * inputs.size(0)
            
        epoch_train_loss = running_loss / len(train_ds)
        train_losses.append(epoch_train_loss)
        
        model.eval()
        running_val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.amp.autocast(device.type) if scaler else torch.autocast('cpu', enabled=False):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                running_val_loss += loss.item() * inputs.size(0)
                all_preds.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())
                
        epoch_val_loss = running_val_loss / len(val_ds)
        val_losses.append(epoch_val_loss)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train MSE: {epoch_train_loss:.4f} | Val MSE: {epoch_val_loss:.4f}")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            os.makedirs(os.path.dirname(NEW_CNN_CHECKPOINT), exist_ok=True)
            # Сохраняем только encoder (с тем же форматом, что и backbone)
            torch.save(model.encoder.state_dict(), NEW_CNN_CHECKPOINT)
            print(f"  --> Saved new best checkpoint: {NEW_CNN_CHECKPOINT}")
            best_preds, best_targets = all_preds, all_targets

    print("[INFO] Generating plots and metrics...")
    out_dir = os.path.join("reports", "figures", "pretrain_cnn_rul")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Plots
    plot_pretrain_results(train_losses, val_losses, best_targets, best_preds, out_dir)
    plot_residuals(best_targets, best_preds, out_dir)
    
    # 2. Metrics
    mse = mean_squared_error(best_targets, best_preds)
    mae = mean_absolute_error(best_targets, best_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(best_targets, best_preds)
    
    metrics_df = pd.DataFrame([{
        "MSE": mse, "MAE": mae, "RMSE": rmse, "R2": r2
    }])
    metrics_df.to_csv(os.path.join(out_dir, "pretrain_metrics.csv"), index=False)
    print(f"[INFO] Validation Metrics -> MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    
    # 3. SHAP Analysis
    print("[INFO] Running SHAP explanation...")
    try:
        # Get background batch
        bg_loader = iter(DataLoader(train_ds, batch_size=16, shuffle=True))
        bg_batch, _ = next(bg_loader)
        
        # Get test batch (e.g. from validation)
        test_loader = iter(DataLoader(val_ds, batch_size=3, shuffle=False))
        test_batch, _ = next(test_loader)
        
        plot_shap_explanations(model, bg_batch, test_batch, out_dir, device)
        print("[INFO] SHAP analysis saved.")
    except Exception as e:
        print(f"[ERROR] SHAP analysis failed: {e}")
        
    print(f"[INFO] Done! Pretrained weights saved to {NEW_CNN_CHECKPOINT}")

if __name__ == "__main__":
    main()
