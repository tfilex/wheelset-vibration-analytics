"""
config.py — Единая конфигурация для модуля prediction (RUL).

Все пути, гиперпараметры и параметры CNN-энкодера
определены здесь, чтобы избежать дублирования в train.py и train_boosting.py.
"""

import os
from typing import List

# ========================== PATHS ==========================================

DATA_BASE_DIR: str = "data/raw/XJTU-SY/35Hz12kN"
TRAIN_DIR: str = os.path.join(DATA_BASE_DIR, "Bearing1_1")
VAL_DIR: str = os.path.join(DATA_BASE_DIR, "Bearing1_2")
TEST_DIR: str = os.path.join(DATA_BASE_DIR, "Bearing1_3")

FIGURES_DIR: str = "reports/figures/summary"
MODELS_DIR: str = "models"

# src/prediction/config.py → src/prediction → src → VKR (project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
MLFLOW_TRACKING_URI: str = f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}"

# ========================== TRAINING =======================================

RANDOM_SEED: int = 42
N_TRIALS: int = 50
EPOCHS: int = 50
PATIENCE: int = 4

# ========================== CNN ENCODER ====================================

CNN_BACKBONE: str = "resnet18"
CNN_IN_CHANNELS: int = 2  # Горизонтальный + вертикальный вибросигналы
CNN_FREEZE: bool = True
_rul_ckpt = os.path.join(MODELS_DIR, "cnn", "best_resnet18_rul.pth")
_cls_ckpt = os.path.join(MODELS_DIR, "cnn", "best_resnet18.pth")
CNN_CHECKPOINT_PATH: str = _rul_ckpt if os.path.exists(_rul_ckpt) else _cls_ckpt
# ========================== NAS ============================================

NAS_TEMPORAL_TYPES: List[str] = ["lstm", "gru", "transformer"]
