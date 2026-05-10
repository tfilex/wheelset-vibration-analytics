"""
train_three_models_3.py — расширенный pipeline для трех temporal-моделей RUL.

Что нового относительно train_three_models.py:
1) Больше данных:
   - объединение нескольких подшипников из нескольких режимов XJTU-SY;
   - скользящее окно по CSV (stride), чтобы увеличить число последовательностей.
2) Optuna HPO:
   - подбор гиперпараметров отдельно для каждой temporal-архитектуры:
     lstm, gru, transformer.
3) Финальное обучение и сохранение:
   - обучает и сохраняет все 3 модели с лучшими параметрами.

Запуск (полный):
    uv run python src/prediction/train_three_models_3.py --profile full

Запуск (быстрый sanity-check перед ночным прогоном):
    uv run python src/prediction/train_three_models_3.py --profile fast

Запуск (рекомендуемый дневной режим):
    uv run python src/prediction/train_three_models_3.py --profile balanced

Запуск с именем MLflow-эксперимента:
    uv run python src/prediction/train_three_models_3.py --profile balanced --experiment-name "my_experiment"

Что нового относительно v2:
1) Piecewise RUL target через rul_clip.
2) AsymmetricHuberLoss с HPO по delta/alpha.
3) Monotonicity penalty для штрафа за рост RUL-предсказаний.
4) AdamW + weight decay вместо Adam.
5) ReduceLROnPlateau scheduler в финальном обучении.
6) EMA-сглаживание предсказаний и отдельные smoothed-графики.
7) Дополнительные метрики: R², PHM score, RMSE.
8) Расширенное HPO-пространство: lr, hidden_size, dropout, batch_size, num_layers.
9) Двухфазный подход: HPO с замороженным CNN, final fit с размороженным CNN.
10) Discriminative learning rate: CNN encoder обучается с lr × 0.1.
11) Warmup fine-tuning: первые 3 эпохи обучается только temporal-голова.

Режим --profile fast:
    - только одно значение window_size (1024)
    - только один temporal тип (lstm)
    - N_TRIALS = 3, EPOCHS = 3, PATIENCE = 2
    - seq_stride = 5 (меньше сэмплов), cwt_scales = 8 (быстрее CWT)
    - чекпоинты сохраняются с суффиксом _fast, не перезаписывают продовые
    - HPO использует кэш замороженных CNN-фичей, final fit дообучает CNN
"""

from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, r2_score
import torch.optim as optim
import torch.nn as nn
import torch
import pywt
import pandas as pd
import optuna
import numpy as np
import mlflow
import matplotlib.pyplot as plt
import argparse
import copy
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")

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
)
from model import TemporalOnlyRULNet, UniversalHybridRULNet, create_cnn_encoder  # noqa: E402
from utils import get_device  # noqa: E402


DATASET_ROOT = "data/raw/XJTU-SY"
MLFLOW_EXPERIMENT = "XJTU_SY_RUL_ThreeModels_v3"
TEMPORAL_TYPES = ["lstm", "gru", "transformer"]
N_TRIALS = 30
# FIX: Используем os.cpu_count() вместо жёстко прописанного 2,
# чтобы избежать deadlock на Windows и использовать ресурсы правильно.
NUM_WORKERS = min(4, (os.cpu_count() or 1) // 2)
PREDS3_MODELS_DIR = os.path.join(MODELS_DIR, "preds_3")
WINDOW_SIZE_CANDIDATES = [1024, 2048]
# Путь для дискового кэша CWT-скалограмм
CWT_CACHE_DIR = "data/cache/cwt_scalograms"
CNN_FEATURE_CACHE_DIR = "data/cache/cnn_features"
FIGURES_BASE_DIR = "reports/figures/summary/train_three_models_3"

# --fast режим: минимальные параметры для быстрого sanity-check.
# Не изменяет архитектуру — проверяет именно pipeline end-to-end.
FAST_N_TRIALS = 3
FAST_EPOCHS = 3
FAST_PATIENCE = 2
FAST_WINDOW_SIZES = [1024]          # одно значение вместо двух
FAST_TEMPORAL_TYPES = ["lstm"]      # один тип вместо трёх
FAST_SEQ_STRIDE = 5                 # меньше сэмплов → быстрее эпоха
FAST_CWT_SCALES = 8                 # меньше масштабов → быстрее CWT
INFERENCE_BENCHMARK_WARMUP_BATCHES = 3
INFERENCE_BENCHMARK_MAX_BATCHES = 50
EMA_ALPHA = 0.3
FINETUNE_WARMUP_EPOCHS = 3


@dataclass
class DatasetConfig:
    """Конфигурация расширенного датасета."""

    modes: Sequence[str]
    train_bearings: Sequence[str]
    val_bearings: Sequence[str]
    test_bearings: Sequence[str]
    seq_length: int
    window_size: int
    seq_stride: int
    val_test_stride: int   # FIX: отдельный stride для val/test (должен быть 1)
    cwt_scales: int
    rul_clip: float = 1.0   # 1.0 = без обрезки; 0.8 отсекает начальную «здоровую» фазу


DEFAULT_DATASET_CONFIG = DatasetConfig(
    modes=("35Hz12kN", "37.5Hz11kN", "40Hz10kN"),
    train_bearings=("1", "2", "4"),
    val_bearings=("5",),
    test_bearings=("3",),
    seq_length=10,
    window_size=1024,
    seq_stride=2,
    val_test_stride=1,  # FIX: для val/test stride=1, чтобы не пропускать данные
    cwt_scales=32,
    rul_clip=0.8,
)

PROFILE_DEFAULTS: Dict[str, Dict[str, object]] = {
    "fast": {
        "window_sizes": FAST_WINDOW_SIZES,
        "temporal_types": FAST_TEMPORAL_TYPES,
        "n_trials": FAST_N_TRIALS,
        "epochs": FAST_EPOCHS,
        "patience": FAST_PATIENCE,
        "seq_stride": FAST_SEQ_STRIDE,
        "val_test_stride": 1,
        "cwt_scales": FAST_CWT_SCALES,
        "rul_clip": 0.8,
        "ckpt_suffix": "fast",
    },
    "balanced": {
        "window_sizes": [1024],
        "temporal_types": ["lstm", "gru"],
        "n_trials": 10,
        "epochs": 15,
        "patience": PATIENCE,
        "seq_stride": 5,
        "val_test_stride": 2,
        "cwt_scales": DEFAULT_DATASET_CONFIG.cwt_scales,
        "rul_clip": 0.8,
        "ckpt_suffix": "balanced",
    },
    "full": {
        "window_sizes": WINDOW_SIZE_CANDIDATES,
        "temporal_types": TEMPORAL_TYPES,
        "n_trials": N_TRIALS,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "seq_stride": DEFAULT_DATASET_CONFIG.seq_stride,
        "val_test_stride": DEFAULT_DATASET_CONFIG.val_test_stride,
        "cwt_scales": DEFAULT_DATASET_CONFIG.cwt_scales,
        "rul_clip": 0.8,
        "ckpt_suffix": "",
    },
}


def _worker_init_fn(worker_id: int) -> None:
    """
    FIX: Инициализация seed в каждом worker-процессе DataLoader.
    Без этого при num_workers > 0 воспроизводимость нарушается.
    """
    seed = RANDOM_SEED + worker_id
    np.random.seed(seed)
    torch.manual_seed(seed)


def _cwt_cache_path(file_path: str, window_size: int, cwt_scales: int) -> str:
    """Возвращает путь к кэшированной скалограмме для данного файла."""
    rel = os.path.relpath(file_path, DATASET_ROOT)
    safe = rel.replace(os.sep, "__").replace(".csv", "")
    return os.path.join(CWT_CACHE_DIR, f"{safe}_ws{window_size}_sc{cwt_scales}.npy")


def _cnn_feature_cache_path(file_path: str, window_size: int, cwt_scales: int) -> str:
    """Возвращает путь к кэшированным CNN-фичам для данного CSV."""
    rel = os.path.relpath(file_path, DATASET_ROOT)
    safe = rel.replace(os.sep, "__").replace(".csv", "")
    backbone = CNN_BACKBONE.replace("/", "_")
    ckpt_name = os.path.splitext(os.path.basename(CNN_CHECKPOINT_PATH))[0]
    ckpt_tag = ckpt_name if os.path.exists(CNN_CHECKPOINT_PATH) else "no_ckpt"
    return os.path.join(
        CNN_FEATURE_CACHE_DIR,
        f"{safe}_{backbone}_{ckpt_tag}_ch{CNN_IN_CHANNELS}_ws{
            window_size}_sc{cwt_scales}.npy",
    )


def _atomic_npy_save(cache_path: str, arr: "np.ndarray") -> None:
    """
    Атомарно сохраняет массив numpy в cache_path.

    Алгоритм: записываем во временный файл в той же директории,
    затем переименовываем через os.replace() — операция атомарна
    на POSIX и Windows (NTFS). Благодаря этому конкурирующие
    DataLoader-workers не могут прочитать частично записанный файл:
    они либо видят старый файл, либо уже готовый новый.
    """
    cache_dir = os.path.dirname(cache_path)
    os.makedirs(cache_dir, exist_ok=True)
    # Создаём tmp-файл в той же директории — os.replace требует одной ФС.
    fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix=".npy.tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, arr)
        # Атомарная замена: если файл уже создан другим worker-ом — не страшно,
        # оба запишут идентичные данные и os.replace просто перезапишет.
        os.replace(tmp_path, cache_path)
    except Exception:
        # Удаляем мусорный tmp-файл при любой ошибке.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class MultiBearingRULDataset(Dataset):
    """
    Расширенный RUL-датасет: объединяет несколько bearing-директорий.

    Каждый элемент:
        X: (seq_len, 2, num_scales, window_size)  — нормализованная скалограмма
        y: (1,) нормализованный RUL [0..1]

    Исправления:
    - RUL метится по start-индексу окна (а не по end), что соответствует
      «остаточному ресурсу в начале наблюдаемой последовательности».
    - CWT-скалограммы нормализуются per-sample (z-score) перед стекингом.
    - Скалограммы кэшируются на диск, чтобы не пересчитывать при каждой эпохе.
    """

    def __init__(
        self,
        bearing_dirs: Sequence[str],
        seq_length: int = 10,
        window_size: int = 1024,
        seq_stride: int = 1,
        cwt_scales: int = 32,
        rul_clip: float = 1.0,
        use_cache: bool = True,
    ):
        self.bearing_dirs = list(bearing_dirs)
        self.seq_length = seq_length
        self.window_size = window_size
        self.seq_stride = seq_stride
        self.cwt_widths = np.arange(1, cwt_scales + 1)
        self.cwt_scales = cwt_scales
        self.rul_clip = rul_clip
        self.use_cache = use_cache
        self.samples: List[Tuple[List[str], float]] = []
        self._build_index()

    def _build_index(self) -> None:
        for bearing_dir in self.bearing_dirs:
            files = [f for f in os.listdir(bearing_dir) if f.endswith(".csv")]
            files.sort(key=lambda f: int(re.sub(r"\D", "", f)))
            paths = [os.path.join(bearing_dir, f) for f in files]

            total_files = len(paths)
            if total_files < self.seq_length:
                continue

            total_steps = max(total_files - 1, 1)
            last_start = total_files - self.seq_length
            for start in range(0, last_start + 1, self.seq_stride):
                # FIX: RUL метится по start-индексу, а не по end_idx.
                # Это соответствует «сколько ресурса осталось на момент
                # начала наблюдаемого окна».
                rul = min(1.0 - (start / total_steps), self.rul_clip)
                seq_paths = paths[start: start + self.seq_length]
                self.samples.append((seq_paths, float(rul)))

        if not self.samples:
            raise ValueError(
                "Пустой датасет: не удалось сформировать ни одной последовательности."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _process_file(self, file_path: str) -> np.ndarray:
        """
        Читает CSV, вычисляет (или загружает из кэша) CWT-скалограмму.
        FIX: результат нормализован z-score per-channel, чтобы сгладить
        разброс амплитуд между подшипниками и режимами.
        """
        # FIX: попытка загрузить из кэша, чтобы не пересчитывать CWT каждую эпоху.
        if self.use_cache:
            cache_path = _cwt_cache_path(
                file_path, self.window_size, self.cwt_scales)
            if os.path.exists(cache_path):
                return np.load(cache_path)

        df = pd.read_csv(file_path)
        n = len(df)
        if n < self.window_size:
            h_sig = np.pad(df.iloc[:, 0].values, (0, self.window_size - n))
            v_sig = np.pad(df.iloc[:, 1].values, (0, self.window_size - n))
        else:
            h_sig = df.iloc[: self.window_size, 0].values.astype(np.float32)
            v_sig = df.iloc[: self.window_size, 1].values.astype(np.float32)

        cwt_h, _ = pywt.cwt(h_sig, self.cwt_widths, "mexh")
        cwt_v, _ = pywt.cwt(v_sig, self.cwt_widths, "mexh")

        # FIX: нормализация z-score per-channel, чтобы убрать разброс амплитуд.
        def _normalize(arr: np.ndarray) -> np.ndarray:
            mu = arr.mean()
            std = arr.std()
            return (arr - mu) / (std + 1e-8)

        scalogram = np.stack([_normalize(cwt_h), _normalize(cwt_v)], axis=0)
        scalogram = scalogram.astype(np.float32)

        if self.use_cache:
            _atomic_npy_save(cache_path, scalogram)

        return scalogram

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_paths, rul = self.samples[idx]
        scalograms = [self._process_file(path) for path in seq_paths]
        x = torch.tensor(np.stack(scalograms, axis=0), dtype=torch.float32)
        y = torch.tensor([rul], dtype=torch.float32)
        return x, y


class FeatureBearingRULDataset(MultiBearingRULDataset):
    """RUL-датасет, который читает заранее посчитанные CNN-фичи."""

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_paths, rul = self.samples[idx]
        features = []
        for path in seq_paths:
            cache_path = _cnn_feature_cache_path(
                path, self.window_size, self.cwt_scales)
            if not os.path.exists(cache_path):
                raise FileNotFoundError(
                    "CNN feature cache is missing. Run with --feature-cache "
                    f"precompute enabled first: {cache_path}"
                )
            features.append(np.load(cache_path))

        x = torch.tensor(np.stack(features, axis=0), dtype=torch.float32)
        y = torch.tensor([rul], dtype=torch.float32)
        return x, y


def discover_bearing_dirs(
    dataset_root: str,
    modes: Sequence[str],
    bearing_suffixes: Sequence[str],
) -> List[str]:
    suffix_set = set(bearing_suffixes)
    dirs: List[str] = []
    for mode in modes:
        mode_dir = os.path.join(dataset_root, mode)
        if not os.path.isdir(mode_dir):
            continue
        for name in sorted(os.listdir(mode_dir)):
            path = os.path.join(mode_dir, name)
            if not os.path.isdir(path):
                continue
            suffix = name.split("_")[-1]
            if suffix in suffix_set:
                dirs.append(path)
    return dirs


def build_model(
    temporal_type: str,
    hidden_size: int,
    dropout: float,
    device: torch.device,
    fine_tune: bool = True,
    num_layers: int = 2,
) -> nn.Module:
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
        num_temporal_layers=num_layers,
        fine_tune=fine_tune,
    )
    return model.to(device)


def build_temporal_model(
    temporal_type: str,
    hidden_size: int,
    dropout: float,
    encoder_dim: int,
    device: torch.device,
    num_layers: int = 2,
) -> nn.Module:
    model = TemporalOnlyRULNet(
        encoder_dim=encoder_dim,
        temporal_type=temporal_type,
        hidden_size=hidden_size,
        dropout=dropout,
        num_temporal_layers=num_layers,
    )
    return model.to(device)


def _make_scaler(device: torch.device) -> Optional[torch.amp.GradScaler]:
    """
    FIX: GradScaler создаётся только при реальном использовании CUDA AMP.
    На CPU scaler не нужен и device="cuda" вызвал бы ошибку.
    """
    if device.type == "cuda":
        return torch.amp.GradScaler(device="cuda", enabled=True)
    return None


def build_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    loader_kwargs: Dict = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": NUM_WORKERS,
        "pin_memory": device.type == "cuda",
        # FIX: передаём worker_init_fn для воспроизводимости при num_workers > 0.
        "worker_init_fn": _worker_init_fn if NUM_WORKERS > 0 else None,
    }
    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **loader_kwargs)


def _unique_sample_paths(*datasets: MultiBearingRULDataset) -> List[str]:
    paths = {
        path
        for dataset in datasets
        for seq_paths, _ in dataset.samples
        for path in seq_paths
    }
    return sorted(paths)


def precompute_cnn_feature_cache(
    train_ds: MultiBearingRULDataset,
    val_ds: MultiBearingRULDataset,
    test_ds: MultiBearingRULDataset,
    device: torch.device,
    *,
    batch_size: int = 64,
) -> int:
    """One-time CNN feature extraction for frozen encoders."""
    if not CNN_FREEZE:
        raise ValueError(
            "--feature-cache requires CNN_FREEZE=True because cached features "
            "would become stale when the encoder is trainable."
        )

    encoder, encoder_dim = create_cnn_encoder(
        backbone_name=CNN_BACKBONE,
        in_channels=CNN_IN_CHANNELS,
        pretrained=False,
        freeze=True,
        checkpoint_path=CNN_CHECKPOINT_PATH,
    )
    encoder = encoder.to(device)
    encoder.eval()

    all_paths = _unique_sample_paths(train_ds, val_ds, test_ds)
    pending_paths = [
        path
        for path in all_paths
        if not os.path.exists(
            _cnn_feature_cache_path(
                path, train_ds.window_size, train_ds.cwt_scales)
        )
    ]
    print(
        "[INFO] CNN feature cache: "
        f"{len(all_paths) - len(pending_paths)
           } hit, {len(pending_paths)} missing"
    )

    if not pending_paths:
        return encoder_dim

    use_amp = device.type == "cuda"
    processor_ds = train_ds
    os.makedirs(CNN_FEATURE_CACHE_DIR, exist_ok=True)

    with torch.no_grad():
        for start in range(0, len(pending_paths), batch_size):
            batch_paths = pending_paths[start: start + batch_size]
            scalograms = [
                processor_ds._process_file(path) for path in batch_paths
            ]
            inputs = torch.tensor(
                np.stack(scalograms, axis=0), dtype=torch.float32
            ).to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                features = encoder(inputs)
            features_np = features.detach().cpu().numpy().astype(np.float32)

            for path, feature in zip(batch_paths, features_np):
                cache_path = _cnn_feature_cache_path(
                    path, train_ds.window_size, train_ds.cwt_scales)
                _atomic_npy_save(cache_path, feature)

            done = min(start + len(batch_paths), len(pending_paths))
            print(f"[INFO] CNN feature cache warmup: {
                  done}/{len(pending_paths)}")

    return encoder_dim


class AsymmetricHuberLoss(nn.Module):
    """
    Huber loss с асимметричным весом:
    - alpha > 1: штрафует за переоценку RUL (residual < 0, т.е. pred > true)
      сильнее, чем за недооценку — важно для предупреждения отказа.
    - delta: порог Huber, ниже которого ошибка квадратична.
    """

    def __init__(self, delta: float = 0.1, alpha: float = 1.5):
        super().__init__()
        self.delta = delta
        self.alpha = alpha

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        residual = target - input
        abs_res = residual.abs()
        huber = torch.where(
            abs_res <= self.delta,
            0.5 * residual ** 2,
            self.delta * (abs_res - 0.5 * self.delta),
        )
        weight = torch.where(
            residual < 0,
            torch.full_like(residual, self.alpha),
            torch.ones_like(residual),
        )
        return (weight * huber).mean()


def monotonicity_penalty(preds: torch.Tensor) -> torch.Tensor:
    """
    Штрафует за возрастание предсказаний внутри батча.
    RUL должен монотонно убывать по времени.
    Применяется только если батч > 1.
    """
    if preds.shape[0] < 2:
        return torch.tensor(0.0, device=preds.device)
    diff = preds[1:] - preds[:-1]
    return torch.relu(diff).mean()


def phm_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Стандартная PHM-метрика для RUL: экспоненциально асимметричный штраф.
    Переоценка (pred > true) штрафуется сильнее недооценки.
    Нормализованные значения [0..1], результат — среднее по выборке.
    """
    diff = y_pred - y_true
    scores = np.where(
        diff < 0,
        np.exp(-diff / 0.13) - 1,
        np.exp(diff / 0.10) - 1,
    )
    return float(np.mean(scores))


def _regression_metrics(labels: List[float], preds: List[float]) -> Dict[str, float]:
    y_true = np.asarray(labels, dtype=np.float64)
    y_pred = np.asarray(preds, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    if len(y_true) >= 2:
        r2 = float(r2_score(y_true, y_pred))
    else:
        r2 = float("nan")
    return {
        "r2": r2,
        "rmse": rmse,
        "phm_score": phm_score(y_true, y_pred),
    }


def ema_smooth(values: list, alpha: float = 0.3) -> list:
    """Exponential Moving Average сглаживание для временного ряда предсказаний."""
    if not values:
        return values
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
    return smoothed


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler],
    mono_weight: float = 0.0,
    freeze_encoder_module: bool = False,
) -> float:
    model.train()
    if freeze_encoder_module and hasattr(model, "encoder"):
        # Warmup freezes weights and BatchNorm statistics in the CNN encoder.
        model.encoder.eval()
    use_amp = scaler is not None
    running_loss = 0.0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if mono_weight > 0.0:
                loss = loss + mono_weight * \
                    monotonicity_penalty(outputs.squeeze())

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        total += labels.size(0)
    return running_loss / max(total, 1)


def set_encoder_trainable(model: nn.Module, trainable: bool) -> None:
    """Enables/disables gradients for the CNN encoder part of hybrid models."""
    if not hasattr(model, "encoder"):
        return
    for name, param in model.named_parameters():
        if "encoder" in name:
            param.requires_grad = trainable


def build_finetune_optimizer(
    model: nn.Module,
    *,
    base_lr: float,
    weight_decay: float,
) -> optim.Optimizer:
    """
    AdamW with discriminative LR:
    - CNN encoder: base_lr * 0.1
    - temporal head/regression layers: base_lr
    """
    encoder_params = [
        p for name, p in model.named_parameters()
        if "encoder" in name and p.requires_grad
    ]
    head_params = [
        p for name, p in model.named_parameters()
        if "encoder" not in name and p.requires_grad
    ]
    param_groups = []
    if encoder_params:
        param_groups.append(
            {"params": encoder_params, "lr": base_lr * 0.1, "name": "encoder"})
    if head_params:
        param_groups.append(
            {"params": head_params, "lr": base_lr, "name": "head"})
    if not param_groups:
        raise ValueError("No trainable parameters found for fine-tuning")
    return optim.AdamW(param_groups, weight_decay=weight_decay)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float, float]:
    model.eval()
    use_amp = device.type == "cuda"
    running_loss = 0.0
    total = 0
    preds: List[float] = []
    labels_list: List[float] = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            total += labels.size(0)
            preds.extend(outputs.cpu().numpy().flatten().tolist())
            labels_list.extend(labels.cpu().numpy().flatten().tolist())
    mae = mean_absolute_error(labels_list, preds)
    avg_loss = running_loss / max(total, 1)
    metrics = _regression_metrics(labels_list, preds)
    return avg_loss, mae, metrics["r2"], metrics["rmse"], metrics["phm_score"]


def evaluate_with_predictions(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float, float, List[float], List[float]]:
    """Оценка модели с возвратом предсказаний и истинных значений."""
    model.eval()
    use_amp = device.type == "cuda"
    running_loss = 0.0
    total = 0
    preds: List[float] = []
    labels_list: List[float] = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            total += labels.size(0)
            preds.extend(outputs.cpu().numpy().flatten().tolist())
            labels_list.extend(labels.cpu().numpy().flatten().tolist())
    mae = mean_absolute_error(labels_list, preds)
    avg_loss = running_loss / max(total, 1)
    metrics = _regression_metrics(labels_list, preds)
    return (
        avg_loss,
        mae,
        metrics["r2"],
        metrics["rmse"],
        metrics["phm_score"],
        preds,
        labels_list,
    )


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_inference_speed(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    warmup_batches: int = INFERENCE_BENCHMARK_WARMUP_BATCHES,
    max_batches: int = INFERENCE_BENCHMARK_MAX_BATCHES,
) -> Dict[str, float]:
    """Measures model forward speed on batches from a DataLoader."""
    model.eval()
    use_amp = device.type == "cuda"
    n_loader_batches = len(loader)
    warmup_limit = min(warmup_batches, max(0, n_loader_batches - 1))
    total_time_sec = 0.0
    total_samples = 0
    timed_batches = 0

    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(loader):
            inputs = inputs.to(device)

            if batch_idx < warmup_limit:
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    _ = model(inputs)
                continue

            if timed_batches >= max_batches:
                break

            _sync_device(device)
            start = time.perf_counter()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                _ = model(inputs)
            _sync_device(device)

            total_time_sec += time.perf_counter() - start
            total_samples += inputs.size(0)
            timed_batches += 1

    if total_samples == 0 or total_time_sec <= 0:
        return {
            "inference_batches": 0.0,
            "inference_samples": 0.0,
            "inference_total_time_sec": 0.0,
            "inference_ms_per_sample": 0.0,
            "inference_ms_per_batch": 0.0,
            "inference_samples_per_sec": 0.0,
        }

    return {
        "inference_batches": float(timed_batches),
        "inference_samples": float(total_samples),
        "inference_total_time_sec": total_time_sec,
        "inference_ms_per_sample": (total_time_sec / total_samples) * 1000.0,
        "inference_ms_per_batch": (total_time_sec / timed_batches) * 1000.0,
        "inference_samples_per_sec": total_samples / total_time_sec,
    }


def plot_learning_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train MSE",
            color="#2563eb", linewidth=2)
    ax.plot(epochs, val_losses, label="Val MSE", color="#dc2626", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_rul_prediction(
    true_rul: List[float],
    pred_rul: List[float],
    save_path: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(true_rul))
    ax.plot(x, true_rul, label="True RUL", color="#2563eb", linewidth=2.5)
    ax.plot(x, pred_rul, label="Predicted RUL",
            color="#dc2626", linestyle="--", linewidth=2)
    ax.fill_between(x, true_rul, pred_rul, alpha=0.1, color="#dc2626")
    ax.set_xlabel("Sequence Index")
    ax.set_ylabel("Normalized RUL")
    ax.set_ylim([-0.05, 1.05])
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(
    true_rul: List[float],
    pred_rul: List[float],
    save_path: str,
    title: str,
) -> None:
    true_arr = np.array(true_rul)
    pred_arr = np.array(pred_rul)
    residuals = true_arr - pred_arr

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.scatter(true_arr, pred_arr, alpha=0.5, color="#2563eb")
    ax1.plot([0, 1], [0, 1], "--", color="#dc2626", linewidth=2)
    ax1.set_xlabel("True RUL")
    ax1.set_ylabel("Predicted RUL")
    ax1.set_title(f"{title}: True vs Pred")
    ax1.grid(True, alpha=0.3)

    ax2.hist(residuals, bins=30, color="#059669", alpha=0.8)
    ax2.set_xlabel("Residual (True - Pred)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"{title}: Residual Distribution")
    ax2.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_optuna_dashboard(study: optuna.Study, out_dir: str, prefix: str) -> None:
    """Сохраняет основные графики Optuna в out_dir."""
    import optuna.visualization.matplotlib as vis

    os.makedirs(out_dir, exist_ok=True)
    plots = [
        ("history", vis.plot_optimization_history),
        ("importance", vis.plot_param_importances),
    ]
    for name, plot_fn in plots:
        try:
            plot_fn(study)
            plt.tight_layout()
            path = os.path.join(out_dir, f"{prefix}_{name}.png")
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()
        except Exception:
            plt.close()


def save_summary_dashboard(
    summary_rows: List[Dict[str, object]],
    out_dir: str,
) -> None:
    """Сводная таблица и bar chart по test_mse/test_mae."""
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(out_dir, "summary_metrics.csv")
    df.to_csv(csv_path, index=False)

    if df.empty:
        return

    # Barplot по test_mse
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [f"ws{r['window_size']}_{
        r['temporal_type']}" for _, r in df.iterrows()]
    values = df["test_mse"].values
    ax.bar(labels, values, color="#2563eb")
    ax.set_ylabel("Test MSE")
    ax.set_title("Test MSE comparison across experiments", fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "summary_test_mse_bar.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def objective(
    trial: optuna.Trial,
    temporal_type: str,
    train_ds: Dataset,
    val_ds: Dataset,
    device: torch.device,
    *,
    epochs: int = EPOCHS,
    use_feature_cache: bool = True,
    encoder_dim: Optional[int] = None,
) -> float:
    lr = trial.suggest_float("lr", 5e-5, 1e-2, log=True)
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.05, 0.4)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    num_layers = trial.suggest_categorical("num_layers", [1, 2, 3])
    loss_delta = trial.suggest_float("loss_delta", 0.05, 0.3)
    loss_alpha = trial.suggest_float("loss_alpha", 1.0, 2.5)
    mono_weight = trial.suggest_float("mono_weight", 0.0, 0.2)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

    if encoder_dim is None:
        raise ValueError("encoder_dim is required for frozen-CNN HPO")
    # HPO всегда идёт по feature-cache пути: CNN заморожен, Optuna быстро
    # подбирает temporal/head гиперпараметры, а fine-tuning происходит позже.
    model = build_temporal_model(
        temporal_type, hidden_size, dropout, encoder_dim, device, num_layers=num_layers)
    criterion = AsymmetricHuberLoss(delta=loss_delta, alpha=loss_alpha)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    # FIX: scaler создаётся корректно — только для CUDA.
    scaler = _make_scaler(device)

    train_loader = build_loader(
        train_ds, batch_size=batch_size, shuffle=True, device=device)
    val_loader = build_loader(
        val_ds, batch_size=batch_size, shuffle=False, device=device)

    best_val_mse = float("inf")

    # FIX: try/finally гарантирует завершение mlflow run даже при pruning.
    with mlflow.start_run(run_name=f"{temporal_type}_trial_{trial.number:03d}", nested=True):
        try:
            mlflow.set_tag("hpo_mode", "frozen_cnn")
            mlflow.log_params(
                {
                    "temporal_type": temporal_type,
                    "hpo_mode": "frozen_cnn",
                    "lr": lr,
                    "hidden_size": hidden_size,
                    "dropout": dropout,
                    "batch_size": batch_size,
                    "num_layers": num_layers,
                    "loss_delta": loss_delta,
                    "loss_alpha": loss_alpha,
                    "mono_weight": mono_weight,
                    "weight_decay": weight_decay,
                    "epochs": epochs,
                    "use_feature_cache": True,
                    "requested_use_feature_cache": use_feature_cache,
                    "fine_tune_cnn": False,
                    "cnn_checkpoint_path": CNN_CHECKPOINT_PATH,
                    "encoder_dim": encoder_dim,
                }
            )
            for epoch in range(epochs):
                train_mse = train_one_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    device,
                    scaler=scaler,
                    mono_weight=mono_weight,
                )
                val_mse, val_mae, val_r2, val_rmse, val_phm = evaluate(
                    model, val_loader, criterion, device)
                mlflow.log_metrics(
                    {
                        "train_mse": train_mse,
                        "val_mse": val_mse,
                        "val_mae": val_mae,
                        "val_r2": val_r2,
                        "val_rmse": val_rmse,
                        "val_phm_score": val_phm,
                    },
                    step=epoch,
                )
                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                trial.report(val_mse, epoch)
                if trial.should_prune():
                    # FIX: помечаем run как pruned перед выходом.
                    mlflow.set_tag("pruned", "true")
                    raise optuna.TrialPruned()
            mlflow.log_metric("best_val_mse", best_val_mse)
        except optuna.TrialPruned:
            raise
        except Exception:
            mlflow.set_tag("failed", "true")
            raise

    return best_val_mse


def fit_and_save(
    temporal_type: str,
    params: Dict[str, float],
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    device: torch.device,
    window_size: int,
    *,
    epochs: int = EPOCHS,
    patience: int = PATIENCE,
    ckpt_suffix: str = "",
    figures_dir: Optional[str] = None,
    use_feature_cache: bool = True,
    encoder_dim: Optional[int] = None,
) -> Dict[str, float]:
    num_layers = int(params["num_layers"])
    loss_delta = float(params["loss_delta"])
    loss_alpha = float(params["loss_alpha"])
    mono_weight = float(params["mono_weight"])
    weight_decay = float(params["weight_decay"])
    base_lr = float(params["lr"])

    # Финальный fit всегда идёт end-to-end: CWT -> CNN -> temporal -> RUL.
    # HPO-параметры пришли из frozen-CNN feature-cache этапа через study.best_params.
    model = build_model(
        temporal_type=temporal_type,
        hidden_size=int(params["hidden_size"]),
        dropout=float(params["dropout"]),
        device=device,
        fine_tune=True,
        num_layers=num_layers,
    )
    set_encoder_trainable(model, False)
    criterion = AsymmetricHuberLoss(delta=loss_delta, alpha=loss_alpha)
    optimizer = build_finetune_optimizer(
        model, base_lr=base_lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )
    batch_size = int(params["batch_size"])
    # FIX: scaler создаётся корректно — только для CUDA.
    scaler = _make_scaler(device)

    train_loader = build_loader(
        train_ds, batch_size=batch_size, shuffle=True, device=device)
    val_loader = build_loader(
        val_ds, batch_size=batch_size, shuffle=False, device=device)
    test_loader = build_loader(
        test_ds, batch_size=batch_size, shuffle=False, device=device)

    best_val_mse = float("inf")
    best_state = None
    epochs_no_improve = 0
    train_hist: List[float] = []
    val_hist: List[float] = []

    # Суффикс отделяет fast-чекпоинты от продовых, чтобы не перезаписать их.
    suffix = f"_{ckpt_suffix}" if ckpt_suffix else ""
    run_name = f"final_{temporal_type}{suffix}"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("final_mode", "unfrozen_cnn")
        mlflow.log_params(
            {
                "temporal_type": temporal_type,
                "final_mode": "unfrozen_cnn",
                "best_lr": base_lr,
                "best_hidden_size": int(params["hidden_size"]),
                "best_dropout": float(params["dropout"]),
                "best_batch_size": int(params["batch_size"]),
                "num_layers": num_layers,
                "loss_delta": loss_delta,
                "loss_alpha": loss_alpha,
                "mono_weight": mono_weight,
                "weight_decay": weight_decay,
                "epochs": epochs,
                "patience": patience,
                "ckpt_suffix": ckpt_suffix or "none",
                "use_feature_cache": False,
                "requested_use_feature_cache": use_feature_cache,
                "fine_tune_cnn": True,
                "finetune_warmup_epochs": FINETUNE_WARMUP_EPOCHS,
                "encoder_lr_multiplier": 0.1,
                "cnn_checkpoint_path": CNN_CHECKPOINT_PATH,
                "hpo_encoder_dim": encoder_dim or "unknown",
            }
        )
        encoder_unfrozen = False
        for epoch in range(epochs):
            if (not encoder_unfrozen) and epoch >= FINETUNE_WARMUP_EPOCHS:
                set_encoder_trainable(model, True)
                optimizer = build_finetune_optimizer(
                    model, base_lr=base_lr, weight_decay=weight_decay)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
                )
                encoder_unfrozen = True
                best_val_mse = float("inf")
                best_state = None
                epochs_no_improve = 0
                mlflow.log_metric("encoder_unfrozen", 1.0, step=epoch)

            freeze_encoder_module = not encoder_unfrozen
            train_mse = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                scaler=scaler,
                mono_weight=mono_weight,
                freeze_encoder_module=freeze_encoder_module,
            )
            val_mse, val_mae, val_r2, val_rmse, val_phm = evaluate(
                model, val_loader, criterion, device)
            train_hist.append(train_mse)
            val_hist.append(val_mse)
            scheduler.step(val_mse)
            lr_by_group = {
                group.get("name", f"group_{idx}"): group["lr"]
                for idx, group in enumerate(optimizer.param_groups)
            }
            mlflow.log_metrics(
                {
                    "final_train_mse": train_mse,
                    "final_val_mse": val_mse,
                    "final_val_mae": val_mae,
                    "final_val_r2": val_r2,
                    "final_val_rmse": val_rmse,
                    "final_val_phm_score": val_phm,
                    "lr": max(group["lr"] for group in optimizer.param_groups),
                    "encoder_lr": lr_by_group.get("encoder", 0.0),
                    "head_lr": lr_by_group.get("head", 0.0),
                    "encoder_trainable": float(encoder_unfrozen),
                },
                step=epoch,
            )
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if (
                epochs_no_improve >= patience
                and epoch + 1 > FINETUNE_WARMUP_EPOCHS
            ):
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        (
            test_mse,
            test_mae,
            test_r2,
            test_rmse,
            test_phm,
            test_preds,
            test_labels,
        ) = evaluate_with_predictions(
            model, test_loader, criterion, device
        )
        test_preds_smoothed = ema_smooth(test_preds, alpha=EMA_ALPHA)
        inference_metrics = benchmark_inference_speed(
            model, test_loader, device)
        mlflow.log_metrics(
            {
                "test_mse": test_mse,
                "test_mae": test_mae,
                "test_r2": test_r2,
                "test_rmse": test_rmse,
                "test_phm_score": test_phm,
                **inference_metrics,
            }
        )
        print(
            "[INFO] Inference speed: "
            f"{inference_metrics['inference_ms_per_sample']:.4f} ms/sample, "
            f"{inference_metrics['inference_samples_per_sec']:.1f} samples/s "
            f"({int(inference_metrics['inference_batches'])} batches)"
        )

        os.makedirs(PREDS3_MODELS_DIR, exist_ok=True)
        ckpt_path = os.path.join(
            PREDS3_MODELS_DIR, f"best_rul_{temporal_type}_ws{
                window_size}_v3{suffix}.pth"
        )
        torch.save(
            {
                "temporal_type": temporal_type,
                "state_dict": model.state_dict(),
                "best_params": params,
                "best_val_mse": best_val_mse,
                "test_mse": test_mse,
                "test_mae": test_mae,
                "test_r2": test_r2,
                "test_rmse": test_rmse,
                "test_phm_score": test_phm,
                "test_predictions_raw": test_preds,
                "test_predictions_smoothed": test_preds_smoothed,
                "test_labels": test_labels,
                "ema_alpha": EMA_ALPHA,
                "inference_metrics": inference_metrics,
                "ckpt_suffix": ckpt_suffix or "none",
                "final_mode": "unfrozen_cnn",
                "use_feature_cache": False,
                "requested_use_feature_cache": use_feature_cache,
                "fine_tune_cnn": True,
                "finetune_warmup_epochs": FINETUNE_WARMUP_EPOCHS,
                "encoder_lr_multiplier": 0.1,
                "cnn_checkpoint_path": CNN_CHECKPOINT_PATH,
                "hpo_encoder_dim": encoder_dim,
            },
            ckpt_path,
        )
        mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")

        if figures_dir is not None:
            os.makedirs(figures_dir, exist_ok=True)
            title_suffix = f" {ckpt_suffix.upper()}" if ckpt_suffix else ""
            title_prefix = f"{temporal_type.upper()} ws={window_size}{
                title_suffix}"

            lc_path = os.path.join(figures_dir, f"{temporal_type}_ws{
                                   window_size}_learning_curves{suffix}.png")
            plot_learning_curves(
                train_hist,
                val_hist,
                lc_path,
                title=f"{title_prefix} learning curves",
            )
            mlflow.log_artifact(lc_path, artifact_path="figures")

            rul_path = os.path.join(figures_dir, f"{temporal_type}_ws{
                                    window_size}_rul_prediction{suffix}.png")
            plot_rul_prediction(
                test_labels,
                test_preds,
                rul_path,
                title=f"{title_prefix} test prediction",
            )
            mlflow.log_artifact(rul_path, artifact_path="figures")

            rul_smooth_path = os.path.join(figures_dir, f"{temporal_type}_ws{
                                           window_size}_rul_prediction_smoothed{suffix}.png")
            plot_rul_prediction(
                test_labels,
                test_preds_smoothed,
                rul_smooth_path,
                title=f"{title_prefix} test prediction smoothed",
            )
            mlflow.log_artifact(rul_smooth_path, artifact_path="figures")

            res_path = os.path.join(figures_dir, f"{temporal_type}_ws{
                                    window_size}_residuals{suffix}.png")
            plot_residuals(
                test_labels,
                test_preds,
                res_path,
                title=f"{title_prefix}",
            )
            mlflow.log_artifact(res_path, artifact_path="figures")

            res_smooth_path = os.path.join(figures_dir, f"{temporal_type}_ws{
                                           window_size}_residuals_smoothed{suffix}.png")
            plot_residuals(
                test_labels,
                test_preds_smoothed,
                res_smooth_path,
                title=f"{title_prefix} smoothed",
            )
            mlflow.log_artifact(res_smooth_path, artifact_path="figures")

    return {
        "best_val_mse": best_val_mse,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "test_r2": test_r2,
        "test_rmse": test_rmse,
        "test_phm_score": test_phm,
        **inference_metrics,
    }


def run_for_window_size(
    cfg: DatasetConfig,
    device: torch.device,
    *,
    experiment_name: str = MLFLOW_EXPERIMENT,
    profile: str = "full",
    n_trials: int = N_TRIALS,
    epochs: int = EPOCHS,
    patience: int = PATIENCE,
    temporal_types: Sequence[str] = TEMPORAL_TYPES,
    ckpt_suffix: str = "",
    figures_root: Optional[str] = None,
    use_feature_cache: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Полный цикл HPO + финального обучения для одного window_size.

    Параметры fast-режима передаются явно, чтобы не было глобального состояния
    и функция оставалась тестируемой независимо.
    """
    train_dirs = discover_bearing_dirs(
        DATASET_ROOT, cfg.modes, cfg.train_bearings)
    val_dirs = discover_bearing_dirs(DATASET_ROOT, cfg.modes, cfg.val_bearings)
    test_dirs = discover_bearing_dirs(
        DATASET_ROOT, cfg.modes, cfg.test_bearings)

    print("[INFO] Train bearing dirs:", len(train_dirs))
    print("[INFO] Val bearing dirs  :", len(val_dirs))
    print("[INFO] Test bearing dirs :", len(test_dirs))

    train_ds = MultiBearingRULDataset(
        train_dirs,
        seq_length=cfg.seq_length,
        window_size=cfg.window_size,
        seq_stride=cfg.seq_stride,
        cwt_scales=cfg.cwt_scales,
        rul_clip=cfg.rul_clip,
    )
    # FIX: для val и test используем val_test_stride=1,
    # чтобы оценка покрывала весь временной ряд без пропусков.
    val_ds = MultiBearingRULDataset(
        val_dirs,
        seq_length=cfg.seq_length,
        window_size=cfg.window_size,
        seq_stride=cfg.val_test_stride,
        cwt_scales=cfg.cwt_scales,
        rul_clip=cfg.rul_clip,
    )
    test_ds = MultiBearingRULDataset(
        test_dirs,
        seq_length=cfg.seq_length,
        window_size=cfg.window_size,
        seq_stride=cfg.val_test_stride,
        cwt_scales=cfg.cwt_scales,
        rul_clip=cfg.rul_clip,
    )
    print(f"[INFO] Samples: train={len(train_ds)}, val={
          len(val_ds)}, test={len(test_ds)}")
    print("[INFO] Two-phase training: HPO=frozen CNN feature-cache, final fit=unfrozen CNN")

    # HPO всегда использует frozen CNN features, поэтому кэш нужен независимо
    # от CLI-флага --feature-cache/--no-feature-cache.
    encoder_dim: Optional[int] = precompute_cnn_feature_cache(
        train_ds, val_ds, test_ds, device)
    hpo_train_ds: Dataset = FeatureBearingRULDataset(
        train_dirs,
        seq_length=cfg.seq_length,
        window_size=cfg.window_size,
        seq_stride=cfg.seq_stride,
        cwt_scales=cfg.cwt_scales,
        rul_clip=cfg.rul_clip,
    )
    hpo_val_ds: Dataset = FeatureBearingRULDataset(
        val_dirs,
        seq_length=cfg.seq_length,
        window_size=cfg.window_size,
        seq_stride=cfg.val_test_stride,
        cwt_scales=cfg.cwt_scales,
        rul_clip=cfg.rul_clip,
    )

    best_params_by_model: Dict[str, Dict[str, float]] = {}
    results: Dict[str, Dict[str, float]] = {}

    run_tag = profile
    window_figures_dir = None
    if figures_root is not None:
        window_figures_dir = os.path.join(figures_root, f"ws{cfg.window_size}")
        os.makedirs(window_figures_dir, exist_ok=True)

    with mlflow.start_run(run_name=f"three_models_v3_{run_tag}_ws{cfg.window_size}"):
        mlflow.log_params(
            {
                "fast_mode": profile == "fast",
                "experiment_name": experiment_name,
                "profile": profile,
                "n_trials_per_model": n_trials,
                "temporal_types": ",".join(temporal_types),
                "requested_use_feature_cache": use_feature_cache,
                "hpo_mode": "frozen_cnn",
                "final_mode": "unfrozen_cnn",
                "hpo_use_feature_cache": True,
                "hpo_encoder_dim": encoder_dim,
                "cnn_checkpoint_path": CNN_CHECKPOINT_PATH,
                "num_workers": NUM_WORKERS,
                "modes": ",".join(cfg.modes),
                "train_bearing_suffixes": ",".join(cfg.train_bearings),
                "val_bearing_suffixes": ",".join(cfg.val_bearings),
                "test_bearing_suffixes": ",".join(cfg.test_bearings),
                "seq_length": cfg.seq_length,
                "seq_stride": cfg.seq_stride,
                "val_test_stride": cfg.val_test_stride,
                "window_size": cfg.window_size,
                "cwt_scales": cfg.cwt_scales,
                "rul_clip": cfg.rul_clip,
                "epochs": epochs,
                "patience": patience,
            }
        )

        for temporal_type in temporal_types:
            print("\n" + "=" * 70)
            print(f"[INFO] Optuna for {
                  temporal_type.upper()} ({n_trials} trials)")
            print("=" * 70)
            if profile == "full":
                n_startup = min(5, max(1, n_trials // 3))
                n_warmup_steps = 3
            else:
                n_startup = min(3, max(1, n_trials // 3))
                n_warmup_steps = 2
            study = optuna.create_study(
                direction="minimize",
                study_name=f"three_models_v3_{run_tag}_{
                    temporal_type}_ws{cfg.window_size}",
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=n_startup, n_warmup_steps=n_warmup_steps
                ),
            )
            # FIX: используем default-аргумент в lambda, чтобы зафиксировать
            # значение temporal_type на момент итерации, а не захватывать
            # переменную цикла по ссылке. Без этого все три study оптимизировали бы
            # последний тип из TEMPORAL_TYPES.
            study.optimize(
                lambda tr, t=temporal_type: objective(
                    tr,
                    t,
                    hpo_train_ds,
                    hpo_val_ds,
                    device,
                    epochs=epochs,
                    use_feature_cache=use_feature_cache,
                    encoder_dim=encoder_dim,
                ),
                n_trials=n_trials,
                show_progress_bar=True,
            )
            best_params = study.best_trial.params
            best_params_by_model[temporal_type] = best_params
            mlflow.log_params(
                {f"{temporal_type}_best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric(
                f"{temporal_type}_best_val_mse", study.best_value)

            if window_figures_dir is not None:
                save_optuna_dashboard(
                    study,
                    out_dir=window_figures_dir,
                    prefix=f"{temporal_type}_{run_tag}_ws{cfg.window_size}",
                )

    for temporal_type in temporal_types:
        print("\n" + "=" * 70)
        print(f"[INFO] Final fit for {temporal_type.upper()} with best params")
        print("=" * 70)
        results[temporal_type] = fit_and_save(
            temporal_type=temporal_type,
            params=best_params_by_model[temporal_type],
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            device=device,
            window_size=cfg.window_size,
            epochs=epochs,
            patience=patience,
            ckpt_suffix=ckpt_suffix,
            figures_dir=window_figures_dir,
            use_feature_cache=use_feature_cache,
            encoder_dim=encoder_dim,
        )

    return results


def _parse_cli_list(
    values: Optional[Sequence[str]],
    *,
    cast=str,
) -> Optional[List]:
    if values is None:
        return None
    parsed: List = []
    for value in values:
        for part in str(value).split(","):
            item = part.strip()
            if item:
                parsed.append(cast(item))
    return parsed or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train three temporal RUL models.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Sanity-check режим: меньше trials, эпох, масштабов CWT и "
            "только один window_size/temporal_type. "
            "Чекпоинты сохраняются с суффиксом '_fast', продовые файлы не затрагиваются."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=("fast", "balanced", "full"),
        default="balanced",
        help="Training profile. Default: balanced.",
    )
    parser.add_argument(
        "--experiment-name",
        default=MLFLOW_EXPERIMENT,
        help=f"MLflow experiment name. Default: {MLFLOW_EXPERIMENT}.",
    )
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seq-stride", type=int, default=None)
    parser.add_argument("--val-test-stride", type=int, default=None)
    parser.add_argument("--cwt-scales", type=int, default=None)
    parser.add_argument("--rul-clip", type=float, default=None)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers. Use 0 in restricted multiprocessing environments.",
    )
    parser.add_argument("--window-sizes", nargs="+", default=None)
    parser.add_argument("--temporal-types", nargs="+", default=None)
    parser.add_argument(
        "--feature-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Backward-compatible flag. In v3 HPO always uses frozen CNN "
            "feature-cache, and final fit always fine-tunes the CNN end-to-end."
        ),
    )
    args = parser.parse_args()

    global NUM_WORKERS
    if args.num_workers is not None:
        NUM_WORKERS = max(0, args.num_workers)
    elif args.feature_cache:
        # Feature-cache datasets load small precomputed .npy vectors, so
        # multiprocessing usually costs more than it saves and can fail during
        # DataLoader cleanup in some WSL/sandboxed environments.
        NUM_WORKERS = 0

    profile = "fast" if args.fast else args.profile
    defaults = PROFILE_DEFAULTS[profile]
    window_sizes = _parse_cli_list(args.window_sizes, cast=int) or list(
        defaults["window_sizes"]
    )
    temporal_types = _parse_cli_list(args.temporal_types, cast=str) or list(
        defaults["temporal_types"]
    )
    n_trials = args.n_trials if args.n_trials is not None else int(
        defaults["n_trials"])
    epochs = args.epochs if args.epochs is not None else int(
        defaults["epochs"])
    patience = args.patience if args.patience is not None else int(
        defaults["patience"])
    seq_stride = args.seq_stride if args.seq_stride is not None else int(
        defaults["seq_stride"])
    val_test_stride = (
        args.val_test_stride
        if args.val_test_stride is not None
        else int(defaults["val_test_stride"])
    )
    cwt_scales = args.cwt_scales if args.cwt_scales is not None else int(
        defaults["cwt_scales"])
    rul_clip = args.rul_clip if args.rul_clip is not None else float(
        defaults["rul_clip"])
    ckpt_suffix = str(defaults["ckpt_suffix"])

    if profile == "fast":
        print("\n" + "!" * 78)
        print("  FAST MODE — sanity-check прогон (не для продовых результатов)")
        print(f"  trials={n_trials}, epochs={epochs}, "
              f"cwt_scales={cwt_scales}, stride={seq_stride}, "
              f"rul_clip={rul_clip}, workers={NUM_WORKERS}")
        print("!" * 78)
    elif profile == "balanced":
        print("\n" + "!" * 78)
        print("  BALANCED MODE — ускоренный дневной прогон")
        print(f"  trials={n_trials}, epochs={epochs}, "
              f"cwt_scales={cwt_scales}, stride={seq_stride}, "
              f"rul_clip={rul_clip}, workers={NUM_WORKERS}")
        print("!" * 78)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    experiment_name = args.experiment_name.strip()
    if not experiment_name:
        raise ValueError("--experiment-name must not be empty")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    print(f"[INFO] MLflow experiment: {experiment_name}")

    figures_root = os.path.join(FIGURES_BASE_DIR, profile)
    os.makedirs(figures_root, exist_ok=True)

    all_results: Dict[int, Dict[str, Dict[str, float]]] = {}
    summary_rows: List[Dict[str, object]] = []
    for ws in window_sizes:
        cfg = replace(
            DEFAULT_DATASET_CONFIG,
            window_size=ws,
            seq_stride=seq_stride,
            val_test_stride=val_test_stride,
            cwt_scales=cwt_scales,
            rul_clip=rul_clip,
        )
        print("\n" + "#" * 78)
        print(f"[INFO] Running {profile} pipeline for window_size={ws}")
        print("#" * 78)
        all_results[ws] = run_for_window_size(
            cfg,
            device,
            experiment_name=experiment_name,
            profile=profile,
            n_trials=n_trials,
            epochs=epochs,
            patience=patience,
            temporal_types=temporal_types,
            ckpt_suffix=ckpt_suffix,
            figures_root=figures_root,
            use_feature_cache=args.feature_cache,
        )
        for model_name in temporal_types:
            m = all_results[ws][model_name]
            summary_rows.append(
                {
                    "mode": profile,
                    "window_size": ws,
                    "temporal_type": model_name,
                    "best_val_mse": m["best_val_mse"],
                    "test_mse": m["test_mse"],
                    "test_mae": m["test_mae"],
                    "test_r2": m["test_r2"],
                    "test_rmse": m["test_rmse"],
                    "test_phm_score": m["test_phm_score"],
                    "inference_ms_per_sample": m["inference_ms_per_sample"],
                    "inference_samples_per_sec": m["inference_samples_per_sec"],
                }
            )

    mode_label = profile.upper()
    print("\n" + "=" * 78)
    print(f"  TRAINING COMPLETE (THREE_MODELS_V3) — {mode_label}")
    print("=" * 78)
    for ws in window_sizes:
        print(f"\n[window_size={ws}]")
        for model_name in temporal_types:
            m = all_results[ws][model_name]
            print(
                f"  {model_name:11s} | "
                f"best_val_mse={m['best_val_mse']:.6f} | "
                f"test_mse={m['test_mse']:.6f} | "
                f"test_mae={m['test_mae']:.4f} | "
                f"test_r2={m['test_r2']:.4f} | "
                f"test_rmse={m['test_rmse']:.4f} | "
                f"test_phm={m['test_phm_score']:.4f} | "
                f"infer={m['inference_ms_per_sample']:.4f} ms/sample "
                f"({m['inference_samples_per_sec']:.1f} samples/s)"
            )

    if profile == "fast":
        print("\n[!] Это был --fast прогон. Запусти без флага для полного обучения.")

    # Сохраняем общий дашборд/сводку по текущему запуску.
    save_summary_dashboard(summary_rows, figures_root)
    print(f"[INFO] Сводные визуализации сохранены в: {figures_root}")


if __name__ == "__main__":
    main()
