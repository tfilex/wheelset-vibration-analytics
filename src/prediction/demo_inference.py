from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import pywt
import scipy.io
import scipy.signal
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.classification.model import get_model
from src.demo.mock_data import SIGNAL_TYPES
from src.prediction.data_loader import RULDataset
from src.prediction.model import (
    SUPPORTED_TEMPORAL,
    TemporalOnlyRULNet,
    UniversalHybridRULNet,
    create_cnn_encoder,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEMO_MODELS_DIR = PROJECT_ROOT / "models/demo_best"
DEMO_MODELS_DIR = Path(
    os.getenv("DEMO_MODELS_DIR", str(DEFAULT_DEMO_MODELS_DIR))
).expanduser()
DEMO_CLASSIFICATION_MODELS_DIR = DEMO_MODELS_DIR / "classification"
DEMO_RUL_MODELS_DIR = DEMO_MODELS_DIR / "rul"

FALLBACK_CWRU_CHECKPOINT = PROJECT_ROOT / "models/cnn/best_resnet18.pth"
FALLBACK_RUL_CHECKPOINT = PROJECT_ROOT / "models/pred_0/best_rul_lstm.pth"
FALLBACK_RUL_ENCODER_CHECKPOINT = PROJECT_ROOT / "models/cnn/best_resnet18_rul.pth"
DEFAULT_CWRU_CHECKPOINT = DEMO_CLASSIFICATION_MODELS_DIR / "cwru_classifier.pth"
DEFAULT_RUL_CHECKPOINT = DEMO_RUL_MODELS_DIR / "xjtu_rul.pth"
LEGACY_CWRU_CHECKPOINT = DEMO_MODELS_DIR / "cwru_classifier.pth"
LEGACY_RUL_CHECKPOINT = DEMO_MODELS_DIR / "xjtu_rul.pth"

DEMO_RUL_MODEL_DIRS = (
    PROJECT_ROOT / "models/pred_0",
    PROJECT_ROOT / "models/preds_2_unfrozen",
    PROJECT_ROOT / "models/preds_3",
    PROJECT_ROOT / "models/preds_3_frozen",
    PROJECT_ROOT / "models/preds_3_rnn",
)

ADVANCED_RUL_TEMPORAL_TYPES = {
    "bilstm",
    "lstm_attn",
    "bigru",
    "gru_attn",
    "transformer_improved",
}
DEMO_SUPPORTED_RUL_TEMPORAL_TYPES = set(SUPPORTED_TEMPORAL) | ADVANCED_RUL_TEMPORAL_TYPES

CWRU_SAMPLE_FILES = {
    "Норма": PROJECT_ROOT / "data/raw/CWRU/0_Normal/98.mat",
    "Дефект внутреннего кольца": PROJECT_ROOT / "data/raw/CWRU/1_IR_007/106.mat",
    "Дефект внешнего кольца": PROJECT_ROOT / "data/raw/CWRU/7_OR_007/131.mat",
}

XJTU_BEARING_DIRS = {
    "Bearing1_3": PROJECT_ROOT / "data/raw/XJTU-SY/35Hz12kN/Bearing1_3",
    "Bearing1_4": PROJECT_ROOT / "data/raw/XJTU-SY/35Hz12kN/Bearing1_4",
    "Bearing2_5": PROJECT_ROOT / "data/raw/XJTU-SY/37.5Hz11kN/Bearing2_5",
    "Bearing3_3": PROJECT_ROOT / "data/raw/XJTU-SY/40Hz10kN/Bearing3_3",
}

CWRU_CLASS_NAMES = {
    0: "Норма",
    1: "IR 0.007",
    2: "IR 0.014",
    3: "IR 0.021",
    4: "Ball 0.007",
    5: "Ball 0.014",
    6: "Ball 0.021",
    7: "OR 0.007",
    8: "OR 0.014",
    9: "OR 0.021",
}


@dataclass(frozen=True)
class ClassificationBundle:
    model: torch.nn.Module
    device: torch.device
    checkpoint_path: str
    model_name: str
    num_classes: int
    best_val_f1: float | None


@dataclass(frozen=True)
class RULBundle:
    model: torch.nn.Module
    device: torch.device
    checkpoint_path: str
    temporal_type: str
    seq_length: int
    hidden_size: int
    dropout: float
    test_mse: float | None
    test_mae: float | None
    test_r2: float | None
    data_pipeline: str
    window_size: int
    cwt_scales: int
    rul_clip: float


class ClassificationResult(TypedDict):
    is_normal: bool
    status: str
    message: str
    selected_signal: str
    predicted_class_id: int
    predicted_class_name: str
    confidence: float
    source_file: str
    model_name: str
    checkpoint_path: str
    signal: pd.DataFrame
    attribution: np.ndarray


class ModelOption(TypedDict):
    path: str
    label: str


class BiLSTMHead(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size // 2,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


class LSTMAttentionHead(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        weights = torch.softmax(self.attn(out).squeeze(-1), dim=1)
        context = (out * weights.unsqueeze(-1)).sum(dim=1)
        return self.fc(self.dropout(context))


class BiGRUHead(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size // 2,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out[:, -1])


class GRUAttentionHead(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        weights = torch.softmax(self.attn(out).squeeze(-1), dim=1)
        context = (out * weights.unsqueeze(-1)).sum(dim=1)
        return self.fc(self.dropout(context))


class ImprovedTransformerHead(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        nhead: int = 4,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = self.input_proj(x)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : seq_len + 1, :]
        x = self.transformer(x)
        cls_out = self.norm(x[:, 0])
        return self.fc(self.dropout(cls_out))


class EncoderHeadRULNet(nn.Module):
    def __init__(self, encoder: nn.Module, head: nn.Module, fine_tune: bool = False):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.fine_tune = fine_tune
        if not fine_tune:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels, height, width = x.shape
        frames = x.view(batch_size * seq_len, channels, height, width)
        with torch.set_grad_enabled(self.fine_tune):
            features = self.encoder(frames)
        features = features.view(batch_size, seq_len, -1)
        return self.head(features)


class DemoRULDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        seq_length: int,
        window_size: int,
        cwt_scales: int,
        rul_clip: float,
        normalize_scalograms: bool,
    ):
        self.data_dir = data_dir
        self.seq_length = seq_length
        self.window_size = window_size
        self.cwt_widths = np.arange(1, cwt_scales + 1)
        self.rul_clip = rul_clip
        self.normalize_scalograms = normalize_scalograms

        files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
        files.sort(key=lambda f: int(re.sub(r"\D", "", f)))
        self.file_paths = [os.path.join(data_dir, f) for f in files]
        if len(self.file_paths) < self.seq_length:
            raise ValueError(
                f"Not enough files ({len(self.file_paths)}) "
                f"for sequence length {self.seq_length}."
            )

    def __len__(self) -> int:
        return len(self.file_paths) - self.seq_length + 1

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        if not self.normalize_scalograms:
            return values
        return (values - values.mean()) / (values.std() + 1e-8)

    def _process_file(self, file_path: str) -> np.ndarray:
        df = pd.read_csv(file_path)
        if len(df) < self.window_size:
            h_sig = np.pad(df.iloc[:, 0].values, (0, self.window_size - len(df)))
            v_sig = np.pad(df.iloc[:, 1].values, (0, self.window_size - len(df)))
        else:
            h_sig = df.iloc[: self.window_size, 0].values.astype(np.float32)
            v_sig = df.iloc[: self.window_size, 1].values.astype(np.float32)

        cwt_h, _ = pywt.cwt(h_sig, self.cwt_widths, "mexh")
        cwt_v, _ = pywt.cwt(v_sig, self.cwt_widths, "mexh")
        scalogram = np.stack([self._normalize(cwt_h), self._normalize(cwt_v)], axis=0)
        return scalogram.astype(np.float32)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq_paths = self.file_paths[idx: idx + self.seq_length]
        scalograms = [self._process_file(path) for path in seq_paths]

        total_steps = max(len(self.file_paths) - 1, 1)
        rul = min(1.0 - (idx / total_steps), self.rul_clip)

        x = torch.tensor(np.stack(scalograms, axis=0), dtype=torch.float32)
        y = torch.tensor([rul], dtype=torch.float32)
        return x, y


def _resolve_checkpoint(
    env_name: str,
    preferred: Path,
    fallback: Path,
    legacy: Path | None = None,
) -> Path:
    override = os.getenv(env_name)
    if override:
        return Path(override).expanduser()
    if preferred.exists():
        return preferred
    if legacy and legacy.exists():
        return legacy
    return fallback


def get_cwru_checkpoint_path() -> Path:
    return _resolve_checkpoint(
        "CWRU_CLASSIFIER_CHECKPOINT",
        DEFAULT_CWRU_CHECKPOINT,
        FALLBACK_CWRU_CHECKPOINT,
        LEGACY_CWRU_CHECKPOINT,
    )


def get_rul_checkpoint_path() -> Path:
    return _resolve_checkpoint(
        "XJTU_RUL_CHECKPOINT",
        DEFAULT_RUL_CHECKPOINT,
        FALLBACK_RUL_CHECKPOINT,
        LEGACY_RUL_CHECKPOINT,
    )


def _as_project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _is_advanced_rul_checkpoint(checkpoint: dict[str, Any]) -> bool:
    temporal_type = checkpoint.get("temporal_type")
    return (
        temporal_type in ADVANCED_RUL_TEMPORAL_TYPES
        or "ckpt_suffix" in checkpoint
        or "final_mode" in checkpoint
        or "hpo_encoder_dim" in checkpoint
    )


def _infer_window_size(path: Path, checkpoint: dict[str, Any]) -> int:
    for source in (path.name, str(checkpoint.get("ckpt_suffix", ""))):
        match = re.search(r"ws(\d+)", source)
        if match:
            return int(match.group(1))
    return int(checkpoint.get("window_size") or 1024)


def _rul_data_pipeline(checkpoint: dict[str, Any]) -> str:
    return "v3_zscore_start_rul" if _is_advanced_rul_checkpoint(checkpoint) else "legacy"


def _encoder_checkpoint_from_checkpoint(checkpoint: dict[str, Any]) -> Path | None:
    checkpoint_path = checkpoint.get("cnn_checkpoint_path")
    if checkpoint_path:
        return _as_project_path(checkpoint_path)
    if FALLBACK_RUL_ENCODER_CHECKPOINT.exists():
        return FALLBACK_RUL_ENCODER_CHECKPOINT
    if FALLBACK_CWRU_CHECKPOINT.exists():
        return FALLBACK_CWRU_CHECKPOINT
    return None


def get_checkpoint_fingerprint(path: str | Path) -> str:
    resolved_path = _as_project_path(path)
    stat = resolved_path.stat()
    return f"{resolved_path}:{stat.st_mtime_ns}:{stat.st_size}"


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _checkpoint_candidates(*paths: Path) -> list[Path]:
    seen: set[Path] = set()
    candidates: list[Path] = []
    for path in paths:
        expanded = sorted(path.glob("*.pth")) if path.is_dir() else [path]
        for item in expanded:
            resolved = item.resolve()
            if resolved in seen or not item.exists():
                continue
            seen.add(resolved)
            candidates.append(item)
    return candidates


def _load_checkpoint_metadata(path: Path) -> dict[str, Any] | None:
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        return None
    return checkpoint


def _infer_temporal_layers(state_dict: dict[str, torch.Tensor], temporal_type: str) -> int:
    layer_indices: list[int] = []
    if temporal_type in ("lstm", "gru"):
        prefixes = (
            f"{temporal_type}.weight_ih_l",
            f"head.{temporal_type}.weight_ih_l",
            "temporal.weight_ih_l",
        )
        for key in state_dict:
            for prefix in prefixes:
                if not key.startswith(prefix):
                    continue
                suffix = key.removeprefix(prefix)
                suffix = suffix.removesuffix("_reverse")
                if suffix.isdigit():
                    layer_indices.append(int(suffix))
                break
    elif temporal_type in ("bilstm", "lstm_attn"):
        prefixes = ("lstm.weight_ih_l", "head.lstm.weight_ih_l")
        for key in state_dict:
            for prefix in prefixes:
                if not key.startswith(prefix):
                    continue
                suffix = key.removeprefix(prefix).removesuffix("_reverse")
                if suffix.isdigit():
                    layer_indices.append(int(suffix))
                break
    elif temporal_type in ("bigru", "gru_attn"):
        prefixes = ("gru.weight_ih_l", "head.gru.weight_ih_l")
        for key in state_dict:
            for prefix in prefixes:
                if not key.startswith(prefix):
                    continue
                suffix = key.removeprefix(prefix).removesuffix("_reverse")
                if suffix.isdigit():
                    layer_indices.append(int(suffix))
                break
    elif temporal_type in ("transformer", "transformer_improved"):
        prefixes = ("temporal.layers.", "transformer.layers.", "head.transformer.layers.")
        for key in state_dict:
            for prefix in prefixes:
                if not key.startswith(prefix):
                    continue
                parts = key.removeprefix(prefix).split(".")
                if parts and parts[0].isdigit():
                    layer_indices.append(int(parts[0]))
                break
    elif temporal_type == "tcn":
        prefix = "temporal.network."
        for key in state_dict:
            if key.startswith(prefix):
                parts = key.split(".")
                if len(parts) > 2 and parts[2].isdigit():
                    layer_indices.append(int(parts[2]))
    return max(layer_indices) + 1 if layer_indices else 2


def _rul_params_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    params = checkpoint.get("best_params") or checkpoint.get("hyperparams") or {}
    temporal_type = checkpoint.get("temporal_type") or params.get("temporal_type") or "lstm"
    return {
        "temporal_type": temporal_type,
        "hidden_size": int(checkpoint.get("hidden_size") or params.get("hidden_size") or 64),
        "dropout": float(checkpoint.get("dropout") or params.get("dropout") or 0.2),
        "seq_length": int(checkpoint.get("seq_length") or params.get("seq_length") or 10),
        "num_layers": int(
            checkpoint.get("num_temporal_layers")
            or checkpoint.get("num_layers")
            or params.get("num_layers")
            or _infer_temporal_layers(checkpoint["state_dict"], temporal_type)
        ),
    }


def discover_classification_models() -> list[ModelOption]:
    options: list[ModelOption] = []
    candidates = _checkpoint_candidates(
        DEFAULT_CWRU_CHECKPOINT,
        LEGACY_CWRU_CHECKPOINT,
        FALLBACK_CWRU_CHECKPOINT,
        DEMO_CLASSIFICATION_MODELS_DIR,
        PROJECT_ROOT / "models/cnn",
    )

    for path in candidates:
        checkpoint = _load_checkpoint_metadata(path)
        if not checkpoint or "num_classes" not in checkpoint:
            continue
        model_name = checkpoint.get("model_name", "unknown")
        best_val_f1 = checkpoint.get("best_val_f1")
        metric = f", val F1={best_val_f1:.3f}" if isinstance(best_val_f1, float) else ""
        options.append(
            {
                "path": str(path),
                "label": f"{_relative_path(path)} ({model_name}{metric})",
            }
        )
    return options


def discover_rul_models() -> list[ModelOption]:
    ranked_options: list[tuple[float, ModelOption]] = []
    candidates = _checkpoint_candidates(
        DEFAULT_RUL_CHECKPOINT,
        LEGACY_RUL_CHECKPOINT,
        FALLBACK_RUL_CHECKPOINT,
        DEMO_RUL_MODELS_DIR,
        *DEMO_RUL_MODEL_DIRS,
    )

    for path in candidates:
        checkpoint = _load_checkpoint_metadata(path)
        if not checkpoint or "temporal_type" not in checkpoint:
            continue
        params = _rul_params_from_checkpoint(checkpoint)
        temporal_type = params["temporal_type"]
        if temporal_type not in DEMO_SUPPORTED_RUL_TEMPORAL_TYPES:
            continue

        test_mse = checkpoint.get("test_mse")
        test_mae = checkpoint.get("test_mae")
        test_r2 = checkpoint.get("test_r2")
        metrics = []
        if isinstance(test_mse, float):
            metrics.append(f"MSE={test_mse:.4f}")
        if isinstance(test_mae, float):
            metrics.append(f"MAE={test_mae:.4f}")
        if isinstance(test_r2, float):
            metrics.append(f"R2={test_r2:.3f}")
        metrics_text = f", {', '.join(metrics)}" if metrics else ""
        option = (
            {
                "path": str(path),
                "label": (
                    f"{_relative_path(path)} "
                    f"({temporal_type.upper()}, seq={params['seq_length']}{metrics_text})"
                ),
            }
        )
        rank = float(test_mse) if isinstance(test_mse, float) else float("inf")
        ranked_options.append((rank, option))
    return [option for _, option in sorted(ranked_options, key=lambda item: item[0])]


@st.cache_resource
def load_resnet18_classifier(
    checkpoint_path: str | None = None,
    checkpoint_fingerprint: str | None = None,
) -> ClassificationBundle:
    """Загрузка реального ResNet-18 checkpoint для классификации CWRU."""
    resolved_checkpoint_path = (
        _as_project_path(checkpoint_path)
        if checkpoint_path
        else get_cwru_checkpoint_path()
    )
    if not resolved_checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved_checkpoint_path}")
    _ = checkpoint_fingerprint or get_checkpoint_fingerprint(resolved_checkpoint_path)

    checkpoint = torch.load(resolved_checkpoint_path, map_location="cpu")
    model_name = checkpoint.get("model_name", "resnet18")
    num_classes = int(checkpoint.get("num_classes", 10))

    model = get_model(model_name, num_classes=num_classes, in_channels=1)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return ClassificationBundle(
        model=model,
        device=device,
        checkpoint_path=str(resolved_checkpoint_path),
        model_name=model_name,
        num_classes=num_classes,
        best_val_f1=checkpoint.get("best_val_f1"),
    )


def _build_advanced_rul_head(
    temporal_type: str,
    encoder_dim: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
) -> nn.Module:
    if temporal_type == "bilstm":
        return BiLSTMHead(encoder_dim, hidden_size, num_layers, dropout)
    if temporal_type == "lstm_attn":
        return LSTMAttentionHead(encoder_dim, hidden_size, num_layers, dropout)
    if temporal_type == "bigru":
        return BiGRUHead(encoder_dim, hidden_size, num_layers, dropout)
    if temporal_type == "gru_attn":
        return GRUAttentionHead(encoder_dim, hidden_size, num_layers, dropout)
    if temporal_type == "transformer_improved":
        return ImprovedTransformerHead(encoder_dim, hidden_size, num_layers, dropout)
    if temporal_type in SUPPORTED_TEMPORAL:
        return TemporalOnlyRULNet(
            encoder_dim=encoder_dim,
            temporal_type=temporal_type,
            hidden_size=hidden_size,
            dropout=dropout,
            num_temporal_layers=num_layers,
        )
    raise ValueError(
        f"Unsupported RUL temporal_type: {temporal_type}. "
        f"Supported: {sorted(DEMO_SUPPORTED_RUL_TEMPORAL_TYPES)}"
    )


def _build_rul_model_from_checkpoint(
    checkpoint: dict[str, Any],
    temporal_type: str,
    hidden_size: int,
    dropout: float,
    num_layers: int,
) -> nn.Module:
    state_dict = checkpoint["state_dict"]
    has_encoder_state = any(key.startswith("encoder.") for key in state_dict)
    has_head_state = any(key.startswith("head.") for key in state_dict)
    needs_feature_head = temporal_type in ADVANCED_RUL_TEMPORAL_TYPES or not has_encoder_state

    encoder_checkpoint = None if has_encoder_state else _encoder_checkpoint_from_checkpoint(checkpoint)
    encoder, encoder_dim = create_cnn_encoder(
        backbone_name=checkpoint.get("cnn_backbone", "resnet18"),
        in_channels=2,
        pretrained=False,
        freeze=not has_encoder_state,
        checkpoint_path=str(encoder_checkpoint) if encoder_checkpoint else None,
    )

    if needs_feature_head:
        head = _build_advanced_rul_head(
            temporal_type=temporal_type,
            encoder_dim=encoder_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        model = EncoderHeadRULNet(
            encoder=encoder,
            head=head,
            fine_tune=bool(has_encoder_state),
        )
        if has_head_state:
            model.load_state_dict(state_dict)
        else:
            model.head.load_state_dict(state_dict)
        return model

    model = UniversalHybridRULNet(
        encoder=encoder,
        encoder_dim=encoder_dim,
        temporal_type=temporal_type,
        hidden_size=hidden_size,
        dropout=dropout,
        num_temporal_layers=num_layers,
        fine_tune=True,
    )
    model.load_state_dict(state_dict)
    return model


@st.cache_resource
def load_cnn_lstm_rul_model(
    checkpoint_path: str | None = None,
    checkpoint_fingerprint: str | None = None,
) -> RULBundle:
    """Загрузка реального CNN+LSTM checkpoint для прогноза RUL XJTU-SY."""
    resolved_checkpoint_path = (
        _as_project_path(checkpoint_path)
        if checkpoint_path
        else get_rul_checkpoint_path()
    )
    if not resolved_checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved_checkpoint_path}")
    _ = checkpoint_fingerprint or get_checkpoint_fingerprint(resolved_checkpoint_path)

    checkpoint = torch.load(resolved_checkpoint_path, map_location="cpu")
    params = _rul_params_from_checkpoint(checkpoint)
    temporal_type = params["temporal_type"]
    hidden_size = params["hidden_size"]
    dropout = params["dropout"]
    seq_length = params["seq_length"]
    num_layers = params["num_layers"]
    if temporal_type not in DEMO_SUPPORTED_RUL_TEMPORAL_TYPES:
        raise ValueError(
            f"Unsupported RUL temporal_type: {temporal_type}. "
            f"Supported: {sorted(DEMO_SUPPORTED_RUL_TEMPORAL_TYPES)}"
        )

    model = _build_rul_model_from_checkpoint(
        checkpoint=checkpoint,
        temporal_type=temporal_type,
        hidden_size=hidden_size,
        dropout=dropout,
        num_layers=num_layers,
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    data_pipeline = _rul_data_pipeline(checkpoint)

    return RULBundle(
        model=model,
        device=device,
        checkpoint_path=str(resolved_checkpoint_path),
        temporal_type=temporal_type,
        seq_length=seq_length,
        hidden_size=hidden_size,
        dropout=dropout,
        test_mse=checkpoint.get("test_mse"),
        test_mae=checkpoint.get("test_mae"),
        test_r2=checkpoint.get("test_r2"),
        data_pipeline=data_pipeline,
        window_size=_infer_window_size(resolved_checkpoint_path, checkpoint),
        cwt_scales=int(checkpoint.get("cwt_scales") or 32),
        rul_clip=0.8 if data_pipeline != "legacy" else 1.0,
    )


def _extract_de_signal(mat_path: Path) -> np.ndarray:
    mat_data = scipy.io.loadmat(mat_path)
    for key, value in mat_data.items():
        if not key.startswith("__") and key.endswith("_DE_time"):
            return value.flatten().astype(np.float32)
    raise ValueError(f"DE_time signal not found in {mat_path}")


def _build_cwru_input(signal_type: str, window_size: int = 1024) -> tuple[pd.DataFrame, torch.Tensor, str]:
    if signal_type not in SIGNAL_TYPES:
        raise ValueError(f"Unknown signal type: {signal_type}")

    mat_path = CWRU_SAMPLE_FILES[signal_type]
    raw_signal = _extract_de_signal(mat_path)
    if raw_signal.size < window_size:
        raise ValueError(f"Signal is shorter than {window_size}: {mat_path}")

    start = 0
    window = raw_signal[start: start + window_size]
    _, _, zxx = scipy.signal.stft(window, nperseg=256, noverlap=128)
    spectrogram = np.abs(zxx) ** 2

    signal_df = pd.DataFrame(
        {
            "time": np.arange(window_size, dtype=np.float32),
            "amplitude": window.astype(np.float32),
        }
    )
    tensor = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)
    return signal_df, tensor, str(mat_path)


def _build_attribution(bundle: ClassificationBundle, spectrogram: torch.Tensor) -> np.ndarray:
    x = spectrogram.unsqueeze(0).to(bundle.device)
    x.requires_grad_(True)

    logits = bundle.model(x)
    class_idx = int(logits.argmax(dim=1).item())
    score = logits[0, class_idx]
    bundle.model.zero_grad(set_to_none=True)
    score.backward()

    attribution = (x.grad.detach().abs() * x.detach().abs()).squeeze().cpu().numpy()
    attribution = attribution / (attribution.max() + 1e-8)
    return attribution


def classify_signal(
    signal_type: str,
    checkpoint_path: str | None = None,
    checkpoint_fingerprint: str | None = None,
) -> ClassificationResult:
    bundle = load_resnet18_classifier(checkpoint_path, checkpoint_fingerprint)
    signal_df, spectrogram, source_file = _build_cwru_input(signal_type)

    with torch.no_grad():
        logits = bundle.model(spectrogram.unsqueeze(0).to(bundle.device))
        probabilities = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    predicted_id = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_id])
    predicted_name = CWRU_CLASS_NAMES.get(predicted_id, f"Класс {predicted_id}")
    is_normal = predicted_id == 0
    status = "OK" if is_normal else "DEFECT"

    if is_normal:
        message = (
            "Диагностика завершена: модель ResNet-18 классифицировала "
            "сигнал как норму."
        )
    else:
        message = (
            "Диагностика завершена: модель ResNet-18 обнаружила дефект "
            f"({predicted_name})."
        )

    return {
        "is_normal": is_normal,
        "status": status,
        "message": message,
        "selected_signal": signal_type,
        "predicted_class_id": predicted_id,
        "predicted_class_name": predicted_name,
        "confidence": confidence,
        "source_file": source_file,
        "model_name": bundle.model_name,
        "checkpoint_path": bundle.checkpoint_path,
        "signal": signal_df,
        "attribution": _build_attribution(bundle, spectrogram),
    }


def _interpolate_series(anchor_steps: np.ndarray, values: np.ndarray, output_steps: int) -> np.ndarray:
    target_steps = np.linspace(1, output_steps, output_steps)
    interpolated = np.interp(target_steps, anchor_steps, values)
    return np.clip(interpolated, 0.0, 1.0)


@st.cache_data(show_spinner=False)
def predict_rul_series(
    bearing: str,
    checkpoint_path: str | None = None,
    checkpoint_fingerprint: str | None = None,
    output_steps: int = 100,
    anchor_points: int = 16,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    bundle = load_cnn_lstm_rul_model(checkpoint_path, checkpoint_fingerprint)
    bearing_dir = XJTU_BEARING_DIRS[bearing]
    if bundle.data_pipeline == "legacy":
        dataset = RULDataset(
            str(bearing_dir),
            seq_length=bundle.seq_length,
            window_size=bundle.window_size,
            cwt_widths=np.arange(1, bundle.cwt_scales + 1),
        )
    else:
        dataset = DemoRULDataset(
            str(bearing_dir),
            seq_length=bundle.seq_length,
            window_size=bundle.window_size,
            cwt_scales=bundle.cwt_scales,
            rul_clip=bundle.rul_clip,
            normalize_scalograms=True,
        )

    n_anchors = min(anchor_points, len(dataset))
    if n_anchors <= 0:
        raise ValueError(f"No RUL samples available for bearing: {bearing}")

    dataset_indices = np.linspace(0, len(dataset) - 1, n_anchors, dtype=int)
    anchor_steps = np.linspace(1, output_steps, n_anchors)

    samples = []
    true_values = []
    for index in dataset_indices:
        x, y = dataset[int(index)]
        samples.append(x)
        true_values.append(float(y.item()))

    batch = torch.stack(samples).to(bundle.device)
    with torch.no_grad():
        predictions = bundle.model(batch).squeeze(-1).detach().cpu().numpy()

    true_values_np = np.array(true_values, dtype=np.float32)
    predictions_np = np.array(predictions, dtype=np.float32)

    history = pd.DataFrame(
        {
            "step": np.arange(1, output_steps + 1, dtype=np.float32),
            "true_rul": _interpolate_series(anchor_steps, true_values_np, output_steps),
            "pred_rul": _interpolate_series(anchor_steps, predictions_np, output_steps),
        }
    )
    metadata = {
        "model_name": f"CNN + {bundle.temporal_type.upper()}",
        "checkpoint_path": bundle.checkpoint_path,
        "bearing_dir": str(bearing_dir),
        "seq_length": bundle.seq_length,
        "window_size": bundle.window_size,
        "data_pipeline": bundle.data_pipeline,
        "anchor_points": n_anchors,
        "test_mse": bundle.test_mse,
        "test_mae": bundle.test_mae,
        "test_r2": bundle.test_r2,
    }
    return history, metadata
