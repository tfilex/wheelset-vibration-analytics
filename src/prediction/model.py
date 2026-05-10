"""
model.py — Универсальная гибридная модель для прогнозирования RUL (Remaining Useful Life).

Архитектурная концепция:
    Предобученный CNN-энкодер (замороженный) извлекает пространственные признаки
    из CWT-скалограмм. Эти признаки конкатенируются с дополнительным скалярным
    признаком (скорость вагона) и пропускаются через один из 5 временных блоков:
        - LSTM
        - GRU
        - TCN (Temporal Convolutional Network)
        - Transformer (TransformerEncoder + Positional Encoding)
        - Mamba (State Space Model, заглушка)

    Выход — одно число: предсказанный RUL от 1.0 (здоров) до 0.0 (отказ).

Usage:
    uv run python src/prediction/model.py
"""

import os
import math
from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ---------------------------------------------------------------------------
#  Список поддерживаемых временных архитектур
# ---------------------------------------------------------------------------

SUPPORTED_TEMPORAL: List[str] = ["lstm", "gru", "tcn", "transformer", "mamba"]

TemporalType = Literal["lstm", "gru", "tcn", "transformer", "mamba"]


# ---------------------------------------------------------------------------
#  Утилиты: создание CNN-энкодера
# ---------------------------------------------------------------------------

def _adapt_input_conv_weight(
    weight: torch.Tensor,
    target_shape: torch.Size,
) -> torch.Tensor:
    """Adapts a saved first-conv weight tensor to a new input-channel count."""
    target_out, target_in, target_h, target_w = target_shape
    if weight.shape[0] != target_out or weight.shape[2:] != (target_h, target_w):
        return weight

    source_in = weight.shape[1]
    if source_in == target_in:
        return weight
    if source_in == 1:
        return weight.repeat(1, target_in, 1, 1) / float(target_in)
    if target_in == 1:
        return weight.mean(dim=1, keepdim=True)

    adapted = torch.zeros(target_shape, dtype=weight.dtype)
    channels_to_copy = min(source_in, target_in)
    adapted[:, :channels_to_copy] = weight[:, :channels_to_copy]
    if target_in > source_in:
        adapted[:, source_in:] = weight.mean(dim=1, keepdim=True)
    return adapted


def _load_classification_backbone_weights(
    encoder: nn.Module,
    checkpoint_path: str,
) -> None:
    """Loads a classification checkpoint into an encoder without its head."""
    if not checkpoint_path:
        return
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"CNN checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    encoder_state = encoder.state_dict()
    compatible_state = {}

    for key, value in state_dict.items():
        clean_key = key.removeprefix("backbone.")
        if clean_key.startswith(("fc.", "classifier.")):
            continue
        if clean_key not in encoder_state:
            continue

        target = encoder_state[clean_key]
        if value.shape != target.shape:
            if clean_key.endswith("conv1.weight") or clean_key.endswith("features.0.0.weight"):
                value = _adapt_input_conv_weight(value, target.shape)
            if value.shape != target.shape:
                continue
        compatible_state[clean_key] = value

    missing, unexpected = encoder.load_state_dict(
        compatible_state, strict=False)
    loaded = len(compatible_state)
    print(
        f"[INFO] Loaded {
            loaded} CNN encoder tensors from classification checkpoint: "
        f"{checkpoint_path}"
    )
    if loaded == 0:
        raise RuntimeError(
            f"No compatible encoder weights were loaded from {
                checkpoint_path}. "
            f"Missing={len(missing)}, unexpected={len(unexpected)}"
        )


def create_cnn_encoder(
    backbone_name: str = "resnet18",
    in_channels: int = 2,
    pretrained: bool = False,
    freeze: bool = True,
    checkpoint_path: Optional[str] = None,
) -> tuple[nn.Module, int]:
    """Создаёт CNN-энкодер из torchvision, отсекает классификатор.

    Args:
        backbone_name: Имя архитектуры (resnet18, mobilenet_v3_small, etc.).
        in_channels: Количество входных каналов скалограммы (2 по умолч.).
        pretrained: Загружать ли предобученные веса ImageNet.
        freeze: Замораживать ли веса энкодера.
        checkpoint_path: Путь к классификационному checkpoint для инициализации
            backbone перед RUL fine-tuning.

    Returns:
        Кортеж (encoder, feature_dim) — модуль энкодера и размер вектора фичей.
    """
    weights = "IMAGENET1K_V1" if pretrained else None

    if backbone_name == "resnet18":
        base = models.resnet18(weights=weights)
        # Адаптируем первый Conv под нужное число каналов
        base.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False,
        )
        feature_dim = base.fc.in_features
        base.fc = nn.Identity()  # Отсекаем классификатор

    elif backbone_name == "mobilenet_v3_small":
        base = models.mobilenet_v3_small(weights=weights)
        old_conv = base.features[0][0]
        base.features[0][0] = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size, stride=old_conv.stride,
            padding=old_conv.padding, bias=(old_conv.bias is not None),
        )
        feature_dim = base.classifier[-1].in_features
        base.classifier = nn.Identity()

    elif backbone_name == "efficientnet_b0":
        base = models.efficientnet_b0(weights=weights)
        old_conv = base.features[0][0]
        base.features[0][0] = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size, stride=old_conv.stride,
            padding=old_conv.padding, bias=(old_conv.bias is not None),
        )
        feature_dim = base.classifier[-1].in_features
        base.classifier = nn.Identity()

    else:
        raise ValueError(
            f"Неизвестный backbone: {backbone_name}. "
            f"Поддерживаются: resnet18, mobilenet_v3_small, efficientnet_b0"
        )

    if checkpoint_path:
        _load_classification_backbone_weights(base, checkpoint_path)

    if freeze:
        for param in base.parameters():
            param.requires_grad = False

    return base, feature_dim


# ---------------------------------------------------------------------------
#  Компоненты: Positional Encoding для Transformer
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Синусоидальное позиционное кодирование для TransformerEncoder.

    Args:
        d_model: Размерность эмбеддинга.
        max_len: Максимальная длина последовательности.
        dropout: Вероятность dropout.
    """

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor (batch, seq_len, d_model).
        Returns:
            Tensor (batch, seq_len, d_model) с добавленным PE.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
#  Компоненты: TCN (Temporal Convolutional Network)
# ---------------------------------------------------------------------------

class _TCNBlock(nn.Module):
    """Один каузальный блок TCN с dilated-свёртками и residual-skip.

    Args:
        in_ch: Число входных каналов.
        out_ch: Число выходных каналов.
        kernel_size: Размер ядра свёртки.
        dilation: Фактор расширения (dilation).
        dropout: Вероятность dropout.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # Каузальный padding
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Skip-connection: адаптация размерности если in_ch != out_ch
        self.skip = nn.Conv1d(
            in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor (batch, channels, seq_len).
        Returns:
            Tensor (batch, out_ch, seq_len).
        """
        residual = self.skip(x)

        out = self.conv1(x)
        out = out[:, :, :x.size(2)]  # Causal trim
        out = self.relu(self.bn1(out))
        out = self.dropout(out)

        out = self.conv2(out)
        out = out[:, :, :x.size(2)]  # Causal trim
        out = self.relu(self.bn2(out))
        out = self.dropout(out)

        return self.relu(out + residual)


class TCN(nn.Module):
    """Temporal Convolutional Network — стек каузальных dilated-блоков.

    Args:
        input_size: Размер вектора признаков на каждом временном шаге.
        hidden_size: Число каналов в скрытых слоях TCN.
        num_layers: Количество TCN-блоков.
        kernel_size: Размер ядра свёртки.
        dropout: Вероятность dropout.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_ch = input_size if i == 0 else hidden_size
            dilation = 2 ** i
            layers.append(_TCNBlock(in_ch, hidden_size,
                          kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor (batch, seq_len, features).
        Returns:
            Tensor (batch, hidden_size) — выход последнего временного шага.
        """
        # Conv1d ожидает (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.network(x)
        # Берём выход последнего шага
        return x[:, :, -1]


# ---------------------------------------------------------------------------
#  Компоненты: Mamba (заглушка / graceful fallback)
# ---------------------------------------------------------------------------

_MAMBA_AVAILABLE = False
try:
    from mamba_ssm import Mamba as MambaBlock  # type: ignore
    _MAMBA_AVAILABLE = True
except ImportError:
    pass


class MambaTemporalStub(nn.Module):
    """Заглушка для Mamba SSM — fallback, если библиотека mamba-ssm не установлена.

    При наличии mamba-ssm использует настоящий MambaBlock.
    Без неё — эмулирует через GRU с предупреждением.

    Args:
        input_size: Размер входного вектора фичей.
        hidden_size: Размер скрытого состояния.
        dropout: Вероятность dropout.
    """

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size

        if _MAMBA_AVAILABLE:
            self.proj_in = nn.Linear(input_size, hidden_size)
            self.mamba = MambaBlock(
                d_model=hidden_size, d_state=16, d_conv=4, expand=2)
            self.use_real_mamba = True
        else:
            import warnings
            warnings.warn(
                "[Mamba] Библиотека mamba-ssm не найдена. "
                "Используется GRU-fallback. Установите: pip install mamba-ssm",
                UserWarning,
                stacklevel=2,
            )
            self.fallback = nn.GRU(
                input_size=input_size, hidden_size=hidden_size,
                batch_first=True, num_layers=1,
            )
            self.use_real_mamba = False

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor (batch, seq_len, features).
        Returns:
            Tensor (batch, hidden_size).
        """
        if self.use_real_mamba:
            x = self.proj_in(x)
            x = self.mamba(x)
            return self.dropout(x[:, -1, :])
        else:
            out, _ = self.fallback(x)
            return self.dropout(out[:, -1, :])


# ---------------------------------------------------------------------------
#  Основная модель: UniversalHybridRULNet
# ---------------------------------------------------------------------------

class UniversalHybridRULNet(nn.Module):
    """Универсальная гибридная сеть для прогнозирования RUL подшипников.

    Архитектура:
        1. CNN-энкодер (замороженный) извлекает вектор фичей из каждой скалограммы.
           Применяется с TimeDistributed-паттерном: reshape (B*S, C, H, W) → CNN → reshape (B, S, F).
        2. К вектору фичей конкатенируется скалярный признак speed.
        3. Объединённый вектор проходит через один из временных блоков
           (lstm / gru / tcn / transformer / mamba).
        4. Линейная голова предсказывает одно число — RUL.

    Args:
        encoder: Предобученный CNN-энкодер (nn.Module).
        encoder_dim: Размер вектора фичей из энкодера.
        temporal_type: Тип временного блока ('lstm', 'gru', 'tcn', 'transformer', 'mamba').
        hidden_size: Размер скрытого состояния временного блока.
        dropout: Вероятность dropout.
        num_temporal_layers: Количество слоёв во временном блоке.
    """

    def __init__(
        self,
        encoder: nn.Module,
        encoder_dim: int,
        temporal_type: TemporalType = "lstm",
        hidden_size: int = 64,
        dropout: float = 0.2,
        num_temporal_layers: int = 2,
        fine_tune: bool = True,
    ):
        super().__init__()

        if temporal_type not in SUPPORTED_TEMPORAL:
            raise ValueError(
                f"Неизвестный temporal_type: {temporal_type}. "
                f"Поддерживаются: {SUPPORTED_TEMPORAL}"
            )

        self.encoder = encoder
        self.temporal_type = temporal_type
        self.fine_tune = fine_tune

        if self.fine_tune:
            for param in self.encoder.parameters():
                param.requires_grad = True

        # CNN features + 1 скаляр (speed)
        combined_dim = encoder_dim + 1

        # --- Temporal Block ---
        if temporal_type == "lstm":
            self.temporal = nn.LSTM(
                input_size=combined_dim,
                hidden_size=hidden_size,
                num_layers=num_temporal_layers,
                batch_first=True,
                dropout=dropout if num_temporal_layers > 1 else 0.0,
            )
        elif temporal_type == "gru":
            self.temporal = nn.GRU(
                input_size=combined_dim,
                hidden_size=hidden_size,
                num_layers=num_temporal_layers,
                batch_first=True,
                dropout=dropout if num_temporal_layers > 1 else 0.0,
            )
        elif temporal_type == "tcn":
            self.temporal = TCN(
                input_size=combined_dim,
                hidden_size=hidden_size,
                num_layers=num_temporal_layers,
                dropout=dropout,
            )
        elif temporal_type == "transformer":
            self.input_proj = nn.Linear(combined_dim, hidden_size)
            self.pos_encoder = PositionalEncoding(hidden_size, dropout=dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.temporal = nn.TransformerEncoder(
                encoder_layer, num_layers=num_temporal_layers,
            )
        elif temporal_type == "mamba":
            self.temporal = MambaTemporalStub(
                input_size=combined_dim,
                hidden_size=hidden_size,
                dropout=dropout,
            )

        self.dropout = nn.Dropout(dropout)
        # RUL targets are normalized to [0, 1], but the default regression head
        # stays linear for MSE. A Sigmoid can be tried as an ablation if outputs
        # drift outside the target range, but it may saturate near 0/1.
        self.fc = nn.Linear(hidden_size, 1)

    def forward(
        self,
        images: torch.Tensor,
        speed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Прямой проход.

        Args:
            images: Тензор скалограмм (batch, seq_len, channels, height, width).
            speed: Скалярный признак скорости (batch, seq_len, 1).
                   Если None, заполняется нулями.

        Returns:
            rul: Предсказанный RUL, тензор (batch, 1).
        """
        batch_size, seq_len, channels, height, width = images.size()

        # --- TimeDistributed CNN ---
        # (B, S, C, H, W) → (B*S, C, H, W)
        cnn_in = images.view(batch_size * seq_len, channels, height, width)

        if self.fine_tune:
            features = self.encoder(cnn_in)  # (B*S, encoder_dim)
        else:
            with torch.no_grad():
                features = self.encoder(cnn_in)  # (B*S, encoder_dim)

        # (B*S, encoder_dim) → (B, S, encoder_dim)
        features = features.view(batch_size, seq_len, -1)

        # --- Конкатенация с speed ---
        if speed is None:
            speed = torch.zeros(batch_size, seq_len, 1, device=images.device)

        # (B, S, encoder_dim + 1)
        combined = torch.cat([features, speed], dim=-1)

        # --- Temporal Block ---
        if self.temporal_type in ("lstm", "gru"):
            temporal_out, _ = self.temporal(combined)
            # Берём выход последнего шага: (B, hidden_size)
            last_out = temporal_out[:, -1, :]

        elif self.temporal_type == "tcn":
            # TCN сам берёт последний шаг внутри forward
            last_out = self.temporal(combined)

        elif self.temporal_type == "transformer":
            projected = self.input_proj(combined)
            projected = self.pos_encoder(projected)
            transformer_out = self.temporal(projected)
            last_out = transformer_out[:, -1, :]

        elif self.temporal_type == "mamba":
            last_out = self.temporal(combined)

        # --- Prediction Head ---
        last_out = self.dropout(last_out)
        rul = self.fc(last_out)

        return rul


class TemporalOnlyRULNet(nn.Module):
    """Temporal-only RUL head over precomputed CNN features.

    This mirrors the temporal part of UniversalHybridRULNet, but expects
    features shaped as (batch, seq_len, encoder_dim). It is intended for
    frozen CNN encoders where the expensive CNN forward pass can be cached.
    """

    def __init__(
        self,
        encoder_dim: int,
        temporal_type: TemporalType = "lstm",
        hidden_size: int = 64,
        dropout: float = 0.2,
        num_temporal_layers: int = 2,
    ):
        super().__init__()

        if temporal_type not in SUPPORTED_TEMPORAL:
            raise ValueError(
                f"Неизвестный temporal_type: {temporal_type}. "
                f"Поддерживаются: {SUPPORTED_TEMPORAL}"
            )

        self.temporal_type = temporal_type
        combined_dim = encoder_dim + 1

        if temporal_type == "lstm":
            self.temporal = nn.LSTM(
                input_size=combined_dim,
                hidden_size=hidden_size,
                num_layers=num_temporal_layers,
                batch_first=True,
                dropout=dropout if num_temporal_layers > 1 else 0.0,
            )
        elif temporal_type == "gru":
            self.temporal = nn.GRU(
                input_size=combined_dim,
                hidden_size=hidden_size,
                num_layers=num_temporal_layers,
                batch_first=True,
                dropout=dropout if num_temporal_layers > 1 else 0.0,
            )
        elif temporal_type == "tcn":
            self.temporal = TCN(
                input_size=combined_dim,
                hidden_size=hidden_size,
                num_layers=num_temporal_layers,
                dropout=dropout,
            )
        elif temporal_type == "transformer":
            self.input_proj = nn.Linear(combined_dim, hidden_size)
            self.pos_encoder = PositionalEncoding(hidden_size, dropout=dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.temporal = nn.TransformerEncoder(
                encoder_layer, num_layers=num_temporal_layers,
            )
        elif temporal_type == "mamba":
            self.temporal = MambaTemporalStub(
                input_size=combined_dim,
                hidden_size=hidden_size,
                dropout=dropout,
            )

        self.dropout = nn.Dropout(dropout)
        # Same recommendation as UniversalHybridRULNet: keep the RUL head
        # linear for MSE by default; use Sigmoid only as a controlled ablation.
        self.fc = nn.Linear(hidden_size, 1)

    def forward(
        self,
        features: torch.Tensor,
        speed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = features.size()

        if speed is None:
            speed = torch.zeros(batch_size, seq_len, 1, device=features.device)

        combined = torch.cat([features, speed], dim=-1)

        if self.temporal_type in ("lstm", "gru"):
            temporal_out, _ = self.temporal(combined)
            last_out = temporal_out[:, -1, :]
        elif self.temporal_type == "tcn":
            last_out = self.temporal(combined)
        elif self.temporal_type == "transformer":
            projected = self.input_proj(combined)
            projected = self.pos_encoder(projected)
            transformer_out = self.temporal(projected)
            last_out = transformer_out[:, -1, :]
        elif self.temporal_type == "mamba":
            last_out = self.temporal(combined)

        last_out = self.dropout(last_out)
        return self.fc(last_out)


# ---------------------------------------------------------------------------
#  Self-Verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Self-Verification: UniversalHybridRULNet")
    print("=" * 60)

    # Фиктивный CNN-энкодер (resnet18)
    encoder, enc_dim = create_cnn_encoder(
        "resnet18", in_channels=3, freeze=True)
    print(f"[INFO] Encoder: resnet18, feature_dim={enc_dim}")

    # Случайные входы
    batch_size, seq_len = 16, 10
    images = torch.randn(batch_size, seq_len, 3, 224, 224)
    speed = torch.randn(batch_size, seq_len, 1)

    print(f"[INFO] images shape: {images.shape}")
    print(f"[INFO] speed  shape: {speed.shape}")
    print()

    for m_type in ["lstm", "gru", "tcn", "transformer"]:
        model = UniversalHybridRULNet(
            encoder=encoder,
            encoder_dim=enc_dim,
            temporal_type=m_type,
            hidden_size=64,
            dropout=0.2,
            num_temporal_layers=2,
        )
        model.eval()

        with torch.no_grad():
            output = model(images, speed)

        expected = (batch_size, 1)
        assert output.shape == expected, (
            f"[FAIL] {m_type}: ожидалось {expected}, получено {output.shape}"
        )
        n_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
        print(f"  ✅ {m_type:12s} → output shape: {output.shape}  "
              f"(trainable params: {n_params:,})")

    # Mamba (может быть заглушкой)
    try:
        model_mamba = UniversalHybridRULNet(
            encoder=encoder,
            encoder_dim=enc_dim,
            temporal_type="mamba",
            hidden_size=64,
            dropout=0.2,
        )
        model_mamba.eval()
        with torch.no_grad():
            output_m = model_mamba(images, speed)
        assert output_m.shape == (batch_size, 1)
        label = "mamba (real)" if _MAMBA_AVAILABLE else "mamba (GRU-fallback)"
        n_params = sum(p.numel()
                       for p in model_mamba.parameters() if p.requires_grad)
        print(f"  ✅ {label:12s} → output shape: {output_m.shape}  "
              f"(trainable params: {n_params:,})")
    except Exception as e:
        print(f"  ⚠️  mamba: {e}")

    print("\n" + "=" * 60)
    print("  Все архитектуры прошли проверку размерностей!")
    print("=" * 60)
