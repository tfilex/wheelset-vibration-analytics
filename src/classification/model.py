"""
model.py — Фабрика моделей для классификации дефектов подшипников.

Реализует элемент Neural Architecture Search (NAS): сравнение 5 легковесных
архитектур из torchvision, подходящих для Edge Computing.

Поддерживаемые архитектуры:
    - resnet18
    - squeezenet1_1
    - mobilenet_v3_small
    - efficientnet_b0
    - shufflenet_v2_x1_0

Все модели адаптируются под одноканальные STFT-спектрограммы (1×129×9)
через SpectrogramAdapter, который автоматически ресайзит вход до
безопасного размера для глубоких сетей.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List

# ---- Список архитектур для NAS ----
SUPPORTED_MODELS: List[str] = [
    # ---- Классические CNN ----
    "resnet18",
    "squeezenet1_1",
    "mobilenet_v3_small",
    "efficientnet_b0",
    "shufflenet_v2_x1_0",

    # ---- Современные SOTA CNN ----
    "convnext_tiny",
    "efficientnet_v2_s",
    "regnet_y_400mf",
]

# Минимальный пространственный размер для безопасного прохода
# через глубокие архитектуры (SqueezeNet, EfficientNet и т.д.)
_MIN_SPATIAL = 64


class SpectrogramAdapter(nn.Module):
    """Обёртка над backbone для работы с STFT-спектрограммами.

    Выполняет:
        1. Билинейный ресайз входа, если spatial-размеры меньше _MIN_SPATIAL
        2. Прямой проход через backbone-классификатор
    """

    def __init__(self, backbone: nn.Module, min_spatial: int = _MIN_SPATIAL, strict_size: tuple = None):
        super().__init__()
        self.backbone = backbone
        self.min_spatial = min_spatial
        self.strict_size = strict_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.strict_size is not None:
            x = F.interpolate(x, size=self.strict_size,
                              mode="bilinear", align_corners=False)
        else:
            h, w = x.shape[2], x.shape[3]
            need_resize = h < self.min_spatial or w < self.min_spatial
            if need_resize:
                new_h = max(h, self.min_spatial)
                new_w = max(w, self.min_spatial)
                x = F.interpolate(
                    x, size=(new_h, new_w), mode="bilinear", align_corners=False
                )
        return self.backbone(x)


def _adapt_first_conv(
    old_conv: nn.Conv2d, in_channels: int
) -> nn.Conv2d:
    """Создаёт новый Conv2d с нужным in_channels, копируя остальные параметры."""
    return nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None),
    )


def get_model(
    model_name: str,
    num_classes: int = 10,
    in_channels: int = 1,
) -> nn.Module:
    """Фабрика моделей: загрузка и адаптация torchvision-архитектуры.

    Args:
        model_name: Одна из SUPPORTED_MODELS.
        num_classes: Количество выходных классов.
        in_channels: Количество входных каналов (1 для STFT).

    Returns:
        SpectrogramAdapter-обёрнутая модель.

    Raises:
        ValueError: Если model_name не поддерживается.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Неизвестная модель: {model_name}. "
            f"Поддерживаемые: {SUPPORTED_MODELS}"
        )

    strict_size = None

    # ---- ResNet-18 ----
    if model_name == "resnet18":
        backbone = models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)

    # ---- SqueezeNet 1.1 ----
    elif model_name == "squeezenet1_1":
        backbone = models.squeezenet1_1(weights=None)
        backbone.features[0] = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=2
        )
        backbone.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        backbone.num_classes = num_classes

    # ---- MobileNet V3 Small ----
    elif model_name == "mobilenet_v3_small":
        backbone = models.mobilenet_v3_small(weights=None)
        backbone.features[0][0] = _adapt_first_conv(
            backbone.features[0][0], in_channels
        )
        backbone.classifier[-1] = nn.Linear(
            backbone.classifier[-1].in_features, num_classes
        )

    # ---- EfficientNet B0 ----
    elif model_name == "efficientnet_b0":
        backbone = models.efficientnet_b0(weights=None)
        backbone.features[0][0] = _adapt_first_conv(
            backbone.features[0][0], in_channels
        )
        backbone.classifier[-1] = nn.Linear(
            backbone.classifier[-1].in_features, num_classes
        )

    # ---- ShuffleNet V2 x1.0 ----
    elif model_name == "shufflenet_v2_x1_0":
        backbone = models.shufflenet_v2_x1_0(weights=None)
        backbone.conv1[0] = nn.Conv2d(
            in_channels, 24, kernel_size=3, stride=2, padding=1, bias=False
        )
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)

    # ---- ConvNeXt Tiny ----
    elif model_name == "convnext_tiny":
        backbone = models.convnext_tiny(weights=None)
        backbone.features[0][0] = _adapt_first_conv(
            backbone.features[0][0], in_channels
        )
        backbone.classifier[2] = nn.Linear(
            backbone.classifier[2].in_features, num_classes
        )

    # ---- EfficientNet V2 S ----
    elif model_name == "efficientnet_v2_s":
        backbone = models.efficientnet_v2_s(weights=None)
        backbone.features[0][0] = _adapt_first_conv(
            backbone.features[0][0], in_channels
        )
        backbone.classifier[1] = nn.Linear(
            backbone.classifier[1].in_features, num_classes
        )

    # ---- RegNet Y 400MF ----
    elif model_name == "regnet_y_400mf":
        backbone = models.regnet_y_400mf(weights=None)
        backbone.stem[0] = _adapt_first_conv(
            backbone.stem[0], in_channels
        )
        backbone.fc = nn.Linear(
            backbone.fc.in_features, num_classes
        )

    # ---- Swin Transformer Tiny ----
    elif model_name == "swin_t":
        backbone = models.swin_t(weights=None)
        backbone.features[0][0] = _adapt_first_conv(
            backbone.features[0][0], in_channels
        )
        backbone.head = nn.Linear(
            backbone.head.in_features, num_classes
        )

    # ---- MaxViT Tiny (замена MobileViT) ----
    elif model_name == "maxvit_t":
        backbone = models.maxvit_t(weights=None)
        backbone.stem[0][0] = _adapt_first_conv(
            backbone.stem[0][0], in_channels
        )
        backbone.classifier[5] = nn.Linear(
            backbone.classifier[5].in_features, num_classes, bias=False
        )
        # MaxViT требует строгое разрешение окна внимания
        strict_size = (224, 224)

    return SpectrogramAdapter(backbone, strict_size=strict_size)


if __name__ == "__main__":
    # Тест всех архитектур на dummy-входе (STFT-спектрограмма 129×9)
    dummy = torch.randn(2, 1, 129, 9)
    print(f"Тестовый вход: {dummy.shape}\n")

    for name in SUPPORTED_MODELS:
        model = get_model(name, num_classes=10, in_channels=1)
        out = model(dummy)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {name:24s} → output {out.shape}, params: {n_params:,}")

    print("\nВсе архитектуры прошли forward pass успешно.")
