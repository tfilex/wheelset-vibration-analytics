import pytest
import torch
import torch.nn as nn

from src.classification import model as classification_model


class ShapeBackbone(nn.Module):
    def forward(self, x):
        return x


def test_spectrogram_adapter_resizes_small_inputs():
    adapter = classification_model.SpectrogramAdapter(ShapeBackbone(), min_spatial=64)
    out = adapter(torch.randn(1, 1, 16, 9))
    assert out.shape == (1, 1, 64, 64)


def test_spectrogram_adapter_keeps_large_inputs_and_supports_strict_size():
    adapter = classification_model.SpectrogramAdapter(ShapeBackbone(), min_spatial=64)
    large = adapter(torch.randn(1, 1, 80, 72))
    assert large.shape == (1, 1, 80, 72)

    strict = classification_model.SpectrogramAdapter(
        ShapeBackbone(), min_spatial=64, strict_size=(32, 48)
    )
    resized = strict(torch.randn(1, 1, 80, 72))
    assert resized.shape == (1, 1, 32, 48)


@pytest.mark.parametrize(
    "model_name",
    [name for name in classification_model.SUPPORTED_MODELS if name != "resnet18"],
)
def test_get_model_builds_all_declared_classifiers(model_name):
    model = classification_model.get_model(model_name, num_classes=3, in_channels=1)

    assert isinstance(model, classification_model.SpectrogramAdapter)
    assert model.backbone is not None
