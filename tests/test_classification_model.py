import pytest
import torch

from src.classification.model import get_model


def test_resnet18_classifier_forward_shape():
    model = get_model("resnet18", num_classes=10, in_channels=1)
    model.eval()
    x = torch.randn(2, 1, 129, 9)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 10)


def test_unknown_classifier_model_raises():
    with pytest.raises(ValueError):
        get_model("unknown_model")
