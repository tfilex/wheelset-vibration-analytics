import warnings

import pytest
import torch
import torch.nn as nn

from src.prediction import model as rul_model


class MeanEncoder(nn.Module):
    def __init__(self, out_features: int = 4):
        super().__init__()
        self.proj = nn.Linear(1, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        return self.proj(pooled)


def test_adapt_input_conv_weight_channel_cases():
    one_channel = torch.ones(4, 1, 3, 3)
    adapted = rul_model._adapt_input_conv_weight(one_channel, torch.Size([4, 2, 3, 3]))
    assert adapted.shape == (4, 2, 3, 3)
    assert torch.allclose(adapted[:, 0], torch.full((4, 3, 3), 0.5))

    three_channel = torch.stack(
        [torch.full((4, 3, 3), value) for value in (1.0, 2.0, 3.0)], dim=1
    )
    collapsed = rul_model._adapt_input_conv_weight(three_channel, torch.Size([4, 1, 3, 3]))
    assert collapsed.shape == (4, 1, 3, 3)
    assert torch.allclose(collapsed[:, 0], torch.full((4, 3, 3), 2.0))

    expanded = rul_model._adapt_input_conv_weight(three_channel, torch.Size([4, 5, 3, 3]))
    assert expanded.shape == (4, 5, 3, 3)
    assert torch.allclose(expanded[:, 0], torch.full((4, 3, 3), 1.0))
    assert torch.allclose(expanded[:, 3], torch.full((4, 3, 3), 2.0))

    mismatched = torch.ones(5, 1, 3, 3)
    assert rul_model._adapt_input_conv_weight(mismatched, torch.Size([4, 2, 3, 3])) is mismatched


@pytest.mark.parametrize(
    ("backbone", "feature_dim"),
    [
        ("resnet18", 512),
        ("mobilenet_v3_small", 1024),
        ("efficientnet_b0", 1280),
    ],
)
def test_create_cnn_encoder_supported_backbones(backbone, feature_dim):
    encoder, dim = rul_model.create_cnn_encoder(
        backbone_name=backbone,
        in_channels=2,
        pretrained=False,
        freeze=True,
    )

    assert dim == feature_dim
    assert all(not parameter.requires_grad for parameter in encoder.parameters())


def test_create_cnn_encoder_loads_adapted_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "classifier.pth"
    torch.save({"state_dict": {"conv1.weight": torch.ones(64, 1, 7, 7)}}, checkpoint_path)

    encoder, dim = rul_model.create_cnn_encoder(
        "resnet18",
        in_channels=2,
        freeze=False,
        checkpoint_path=str(checkpoint_path),
    )

    assert dim == 512
    assert encoder.conv1.weight.shape == (64, 2, 7, 7)
    assert torch.allclose(encoder.conv1.weight[:, 0], torch.full((64, 7, 7), 0.5))


def test_create_cnn_encoder_checkpoint_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        rul_model.create_cnn_encoder("resnet18", checkpoint_path=str(tmp_path / "missing.pth"))

    incompatible = tmp_path / "bad.pth"
    torch.save({"state_dict": {"not_encoder.weight": torch.ones(1)}}, incompatible)
    with pytest.raises(RuntimeError):
        rul_model.create_cnn_encoder("resnet18", checkpoint_path=str(incompatible))

    with pytest.raises(ValueError):
        rul_model.create_cnn_encoder("unknown_backbone")


def test_positional_encoding_preserves_shape_for_odd_dimensions():
    encoding = rul_model.PositionalEncoding(d_model=5, max_len=8, dropout=0.0)
    x = torch.zeros(2, 4, 5)

    out = encoding(x)

    assert out.shape == x.shape
    assert not torch.allclose(out, x)


@pytest.mark.parametrize("temporal_type", rul_model.SUPPORTED_TEMPORAL)
def test_temporal_only_all_temporal_blocks(temporal_type):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = rul_model.TemporalOnlyRULNet(
            encoder_dim=4,
            temporal_type=temporal_type,
            hidden_size=8,
            dropout=0.0,
            num_temporal_layers=1,
        )
    model.eval()
    features = torch.randn(2, 3, 4)
    speed = torch.randn(2, 3, 1)

    with torch.no_grad():
        out = model(features, speed)

    assert out.shape == (2, 1)


@pytest.mark.parametrize("temporal_type", rul_model.SUPPORTED_TEMPORAL)
def test_universal_hybrid_all_temporal_blocks(temporal_type):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = rul_model.UniversalHybridRULNet(
            encoder=MeanEncoder(out_features=4),
            encoder_dim=4,
            temporal_type=temporal_type,
            hidden_size=8,
            dropout=0.0,
            num_temporal_layers=1,
            fine_tune=False,
        )
    model.eval()
    images = torch.randn(2, 3, 2, 8, 8)

    with torch.no_grad():
        out = model(images)

    assert out.shape == (2, 1)


def test_temporal_type_validation():
    with pytest.raises(ValueError):
        rul_model.TemporalOnlyRULNet(encoder_dim=4, temporal_type="bad")
    with pytest.raises(ValueError):
        rul_model.UniversalHybridRULNet(
            encoder=MeanEncoder(),
            encoder_dim=4,
            temporal_type="bad",
        )
