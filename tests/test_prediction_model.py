import torch
import torch.nn as nn

from src.prediction.model import TCN, TemporalOnlyRULNet, UniversalHybridRULNet


class TinyEncoder(nn.Module):
    def forward(self, x):
        return torch.ones(x.shape[0], 8, device=x.device)


def test_tcn_output_shape():
    model = TCN(input_size=9, hidden_size=16, num_layers=2)
    x = torch.randn(4, 5, 9)

    out = model(x)

    assert out.shape == (4, 16)


def test_temporal_only_rul_output_shape():
    model = TemporalOnlyRULNet(
        encoder_dim=8,
        temporal_type="gru",
        hidden_size=16,
        num_temporal_layers=1,
    )
    features = torch.randn(4, 5, 8)

    out = model(features)

    assert out.shape == (4, 1)


def test_universal_hybrid_rul_output_shape():
    encoder = TinyEncoder()
    model = UniversalHybridRULNet(
        encoder=encoder,
        encoder_dim=8,
        temporal_type="lstm",
        hidden_size=16,
        num_temporal_layers=1,
    )
    images = torch.randn(2, 5, 2, 16, 16)

    out = model(images)

    assert out.shape == (2, 1)
