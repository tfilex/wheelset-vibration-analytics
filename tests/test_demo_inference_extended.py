import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.prediction import demo_inference as demo


@pytest.mark.parametrize(
    "temporal_type",
    ["bilstm", "lstm_attn", "bigru", "gru_attn", "transformer_improved", "lstm"],
)
def test_build_advanced_rul_heads_forward(temporal_type):
    head = demo._build_advanced_rul_head(
        temporal_type=temporal_type,
        encoder_dim=6,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
    )
    head.eval()

    with torch.no_grad():
        out = head(torch.randn(2, 4, 6))

    assert out.shape == (2, 1)


def test_build_advanced_rul_head_rejects_unknown_type():
    with pytest.raises(ValueError):
        demo._build_advanced_rul_head("unknown", 6, 8, 1, 0.0)


def test_encoder_head_rul_net_freezes_encoder_and_runs_forward():
    encoder = nn.Sequential(nn.Flatten(), nn.Linear(2 * 3 * 3, 6))
    head = demo.BiGRUHead(input_size=6, hidden_size=8, num_layers=1, dropout=0.0)
    model = demo.EncoderHeadRULNet(encoder=encoder, head=head, fine_tune=False)

    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    out = model(torch.randn(2, 4, 2, 3, 3))
    assert out.shape == (2, 1)


def test_demo_rul_dataset_processes_sorted_csv_sequences(tmp_path):
    for name, length in [("10.csv", 3), ("2.csv", 6), ("20.csv", 5)]:
        pd.DataFrame(
            {
                "horizontal": np.linspace(0.0, 1.0, length),
                "vertical": np.linspace(1.0, 0.0, length),
            }
        ).to_csv(tmp_path / name, index=False)

    dataset = demo.DemoRULDataset(
        data_dir=str(tmp_path),
        seq_length=2,
        window_size=5,
        cwt_scales=4,
        rul_clip=0.8,
        normalize_scalograms=True,
    )

    assert len(dataset) == 2
    x, y = dataset[0]
    assert x.shape == (2, 2, 4, 5)
    assert y.shape == (1,)
    assert y.item() == pytest.approx(0.8)


def test_demo_rul_dataset_requires_enough_files(tmp_path):
    pd.DataFrame({"h": [1.0], "v": [0.0]}).to_csv(tmp_path / "1.csv", index=False)

    with pytest.raises(ValueError):
        demo.DemoRULDataset(
            data_dir=str(tmp_path),
            seq_length=2,
            window_size=5,
            cwt_scales=4,
            rul_clip=1.0,
            normalize_scalograms=False,
        )


def test_checkpoint_resolution_and_metadata_helpers(tmp_path, monkeypatch):
    preferred = tmp_path / "preferred.pth"
    fallback = tmp_path / "fallback.pth"
    legacy = tmp_path / "legacy.pth"
    for path in (preferred, fallback, legacy):
        path.write_bytes(b"placeholder")

    assert demo._resolve_checkpoint("TEST_CHECKPOINT", preferred, fallback, legacy) == preferred
    monkeypatch.setenv("TEST_CHECKPOINT", str(legacy))
    assert demo._resolve_checkpoint("TEST_CHECKPOINT", preferred, fallback, legacy) == legacy
    monkeypatch.delenv("TEST_CHECKPOINT")
    preferred.unlink()
    assert demo._resolve_checkpoint("TEST_CHECKPOINT", preferred, fallback, legacy) == legacy
    legacy.unlink()
    assert demo._resolve_checkpoint("TEST_CHECKPOINT", preferred, fallback, legacy) == fallback

    relative = demo._as_project_path("models/example.pth")
    assert relative == demo.PROJECT_ROOT / "models/example.pth"
    assert demo._as_project_path(fallback) == fallback
    assert demo.get_checkpoint_fingerprint(fallback).endswith(f":{fallback.stat().st_size}")


def test_checkpoint_candidate_and_load_helpers(tmp_path, monkeypatch):
    first = tmp_path / "a.pth"
    second = tmp_path / "b.pth"
    ignored = tmp_path / "notes.txt"
    first.write_bytes(b"not a torch checkpoint")
    second.write_bytes(b"not a torch checkpoint either")
    ignored.write_text("skip", encoding="utf-8")

    candidates = demo._checkpoint_candidates(first, first, tmp_path, tmp_path / "missing.pth")
    assert candidates.count(first) == 1
    assert second in candidates
    assert ignored not in candidates

    monkeypatch.setenv("ACTIVE_TEST_CHECKPOINT", str(second))
    assert demo._active_checkpoint_candidates(
        "ACTIVE_TEST_CHECKPOINT", first, tmp_path / "fallback.pth"
    ) == [second]

    assert demo._load_checkpoint_metadata(first) is None
    raw_tensor_path = tmp_path / "tensor.pth"
    torch.save(torch.ones(1), raw_tensor_path)
    assert demo._load_checkpoint_metadata(raw_tensor_path) is None

    checkpoint_path = tmp_path / "checkpoint_ws2048.pth"
    checkpoint = {"state_dict": {"temporal.weight_ih_l0": torch.ones(4, 4)}, "temporal_type": "gru"}
    torch.save(checkpoint, checkpoint_path)
    loaded = demo._load_checkpoint_metadata(checkpoint_path)
    assert loaded["temporal_type"] == "gru"
    assert demo._infer_window_size(checkpoint_path, {}) == 2048


def test_rul_checkpoint_parameter_inference_helpers():
    assert demo._is_advanced_rul_checkpoint({"temporal_type": "bilstm"}) is True
    assert demo._is_advanced_rul_checkpoint({"ckpt_suffix": "x"}) is True
    assert demo._is_advanced_rul_checkpoint({"temporal_type": "lstm"}) is False
    assert demo._rul_data_pipeline({"temporal_type": "transformer_improved"}) == "v3_zscore_start_rul"
    assert demo._rul_data_pipeline({"temporal_type": "lstm"}) == "legacy"

    cases = [
        ("lstm", {"temporal.weight_ih_l0": torch.ones(4, 4), "temporal.weight_ih_l1": torch.ones(4, 4)}, 2),
        ("bilstm", {"head.lstm.weight_ih_l0_reverse": torch.ones(4, 4)}, 1),
        ("bigru", {"head.gru.weight_ih_l2": torch.ones(4, 4)}, 3),
        ("transformer_improved", {"head.transformer.layers.3.self_attn.in_proj_weight": torch.ones(4, 4)}, 4),
        ("tcn", {"temporal.network.2.conv1.weight": torch.ones(4, 4, 3)}, 3),
    ]
    for temporal_type, state_dict, expected in cases:
        assert demo._infer_temporal_layers(state_dict, temporal_type) == expected

    params = demo._rul_params_from_checkpoint(
        {
            "state_dict": {"temporal.weight_ih_l0": torch.ones(4, 4)},
            "hyperparams": {
                "temporal_type": "gru",
                "hidden_size": 12,
                "dropout": 0.1,
                "seq_length": 7,
            },
        }
    )
    assert params == {
        "temporal_type": "gru",
        "hidden_size": 12,
        "dropout": 0.1,
        "seq_length": 7,
        "num_layers": 1,
    }


def test_interpolate_series_clips_values():
    interpolated = demo._interpolate_series(
        np.array([1.0, 3.0]), np.array([-1.0, 2.0]), output_steps=3
    )
    assert np.all((0.0 <= interpolated) & (interpolated <= 1.0))
    assert interpolated.tolist() == [0.0, 0.5, 1.0]
