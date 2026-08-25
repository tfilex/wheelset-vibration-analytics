from pathlib import Path

import pytest

from src.prediction import demo_inference as demo


pytestmark = pytest.mark.slow


def _path_from_option(option: demo.ModelOption) -> Path:
    return Path(option["path"]).expanduser()


def test_demo_catalog_discovers_existing_checkpoints():
    classification_options = demo.discover_classification_models("demo")
    rul_options = demo.discover_rul_models("demo")

    assert classification_options, "Demo catalog must expose a CWRU classifier"
    assert rul_options, "Demo catalog must expose a pinned RUL model"

    for option in [*classification_options, *rul_options]:
        assert _path_from_option(option).exists()


def test_demo_rul_catalog_prefers_pinned_checkpoint():
    if not demo.DEFAULT_RUL_CHECKPOINT.exists():
        pytest.skip(f"Pinned RUL checkpoint is absent: {demo.DEFAULT_RUL_CHECKPOINT}")

    rul_options = demo.discover_rul_models("demo")

    assert rul_options
    assert _path_from_option(rul_options[0]).resolve() == demo.DEFAULT_RUL_CHECKPOINT.resolve()
    assert "TRANSFORMER_IMPROVED" in rul_options[0]["label"]
    assert "MSE=0.0117" in rul_options[0]["label"]


def test_demo_classification_smoke_inference():
    checkpoint_path = demo.get_cwru_checkpoint_path()
    sample_path = demo.CWRU_SAMPLE_FILES["Норма"]
    if not checkpoint_path.exists() or not sample_path.exists():
        pytest.skip("CWRU demo checkpoint or sample data is absent")

    result = demo.classify_signal(
        "Норма",
        checkpoint_path=str(checkpoint_path),
        checkpoint_fingerprint=demo.get_checkpoint_fingerprint(checkpoint_path),
    )

    assert result["checkpoint_path"] == str(checkpoint_path)
    assert result["source_file"] == str(sample_path)
    assert result["signal"].shape[0] == 1024
    assert result["attribution"].ndim == 2
    assert 0.0 <= result["confidence"] <= 1.0


def test_demo_rul_smoke_inference():
    checkpoint_path = demo.get_rul_checkpoint_path()
    bearing_dir = demo.XJTU_BEARING_DIRS["Bearing1_3"]
    if not checkpoint_path.exists() or not bearing_dir.exists():
        pytest.skip("RUL demo checkpoint or XJTU-SY sample data is absent")

    history, metadata = demo.predict_rul_series(
        "Bearing1_3",
        checkpoint_path=str(checkpoint_path),
        checkpoint_fingerprint=demo.get_checkpoint_fingerprint(checkpoint_path),
        output_steps=8,
        anchor_points=2,
    )

    assert list(history.columns) == ["step", "true_rul", "pred_rul"]
    assert len(history) == 8
    assert history["pred_rul"].between(0.0, 1.0).all()
    assert metadata["checkpoint_path"] == str(checkpoint_path)
    assert metadata["anchor_points"] == 2
