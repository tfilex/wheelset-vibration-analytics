import numpy as np
import pytest

from src.evaluation.roc_analysis import compute_roc_ovr, plot_roc_ovr


def test_compute_roc_ovr_validates_probability_shape():
    with pytest.raises(ValueError):
        compute_roc_ovr(np.array([0, 1]), np.array([0.2, 0.8]), ["a", "b"])
    with pytest.raises(ValueError):
        compute_roc_ovr(np.array([0, 1]), np.ones((2, 3)), ["a", "b"])


def test_plot_roc_ovr_writes_figure(tmp_path):
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_prob = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.0, 0.1, 0.9],
        ]
    )
    output = tmp_path / "roc" / "ovr.png"

    auc_values = plot_roc_ovr(y_true, y_prob, ["a", "b", "c"], output)

    assert output.exists()
    assert output.stat().st_size > 0
    assert auc_values["macro_auc"] == pytest.approx(1.0)
