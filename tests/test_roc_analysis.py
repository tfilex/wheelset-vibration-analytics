import numpy as np

from src.evaluation.roc_analysis import compute_roc_ovr


def test_compute_roc_ovr_handles_missing_class():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.2, 0.8, 0.0],
            [0.1, 0.9, 0.0],
        ]
    )
    result = compute_roc_ovr(y_true, y_prob, ["a", "b", "c"])
    assert result["a"] == 1.0
    assert result["b"] == 1.0
    assert np.isnan(result["c"])
    assert result["macro_auc"] == 1.0

