import numpy as np
import pytest

from src.models import hmm_baseline
from src.models.hmm_baseline import HMMRULPredictor


pytestmark = pytest.mark.skipif(
    hmm_baseline.GaussianHMM is None,
    reason="hmmlearn is not installed",
)


def _sequences():
    first = np.column_stack(
        [np.linspace(0.0, 1.0, 12), np.linspace(1.0, 0.0, 12)]
    )
    second = np.column_stack(
        [np.linspace(0.2, 1.2, 10), np.linspace(0.9, 0.1, 10)]
    )
    return [first, second]


def test_hmm_fit_predict_evaluate_and_select_components():
    sequences = _sequences()
    predictor = HMMRULPredictor(
        n_components=2,
        covariance_type="diag",
        n_iter=20,
        random_state=0,
    )

    assert predictor.fit(sequences) is predictor
    prediction = predictor.predict_rul(sequences[0])

    assert prediction.shape == (len(sequences[0]),)
    assert np.all((0.0 <= prediction) & (prediction <= 1.0))
    assert predictor.state_rul_.shape == (2,)

    true_ruls = [predictor._default_true_rul(len(seq)) for seq in sequences]
    metrics = predictor.evaluate(sequences, true_ruls)
    assert set(metrics) == {"rmse", "mae", "r2"}
    assert metrics["rmse"] >= 0.0
    assert metrics["mae"] >= 0.0

    best_k, scores = predictor.select_n_components(sequences, n_range=(2, 3))
    assert best_k in scores
    assert set(scores) == {2, 3}


def test_hmm_validation_errors():
    predictor = HMMRULPredictor(n_components=2, covariance_type="diag", n_iter=2)
    sequences = _sequences()

    with pytest.raises(RuntimeError):
        predictor.predict_rul(sequences[0])
    with pytest.raises(ValueError):
        predictor.fit([])
    with pytest.raises(ValueError):
        predictor.fit([np.ones((3, 2)), np.ones((3, 3))])
    with pytest.raises(ValueError):
        predictor.fit(sequences, lengths=[1, 2])

    predictor.fit(sequences)
    with pytest.raises(ValueError):
        predictor.evaluate([sequences[0]], [np.ones(3)])


def test_hmm_helpers_cover_edge_cases():
    predictor = HMMRULPredictor(n_components=3, covariance_type="full")
    assert np.array_equal(predictor._default_true_rul(1), np.array([0.0]))
    assert predictor._parameter_count(3, 2) > 0
    assert HMMRULPredictor(covariance_type="diag")._parameter_count(3, 2) > 0
    assert HMMRULPredictor(covariance_type="spherical")._parameter_count(3, 2) > 0
