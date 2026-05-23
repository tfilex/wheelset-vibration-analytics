import numpy as np
import pandas as pd

from src.features.stat_features import batch_extract, extract_stat_features


def test_normal_signal():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 1024)
    features = extract_stat_features(x)
    assert set(features.keys()) == {
        "rms",
        "peak_factor",
        "kurtosis",
        "crest_factor",
        "variance",
    }
    assert features["rms"] > 0
    assert 2.5 < features["kurtosis"] < 3.5


def test_zero_signal():
    features = extract_stat_features(np.zeros(512))
    assert features["rms"] == 0.0
    assert np.isnan(features["peak_factor"])
    assert np.isnan(features["crest_factor"])


def test_batch_returns_dataframe():
    segments = np.random.default_rng(42).normal(size=(10, 1024))
    df = batch_extract(segments)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10

