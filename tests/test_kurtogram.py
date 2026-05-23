import numpy as np

from src.features.kurtogram import fast_kurtogram


def test_kurtogram_healthy():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 8192)
    result = fast_kurtogram(x, fs=12_000, n_levels=5)
    _, _, _, k4_value = result.best
    assert k4_value < 5.0


def test_kurtogram_impulsive():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 0.1, 8192)
    t = np.arange(8192) / 12_000
    impulse_freq = 120
    mask = np.abs(t * impulse_freq - np.round(t * impulse_freq)) < 0.001
    x[mask] += rng.exponential(5.0, size=int(mask.sum()))
    result = fast_kurtogram(x, fs=12_000, n_levels=5)
    _, _, _, k4_value = result.best
    assert k4_value > 2.0

