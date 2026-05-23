"""Statistical vibration features in the time domain."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as scipy_kurtosis


FEATURE_COLUMNS = ("rms", "peak_factor", "kurtosis", "crest_factor", "variance")


def extract_stat_features(x: np.ndarray) -> dict[str, float]:
    """Extract basic statistical features from one vibration segment.

    ``peak_factor`` and ``crest_factor`` are intentionally identical here:
    both are defined as ``max(abs(x)) / RMS`` in the VKR specification.

    Args:
        x: One-dimensional signal segment.

    Returns:
        Dictionary with RMS, peak factor, Pearson kurtosis, crest factor and variance.
    """
    values = np.asarray(x, dtype=float).reshape(-1)
    if values.size == 0:
        return {name: float("nan") for name in FEATURE_COLUMNS}

    rms = float(np.sqrt(np.mean(values**2)))
    peak = float(np.max(np.abs(values)))
    variance = float(np.var(values))
    k_value = float(scipy_kurtosis(values, fisher=False, bias=False))

    if rms == 0.0:
        peak_factor = float("nan")
        crest_factor = float("nan")
    else:
        peak_factor = peak / rms
        crest_factor = peak / rms

    return {
        "rms": rms,
        "peak_factor": float(peak_factor),
        "kurtosis": k_value,
        "crest_factor": float(crest_factor),
        "variance": variance,
    }


def batch_extract(segments: np.ndarray) -> pd.DataFrame:
    """Extract statistical features for an array of segments with shape (N, T)."""
    values = np.asarray(segments, dtype=float)
    if values.ndim != 2:
        raise ValueError("segments must have shape (N, T)")
    return pd.DataFrame([extract_stat_features(row) for row in values], columns=FEATURE_COLUMNS)

