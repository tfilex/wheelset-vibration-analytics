"""Health Index helpers for RUL post-processing."""

from __future__ import annotations

import numpy as np


L_MAX_KM = 1_000_000


def compute_hi(rul_norm_series: np.ndarray, window: int = 10) -> np.ndarray:
    """Smooth normalized RUL values and return Health Index in [0, 1].

    Args:
        rul_norm_series: One-dimensional normalized RUL predictions.
        window: Moving-average window size.

    Returns:
        Smoothed Health Index series clipped to [0, 1].
    """
    values = np.asarray(rul_norm_series, dtype=float).reshape(-1)
    if values.size == 0:
        return np.array([], dtype=float)
    if window <= 1:
        return np.clip(values, 0.0, 1.0)

    window = min(int(window), values.size)
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return np.clip(smoothed, 0.0, 1.0)


def compute_slope(hi_series: np.ndarray, n: int = 20) -> float:
    """Estimate the linear trend of the latest Health Index values.

    Args:
        hi_series: Health Index series.
        n: Number of latest samples used for trend fitting.

    Returns:
        Linear regression slope per sample. Negative values indicate degradation.
    """
    values = np.asarray(hi_series, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return 0.0

    n = max(2, min(int(n), values.size))
    tail = values[-n:]
    x = np.arange(tail.size, dtype=float)
    slope, _ = np.polyfit(x, tail, deg=1)
    return float(slope)


def rul_to_km(
    hi_smooth: float,
    slope: float = 0.0,
    alpha: float = 0.2,
    l_max: float = L_MAX_KM,
) -> tuple[float, float]:
    """Convert Health Index to remaining mileage and uncertainty.

    Args:
        hi_smooth: Current smoothed Health Index.
        slope: Current Health Index trend.
        alpha: Trend correction coefficient.
        l_max: Maximum bearing mileage in kilometers.

    Returns:
        Tuple ``(rul_km, sigma_km)``.
    """
    hi = float(np.clip(hi_smooth, 0.0, 1.0))
    slope = float(slope)
    rul_base = hi * float(l_max)
    rul_adj = rul_base * (1.0 + float(alpha) * slope)
    sigma = abs(slope) * float(l_max) * 0.05
    return max(0.0, float(rul_adj)), max(0.0, float(sigma))


def get_status(hi: float) -> tuple[int, str, str]:
    """Return diagnostic status level, label and display color for HI."""
    value = float(hi)
    if value > 0.85:
        return 1, "Норма", "green"
    if value > 0.60:
        return 2, "Удовлетворительно", "orange"
    if value > 0.35:
        return 3, "Требует контроля", "red"
    return 4, "АВАРИЙНОЕ", "red"


def format_rul_display(rul_km: float, sigma_km: float) -> str:
    """Format remaining mileage for the Streamlit UI."""

    def fmt(value: float) -> str:
        return f"{int(round(float(value))):,}".replace(",", " ")

    return f"~{fmt(rul_km)} км ± {fmt(sigma_km)} км"

