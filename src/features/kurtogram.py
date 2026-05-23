"""Fast-kurtogram style search for impulsive vibration bands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import filtfilt, firwin, hilbert, lfilter


@dataclass(frozen=True)
class KurtogramResult:
    """Kurtogram computation result.

    Attributes:
        kurtosis_map: Two-dimensional K4 map. Rows are levels, columns are band ids.
        centers_hz: Band center frequencies in Hz with the same shape as ``kurtosis_map``.
        bandwidths_hz: Bandwidth values in Hz with the same shape as ``kurtosis_map``.
        best: Tuple ``(level, band_index, f_center, k4_value)``.
        best_band: Tuple ``(f_center, bandwidth, k4_value)``.
    """

    kurtosis_map: np.ndarray
    centers_hz: np.ndarray
    bandwidths_hz: np.ndarray
    best: tuple[int, int, float, float]
    best_band: tuple[float, float, float]

    def __iter__(self):
        """Allow compatibility with ``kurtosis_map, best = fast_kurtogram(...)``."""
        yield self.kurtosis_map
        yield self.best


def _safe_filter_length(signal_size: int) -> int:
    max_len = max(15, min(129, signal_size // 4))
    if max_len % 2 == 0:
        max_len -= 1
    return max(15, max_len)


def _bandpass_fir(x: np.ndarray, fs: float, low_hz: float, high_hz: float) -> np.ndarray:
    nyquist = fs / 2.0
    eps = max(1.0, nyquist * 1e-5)
    low = max(float(low_hz), eps)
    high = min(float(high_hz), nyquist - eps)
    if low >= high:
        return np.zeros_like(x, dtype=float)

    numtaps = _safe_filter_length(x.size)
    taps = firwin(numtaps, [low, high], pass_zero=False, fs=fs)
    padlen = 3 * (numtaps - 1)
    if x.size > padlen:
        return filtfilt(taps, [1.0], x)
    return lfilter(taps, [1.0], x)


def _spectral_kurtosis(filtered: np.ndarray) -> float:
    analytic = hilbert(filtered)
    envelope_power = np.abs(analytic) ** 2
    second = float(np.mean(envelope_power))
    if second <= 1e-15:
        return 0.0
    fourth = float(np.mean(envelope_power**2))
    return float(fourth / (second**2) - 2.0)


def fast_kurtogram(x: np.ndarray, fs: float, n_levels: int = 8) -> KurtogramResult:
    """Compute a binary-band kurtogram and select the maximum K4 band.

    Args:
        x: One-dimensional vibration signal.
        fs: Sampling frequency in Hz.
        n_levels: Binary filter-bank depth.

    Returns:
        ``KurtogramResult`` with the K4 map and best band metadata.
    """
    signal = np.asarray(x, dtype=float).reshape(-1)
    if signal.size < 32:
        raise ValueError("x must contain at least 32 samples")
    if fs <= 0:
        raise ValueError("fs must be positive")
    if n_levels < 1:
        raise ValueError("n_levels must be positive")

    signal = signal - np.mean(signal)
    max_bands = 2 ** int(n_levels)
    k_map = np.full((n_levels, max_bands), np.nan, dtype=float)
    centers = np.full_like(k_map, np.nan)
    bandwidths = np.full_like(k_map, np.nan)

    nyquist = float(fs) / 2.0
    for level in range(1, n_levels + 1):
        band_count = 2**level
        bandwidth = nyquist / band_count
        for band_idx in range(band_count):
            low = band_idx * bandwidth
            high = (band_idx + 1) * bandwidth
            filtered = _bandpass_fir(signal, float(fs), low, high)
            k4_value = _spectral_kurtosis(filtered)
            row = level - 1
            k_map[row, band_idx] = k4_value
            centers[row, band_idx] = low + bandwidth / 2.0
            bandwidths[row, band_idx] = bandwidth

    best_flat_idx = int(np.nanargmax(k_map))
    best_row, best_col = np.unravel_index(best_flat_idx, k_map.shape)
    best_level = best_row + 1
    best_center = float(centers[best_row, best_col])
    best_bandwidth = float(bandwidths[best_row, best_col])
    best_k4 = float(k_map[best_row, best_col])
    best = (best_level, int(best_col), best_center, best_k4)
    best_band = (best_center, best_bandwidth, best_k4)

    return KurtogramResult(
        kurtosis_map=k_map,
        centers_hz=centers,
        bandwidths_hz=bandwidths,
        best=best,
        best_band=best_band,
    )


def plot_kurtogram(kurtogram_result: KurtogramResult, save_path: str | Path) -> Path:
    """Save kurtogram heatmap to PNG.

    Args:
        kurtogram_result: Result returned by ``fast_kurtogram``.
        save_path: Output image path.

    Returns:
        Path to the saved PNG.
    """
    import matplotlib.pyplot as plt

    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    data = np.ma.masked_invalid(kurtogram_result.kurtosis_map)
    image = ax.imshow(data, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title("Fast Kurtogram: spectral kurtosis K4")
    ax.set_xlabel("Band index")
    ax.set_ylabel("Level")
    ax.set_yticks(np.arange(kurtogram_result.kurtosis_map.shape[0]))
    ax.set_yticklabels(np.arange(1, kurtogram_result.kurtosis_map.shape[0] + 1))
    fig.colorbar(image, ax=ax, label="K4")

    level, band_idx, center_hz, k4_value = kurtogram_result.best
    ax.scatter([band_idx], [level - 1], color="red", marker="x", s=80)
    ax.annotate(
        f"{center_hz:.0f} Hz, K4={k4_value:.2f}",
        xy=(band_idx, level - 1),
        xytext=(8, 8),
        textcoords="offset points",
        color="red",
    )

    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output

