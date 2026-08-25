"""Create full-record and classifier-window STFT figures for CWRU classes."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy.signal
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "CWRU"
OUT_DIR = ROOT / "reports" / "figures" / "summary"
FULL_OUT = OUT_DIR / "cwru_stft_classifier_full_record_four_classes.png"
WINDOW_OUT = OUT_DIR / "cwru_stft_classifier_window_four_classes.png"

FS_HZ = 12_000
WINDOW_SIZE = 1024
NPERSEG = 256
NOVERLAP = 128
RMS_HOP = WINDOW_SIZE

EXAMPLES = [
    ("а) Исправное состояние", "Normal", DATA_DIR / "0_Normal" / "98.mat"),
    ("б) Дефект внутреннего кольца", "IR_007", DATA_DIR / "1_IR_007" / "106.mat"),
    ("в) Дефект тела качения", "Ball_007", DATA_DIR / "4_Ball_007" / "119.mat"),
    ("г) Дефект наружного кольца", "OR_007", DATA_DIR / "7_OR_007" / "131.mat"),
]

COLORS = ["#2ca25f", "#1f77b4", "#ff9900", "#d62728"]


def load_de_signal(path: Path) -> np.ndarray:
    mat_data = scipy.io.loadmat(path)
    for key, value in mat_data.items():
        if not key.startswith("__") and key.endswith("_DE_time"):
            return value.flatten().astype(np.float64)
    raise ValueError(f"No *_DE_time signal found in {path}")


def normalize_curve(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo, hi = np.min(x), np.max(x)
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def zscore(x: np.ndarray) -> np.ndarray:
    x = x - np.mean(x)
    std = np.std(x)
    return x / std if std > 0 else x


def choose_window(signal: np.ndarray) -> tuple[int, np.ndarray]:
    num_windows = len(signal) // WINDOW_SIZE
    if num_windows == 0:
        raise ValueError("Signal is shorter than one classifier window")
    idx = max(0, min(num_windows - 1, num_windows // 2))
    start = idx * WINDOW_SIZE
    return idx, signal[start : start + WINDOW_SIZE]


def stft_db(signal: np.ndarray, exact_classifier_window: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if exact_classifier_window:
        freqs, times, zxx = scipy.signal.stft(signal, nperseg=NPERSEG, noverlap=NOVERLAP)
    else:
        freqs, times, zxx = scipy.signal.stft(
            zscore(signal),
            fs=FS_HZ,
            nperseg=NPERSEG,
            noverlap=NOVERLAP,
            boundary=None,
            padded=False,
        )
    power_db = 10 * np.log10(np.abs(zxx) ** 2 + 1e-12)
    return freqs, times, power_db


def robust_limits(images: list[np.ndarray]) -> tuple[float, float]:
    all_values = np.concatenate([img.ravel() for img in images])
    return tuple(np.percentile(all_values, [3, 99.5]))


def rms_curve(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = (len(signal) - WINDOW_SIZE) // RMS_HOP + 1
    rms = np.empty(n)
    for i in range(n):
        start = i * RMS_HOP
        window = signal[start : start + WINDOW_SIZE]
        rms[i] = np.sqrt(np.mean(window ** 2))
    times = np.arange(n) * RMS_HOP / FS_HZ
    return times, normalize_curve(rms)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
        }
    )


def save_full_record_figure(signals: list[np.ndarray], chosen_indices: list[int]) -> None:
    stfts = [stft_db(signal, exact_classifier_window=False) for signal in signals]
    vmin, vmax = robust_limits([item[2] for item in stfts])

    fig = plt.figure(figsize=(15.2, 7.6), facecolor="white")
    gs = GridSpec(2, 4, height_ratios=[2.35, 1.05], hspace=0.35, wspace=0.25, figure=fig)

    last_image = None
    for i, ((panel_title, _, _), (freqs, times, power_db)) in enumerate(zip(EXAMPLES, stfts)):
        ax = fig.add_subplot(gs[0, i])
        image = np.clip((power_db - vmin) / (vmax - vmin), 0, 1)
        last_image = ax.imshow(
            image,
            origin="lower",
            aspect="auto",
            cmap="turbo",
            vmin=0,
            vmax=1,
            extent=[times[0], times[-1], freqs[0] / 1000, freqs[-1] / 1000],
            interpolation="nearest",
        )
        ax.set_title(panel_title)
        ax.set_xlabel("Время, с")
        ax.set_ylabel("Частота, кГц")

    if last_image is not None:
        cax = fig.add_axes([0.915, 0.47, 0.012, 0.37])
        cbar = fig.colorbar(last_image, cax=cax)
        cbar.set_label("Нормированная мощность STFT")

    ax_curve = fig.add_subplot(gs[1, :])
    for (title, short_name, _), signal, window_idx, color in zip(EXAMPLES, signals, chosen_indices, COLORS):
        times, rms = rms_curve(signal)
        label = title.split(") ", 1)[1]
        ax_curve.plot(times, rms, color=color, linewidth=1.8, label=label)
        marker_time = window_idx * WINDOW_SIZE / FS_HZ
        marker_idx = int(np.argmin(np.abs(times - marker_time)))
        ax_curve.scatter(times[marker_idx], rms[marker_idx], s=46, color=color, edgecolor="black", zorder=5)

    ax_curve.set_title("Нормированная RMS-энергия исходных записей")
    ax_curve.set_xlabel("Время, с")
    ax_curve.set_ylabel("Нормированная энергия")
    ax_curve.grid(True, alpha=0.28)
    ax_curve.set_ylim(-0.05, 1.05)
    ax_curve.legend(loc="upper right", ncol=2, frameon=True)

    fig.subplots_adjust(left=0.055, right=0.895, top=0.94, bottom=0.08)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FULL_OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_window_figure(windows: list[np.ndarray]) -> None:
    stfts = [stft_db(window, exact_classifier_window=True) for window in windows]
    vmin, vmax = robust_limits([item[2] for item in stfts])

    fig = plt.figure(figsize=(15.2, 7.6), facecolor="white")
    gs = GridSpec(2, 4, height_ratios=[2.35, 1.05], hspace=0.35, wspace=0.25, figure=fig)

    last_image = None
    for i, ((panel_title, _, _), (freqs, times, power_db)) in enumerate(zip(EXAMPLES, stfts)):
        ax = fig.add_subplot(gs[0, i])
        image = np.clip((power_db - vmin) / (vmax - vmin), 0, 1)
        last_image = ax.imshow(
            image,
            origin="lower",
            aspect="auto",
            cmap="turbo",
            vmin=0,
            vmax=1,
            extent=[0, WINDOW_SIZE, 0, len(freqs) - 1],
            interpolation="bilinear",
        )
        ax.set_title(panel_title)
        ax.set_xlabel("Отсчёт окна")
        ax.set_ylabel("Частотный бин")

    if last_image is not None:
        cax = fig.add_axes([0.915, 0.47, 0.012, 0.37])
        cbar = fig.colorbar(last_image, cax=cax)
        cbar.set_label("Нормированная мощность STFT")

    ax_curve = fig.add_subplot(gs[1, :])
    x = np.arange(WINDOW_SIZE)
    for (title, _, _), window, color in zip(EXAMPLES, windows, COLORS):
        label = title.split(") ", 1)[1]
        ax_curve.plot(x, normalize_curve(window), color=color, linewidth=1.25, label=label)

    ax_curve.set_title("Нормированные сигналы выбранных окон")
    ax_curve.set_xlabel("Отсчёт окна")
    ax_curve.set_ylabel("Нормированная амплитуда")
    ax_curve.grid(True, alpha=0.28)
    ax_curve.set_ylim(-0.05, 1.05)
    ax_curve.legend(loc="upper right", ncol=2, frameon=True)

    fig.subplots_adjust(left=0.055, right=0.895, top=0.94, bottom=0.08)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(WINDOW_OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    setup_style()
    signals = [load_de_signal(path) for _, _, path in EXAMPLES]
    chosen: list[int] = []
    windows: list[np.ndarray] = []
    for signal in signals:
        idx, window = choose_window(signal)
        chosen.append(idx)
        windows.append(window)

    save_full_record_figure(signals, chosen)
    save_window_figure(windows)
    print(f"Saved: {FULL_OUT}")
    print(f"Saved: {WINDOW_OUT}")


if __name__ == "__main__":
    main()
