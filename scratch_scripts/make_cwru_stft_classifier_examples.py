"""Create STFT/Fourier feature examples for CWRU classifier training."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy.signal


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "CWRU"
OUT_DIR = ROOT / "reports" / "figures" / "summary"
OUT_PATH = OUT_DIR / "cwru_stft_classifier_examples_four_classes.png"

WINDOW_SIZE = 1024
NPERSEG = 256
NOVERLAP = 128

EXAMPLES = [
    ("а) исправное состояние", DATA_DIR / "0_Normal" / "98.mat"),
    ("б) дефект внутреннего кольца", DATA_DIR / "1_IR_007" / "106.mat"),
    ("в) дефект тела качения", DATA_DIR / "4_Ball_007" / "119.mat"),
    ("г) дефект наружного кольца", DATA_DIR / "7_OR_007" / "131.mat"),
]


def load_de_signal(path: Path) -> np.ndarray:
    mat_data = scipy.io.loadmat(path)
    for key, value in mat_data.items():
        if not key.startswith("__") and key.endswith("_DE_time"):
            return value.flatten()
    raise ValueError(f"No *_DE_time signal found in {path}")


def representative_window(signal: np.ndarray) -> np.ndarray:
    num_windows = len(signal) // WINDOW_SIZE
    if num_windows == 0:
        raise ValueError("Signal is shorter than one classifier window")
    window_idx = max(0, int(num_windows * 0.35))
    start = window_idx * WINDOW_SIZE
    return signal[start : start + WINDOW_SIZE]


def stft_power(window: np.ndarray) -> np.ndarray:
    _, _, zxx = scipy.signal.stft(window, nperseg=NPERSEG, noverlap=NOVERLAP)
    power = np.abs(zxx) ** 2
    return np.log1p(power)


def main() -> None:
    panels: list[tuple[str, Path, np.ndarray]] = []
    for title, path in EXAMPLES:
        signal = load_de_signal(path)
        panels.append((title, path, stft_power(representative_window(signal))))

    all_values = np.concatenate([power.ravel() for _, _, power in panels])
    vmin, vmax = np.percentile(all_values, [1, 99])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), facecolor="white", constrained_layout=True)
    last_image = None
    for ax, (title, path, power) in zip(axes.ravel(), panels):
        norm_power = np.clip((power - vmin) / (vmax - vmin), 0, 1)
        last_image = ax.imshow(
            norm_power,
            origin="lower",
            aspect="auto",
            cmap="turbo",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        ax.set_title(f"{title}\nфайл {path.name}")
        ax.set_xlabel("Временной бин STFT")
        ax.set_ylabel("Частотный бин")

    if last_image is not None:
        cbar = fig.colorbar(last_image, ax=axes.ravel().tolist(), shrink=0.92, pad=0.02)
        cbar.set_label("Нормированная мощность STFT")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
