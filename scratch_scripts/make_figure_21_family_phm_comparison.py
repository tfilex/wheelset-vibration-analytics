"""Generate Figure 21: normalized PHM Score comparison for best model families.

The training code stores raw PHM penalty where lower is better. This figure uses
the normalized chapter-scale PHM Score where higher is better, matching the VKR
text around the ensemble result.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "reports" / "figures" / "summary" / "figure_21_family_phm_score_comparison.png"

# Normalized PHM Score scale used in the chapter text: higher is better.
MODELS = [
    {
        "family": "Базовая",
        "model": "ResNet+LSTM",
        "phm_score": 0.604,
        "color": "#90A4AE",
    },
    {
        "family": "v3_rnn",
        "model": "ImprovedTransformer\nFrozen",
        "phm_score": 0.712,
        "color": "#2196F3",
    },
    {
        "family": "v4_tcn",
        "model": "BiTCN",
        "phm_score": 0.654,
        "color": "#FF9800",
    },
    {
        "family": "v5_odd",
        "model": "Conformer",
        "phm_score": 0.689,
        "color": "#4CAF50",
    },
    {
        "family": "Ансамбль",
        "model": "0,52 IT + 0,28 Conformer\n+ 0,20 BiTCN",
        "phm_score": 0.728,
        "color": "#AD1457",
    },
]


def main() -> None:
    labels = [f"{item['family']}\n{item['model']}" for item in MODELS]
    values = [item["phm_score"] for item in MODELS]
    colors = [item["color"] for item in MODELS]
    x = np.arange(len(MODELS))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 18,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    fig, ax = plt.subplots(figsize=(14, 7.5), facecolor="white")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F5F5F5")
    ax.grid(axis="y", color="white", linewidth=1.4, zorder=0)

    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=1.2, width=0.62, zorder=3)

    best_idx = int(np.argmax(values))
    bars[best_idx].set_edgecolor("#111827")
    bars[best_idx].set_linewidth(2.0)

    for i, (bar, value) in enumerate(zip(bars, values)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value:.3f}".replace(".", ","),
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold" if i == best_idx else "normal",
            color="#1F2933",
        )

    ax.axhline(0.712, color="#1565C0", linewidth=1.5, linestyle="--", alpha=0.85)
    ax.text(
        len(MODELS) - 0.55,
        0.712 + 0.006,
        "",
        ha="right",
        va="bottom",
        fontsize=11,
        color="#1565C0",
    )

    ax.set_title("Сравнение PHM Score лучших моделей по семействам", fontweight="bold", pad=14)
    ax.set_ylabel("Нормированный PHM Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.55, 0.76)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#B0BEC5")
    ax.spines["bottom"].set_color("#B0BEC5")

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
