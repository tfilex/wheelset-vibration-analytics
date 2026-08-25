"""Create a report-ready CWRU ResNet-18 grouped confusion matrix figure."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
CM_CSV = ROOT / "reports" / "tables_for_vkr" / "cwru_resnet18_confusion_matrix_file_grouped.csv"
METRICS_CSV = ROOT / "reports" / "tables_for_vkr" / "cwru_resnet18_file_grouped_metrics.csv"
OUT_DIR = ROOT / "reports" / "figures" / "summary"
OUT_PATH = OUT_DIR / "figure_21_cwru_resnet18_confusion_matrix_grouped_pretty.png"

CLASS_NAMES = [
    "Normal",
    "IR_007",
    "IR_014",
    "IR_021",
    "Ball_007",
    "Ball_014",
    "Ball_021",
    "OR_007",
    "OR_014",
    "OR_021",
]


def read_confusion_matrix() -> np.ndarray:
    with CM_CSV.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = []
        for row in reader:
            rows.append([int(value) for value in row[1:]])
    return np.asarray(rows, dtype=int)


def read_metrics() -> tuple[float, float, int]:
    with METRICS_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    return float(row["accuracy"]), float(row["macro_f1"]), int(row["test_windows"])


def main() -> None:
    cm = read_confusion_matrix()
    accuracy, macro_f1, test_windows = read_metrics()

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(10.6, 9.1), facecolor="white")
    image = ax.imshow(cm_pct, cmap="YlGnBu", vmin=0, vmax=100, interpolation="nearest")

    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=42, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(CLASS_NAMES)

    ax.set_xticks(np.arange(-0.5, len(CLASS_NAMES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CLASS_NAMES), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            if value == 0:
                continue
            percent = cm_pct[i, j]
            is_diag = i == j
            color = "white" if percent >= 65 else "#1f2933"
            if not is_diag:
                color = "#9d0208"
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="#d00000",
                        linewidth=2.4,
                        zorder=4,
                    )
                )
            ax.text(
                j,
                i,
                f"{value}\n{percent:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=10.5,
                fontweight="bold" if is_diag else "semibold",
                zorder=5,
            )

    for i in range(cm.shape[0]):
        ax.add_patch(
            Rectangle(
                (i - 0.5, i - 0.5),
                1,
                1,
                fill=False,
                edgecolor="#0b3d91",
                linewidth=1.2,
                alpha=0.65,
                zorder=3,
            )
        )

    ax.set_xlabel("Предсказанный класс", labelpad=12)
    ax.set_ylabel("Истинный класс", labelpad=12)
    fig.text(
        0.50,
        0.035,
        f"Accuracy = {accuracy:.3f}; Macro-F1 = {macro_f1:.3f}; число окон = {test_windows}",
        ha="center",
        va="center",
        fontsize=11,
        color="#425466",
    )

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label("Доля внутри истинного класса, %")
    cbar.outline.set_visible(False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.12, right=0.90, top=0.97, bottom=0.19)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
