"""Generate v5_odd residual diagnostics: scatter and residual histograms.

PatchTST and Conformer use saved balanced/ws2048 checkpoints. Mamba is shown as
unavailable because the experiment logs indicate it was skipped when mamba-ssm
was not installed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "figures" / "summary"
TABLE_DIR = ROOT / "reports" / "tables_for_vkr"
FIG_PATH = OUT_DIR / "v5_odd_residuals_diagnostics.png"
CSV_PATH = TABLE_DIR / "v5_odd_residuals_diagnostics_stats.csv"

MODELS = [
    {
        "name": "PatchTST",
        "mode": "finetune",
        "path": ROOT / "models" / "preds_5_odd" / "best_rul_patchtst_ws2048_v5odd_train_rul_hybrid_v5_odd_profilebalanced_trials30_epochs25_featurecache_on.pth",
    },
    {
        "name": "Conformer",
        "mode": "finetune",
        "path": ROOT / "models" / "preds_5_odd" / "best_rul_conformer_ws2048_v5odd_train_rul_hybrid_v5_odd_profilebalanced_trials30_epochs25_featurecache_on.pth",
    },
    {
        "name": "Mamba",
        "mode": "skipped",
        "path": None,
    },
]

PANEL_LETTERS = ["А", "Б", "В", "Г", "Д", "Е"]


def load_checkpoint_series(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    ckpt = torch.load(path, map_location="cpu")
    y_true = np.asarray(ckpt["test_labels"], dtype=float)
    y_pred = np.asarray(ckpt["test_predictions_raw"], dtype=float)
    metrics = {
        "test_mse": ckpt.get("test_mse"),
        "test_mae": ckpt.get("test_mae"),
        "test_r2": ckpt.get("test_r2"),
        "test_rmse": ckpt.get("test_rmse"),
        "test_phm_score": ckpt.get("test_phm_score"),
    }
    return y_true, y_pred, metrics


def style_axis(ax) -> None:
    ax.set_facecolor("#F5F5F5")
    ax.grid(color="white", linewidth=1.1)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#B0BEC5")
    ax.spines["bottom"].set_color("#B0BEC5")


def unavailable_axis(ax, title: str) -> None:
    style_axis(ax)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.text(
        0.5,
        0.52,
        "checkpoint отсутствует",
        ha="center",
        va="center",
        fontsize=13,
        color="#546E7A",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.42,
        "mamba-ssm не установлен",
        ha="center",
        va="center",
        fontsize=11,
        color="#78909C",
        transform=ax.transAxes,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    fig, axes = plt.subplots(3, 2, figsize=(11, 12), facecolor="white")
    fig.suptitle("Residuals для семейства v5_odd", fontsize=18, fontweight="bold", y=0.985)

    stats_rows = []
    for i, spec in enumerate(MODELS):
        scatter_ax = axes[i, 0]
        hist_ax = axes[i, 1]
        left_title = f"{PANEL_LETTERS[2 * i]}. {spec['name']} — true vs predicted"
        right_title = f"{PANEL_LETTERS[2 * i + 1]}. {spec['name']} — residuals"

        if spec["path"] is None or not spec["path"].exists():
            unavailable_axis(scatter_ax, left_title)
            unavailable_axis(hist_ax, right_title)
            stats_rows.append({
                "model": spec["name"],
                "mode": spec["mode"],
                "n": 0,
                "status": "missing checkpoint; mamba-ssm was not installed",
            })
            continue

        y_true, y_pred, metrics = load_checkpoint_series(spec["path"])
        residuals = y_pred - y_true

        style_axis(scatter_ax)
        scatter_ax.scatter(y_true, y_pred, alpha=0.42, s=10, color="#1565C0", edgecolors="none")
        min_axis = min(0.0, float(np.nanmin(y_true)), float(np.nanmin(y_pred)))
        max_axis = max(1.0, float(np.nanmax(y_true)), float(np.nanmax(y_pred)))
        scatter_ax.plot([min_axis, max_axis], [min_axis, max_axis], "--", color="#D32F2F", linewidth=1.8)
        scatter_ax.set_xlim(min_axis, max_axis)
        scatter_ax.set_ylim(min_axis, max_axis)
        scatter_ax.set_title(left_title, loc="left", fontweight="bold")
        scatter_ax.set_xlabel("True RUL")
        scatter_ax.set_ylabel("Predicted RUL")

        style_axis(hist_ax)
        hist_ax.hist(residuals, bins=40, color="#43A047", edgecolor="white", alpha=0.9)
        hist_ax.axvline(0.0, color="#263238", linestyle="--", linewidth=1.4)
        hist_ax.set_title(right_title, loc="left", fontweight="bold")
        hist_ax.set_xlabel("Predicted − True")
        hist_ax.set_ylabel("Count")

        stats_rows.append({
            "model": spec["name"],
            "mode": spec["mode"],
            "n": int(len(y_true)),
            "mean_pred": float(np.mean(y_pred)),
            "std_pred": float(np.std(y_pred)),
            "mean_residual": float(np.mean(residuals)),
            "std_residual": float(np.std(residuals)),
            "min_residual": float(np.min(residuals)),
            "max_residual": float(np.max(residuals)),
            "test_mse": metrics["test_mse"],
            "test_mae": metrics["test_mae"],
            "test_r2": metrics["test_r2"],
            "test_rmse": metrics["test_rmse"],
            "test_phm_score": metrics["test_phm_score"],
            "status": "ok",
        })

    plt.tight_layout(rect=(0, 0, 1, 0.965))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    pd.DataFrame(stats_rows).to_csv(CSV_PATH, index=False)
    print(f"Saved: {FIG_PATH}")
    print(f"Saved: {CSV_PATH}")


if __name__ == "__main__":
    main()
