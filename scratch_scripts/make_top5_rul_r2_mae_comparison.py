"""Create a top-5 RUL configuration comparison by Test R2 and Test MAE."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "reports" / "tables_for_vkr" / "vkr_model_metrics_all.csv"
OUT_DIR = ROOT / "reports" / "figures" / "summary"
TABLE_DIR = ROOT / "reports" / "tables_for_vkr"
FIG_PATH = OUT_DIR / "top5_rul_configs_test_r2_mae_comparison.png"
CSV_PATH = TABLE_DIR / "top5_rul_configs_test_r2_mae_comparison.csv"

NAME_MAP = {
    "transformer_improved_frozen": "ImprovedTransformer\nfrozen",
    "transformer_improved": "ImprovedTransformer\nfast",
    "bilstm": "BiLSTM\nfinetune",
    "lstm_attn_frozen": "LSTM-Attention\nfrozen",
    "transformer": "Transformer\nfrozen",
    "lstm": "LSTM\nfrozen",
    "gru": "GRU",
}


def clean_label(row: pd.Series) -> str:
    temporal = str(row["temporal_type"])
    label = NAME_MAP.get(temporal, temporal.replace("_", " "))
    family = str(row["family"])
    if family not in label:
        label = f"{label}\n{family}"
    return label


def load_top5() -> pd.DataFrame:
    df = pd.read_csv(METRICS_PATH)
    df = df.replace([np.inf, -np.inf], np.nan)
    valid = df.dropna(subset=["test_r2", "test_mae"]).copy()
    valid = valid[(valid["test_mae"] >= 0) & (valid["test_mae"] <= 1.0)]
    top5 = valid.sort_values(["test_r2", "test_mae"], ascending=[False, True]).head(5).copy()
    top5["label"] = top5.apply(clean_label, axis=1)
    return top5.reset_index(drop=True)


def draw(top5: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    x = np.arange(len(top5))
    width = 0.36
    r2 = top5["test_r2"].to_numpy(dtype=float)
    mae = top5["test_mae"].to_numpy(dtype=float)

    fig, ax_r2 = plt.subplots(figsize=(12.2, 6.0), facecolor="white")
    ax_mae = ax_r2.twinx()

    bars_r2 = ax_r2.bar(
        x - width / 2,
        r2,
        width,
        color="#2563EB",
        edgecolor="white",
        linewidth=1.0,
        label="Test R²",
        zorder=3,
    )
    bars_mae = ax_mae.bar(
        x + width / 2,
        mae,
        width,
        color="#F59E0B",
        edgecolor="white",
        linewidth=1.0,
        label="Test MAE",
        zorder=3,
    )

    ax_r2.axhline(0, color="#64748B", linewidth=1.1, linestyle="--", alpha=0.8)
    ax_r2.set_ylabel("Test R²")
    ax_mae.set_ylabel("Test MAE")
    ax_r2.set_xticks(x)
    ax_r2.set_xticklabels(top5["label"], ha="center")
    ax_r2.set_xlabel("Конфигурация модели")

    ymin = min(-0.04, float(r2.min()) - 0.02)
    ymax = max(0.13, float(r2.max()) + 0.03)
    ax_r2.set_ylim(ymin, ymax)
    ax_mae.set_ylim(0, max(0.28, float(mae.max()) + 0.04))

    ax_r2.grid(True, axis="y", alpha=0.22, zorder=0)
    ax_r2.spines[["top", "right"]].set_visible(False)
    ax_mae.spines[["top", "left"]].set_visible(False)

    for bar, value in zip(bars_r2, r2):
        offset = 0.006 if value >= 0 else -0.012
        va = "bottom" if value >= 0 else "top"
        ax_r2.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:+.3f}",
            ha="center",
            va=va,
            fontsize=9,
            color="#1E3A8A",
            fontweight="bold" if value > 0 else "normal",
        )

    for bar, value in zip(bars_mae, mae):
        ax_mae.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.008,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#92400E",
        )

    handles = [bars_r2, bars_mae]
    labels = ["Test R²", "Test MAE"]
    ax_r2.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=True)

    fig.text(
        0.5,
        0.025,
        "Топ-5 выбраны по Test R² среди валидных конфигураций с 0 ≤ MAE ≤ 1",
        ha="center",
        va="center",
        fontsize=10,
        color="#52616B",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.075, right=0.925, top=0.86, bottom=0.24)
    fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    top5 = load_top5()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    cols = [
        "family",
        "profile",
        "final_fit_mode",
        "window_size",
        "temporal_type",
        "test_r2",
        "test_mae",
        "test_rmse",
        "test_phm_score",
        "label",
    ]
    top5[cols].to_csv(CSV_PATH, index=False, encoding="utf-8")
    draw(top5)
    print(f"Saved: {FIG_PATH}")
    print(f"Saved: {CSV_PATH}")
    print(top5[cols].to_string(index=False))


if __name__ == "__main__":
    main()
