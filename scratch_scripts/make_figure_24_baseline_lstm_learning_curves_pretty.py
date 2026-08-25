"""Create report-ready learning curves for baseline ResNet+LSTM RUL model."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MLFLOW_DB = ROOT / "mlflow.db"
OUT_DIR = ROOT / "reports" / "figures" / "summary"
TABLE_DIR = ROOT / "reports" / "tables_for_vkr"
OUT_PATH = OUT_DIR / "figure_24_baseline_resnet_lstm_learning_curves_pretty.png"
CSV_PATH = TABLE_DIR / "figure_24_baseline_resnet_lstm_learning_curves.csv"

RUN_UUID = "6abf0898caf3445fa7546894c48e1498"


def read_metric(cur: sqlite3.Cursor, key: str) -> list[float]:
    rows = cur.execute(
        "select step, value from metrics where run_uuid = ? and key = ? order by step",
        (RUN_UUID, key),
    ).fetchall()
    return [float(value) for _, value in rows]


def load_history() -> pd.DataFrame:
    con = sqlite3.connect(MLFLOW_DB)
    cur = con.cursor()
    run = cur.execute("select name, experiment_id, status from runs where run_uuid = ?", (RUN_UUID,)).fetchone()
    if run is None:
        raise RuntimeError(f"Run not found: {RUN_UUID}")

    train_loss = read_metric(cur, "final_train_mse")
    val_loss = read_metric(cur, "final_val_mse")
    val_mae = read_metric(cur, "final_val_mae")
    con.close()

    n = min(len(train_loss), len(val_loss), len(val_mae))
    if n == 0:
        raise RuntimeError("No learning-curve metrics found")
    return pd.DataFrame(
        {
            "epoch": np.arange(1, n + 1),
            "train_loss": train_loss[:n],
            "val_loss": val_loss[:n],
            "val_mae": val_mae[:n],
        }
    )


def draw(history: pd.DataFrame) -> None:
    best_idx = int(history["val_loss"].idxmin())
    best = history.loc[best_idx]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), facecolor="white", gridspec_kw={"wspace": 0.24})
    ax_loss, ax_mae = axes

    blue = "#2563eb"
    red = "#dc2626"
    green = "#0f9f6e"
    gray = "#64748b"

    ax_loss.plot(history["epoch"], history["train_loss"], color=blue, linewidth=2.3, marker="o", markersize=4.5, label="Обучение")
    ax_loss.plot(history["epoch"], history["val_loss"], color=red, linewidth=2.3, marker="o", markersize=4.5, label="Валидация")
    ax_loss.scatter(best["epoch"], best["val_loss"], s=120, color="#facc15", edgecolor="#7a4b00", linewidth=1.0, zorder=5)
    ax_loss.axvline(best["epoch"], color=gray, linestyle="--", linewidth=1.1, alpha=0.55)
    ax_loss.annotate(
        f"лучшая эпоха {int(best['epoch'])}\nval loss={best['val_loss']:.3f}",
        xy=(best["epoch"], best["val_loss"]),
        xytext=(best["epoch"] + 0.7, best["val_loss"] + 0.018),
        arrowprops={"arrowstyle": "->", "color": gray, "lw": 1.0},
        fontsize=9,
        ha="left",
        va="center",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d9e2ec", "alpha": 0.95},
    )
    ax_loss.set_title("а) Функция потерь")
    ax_loss.set_xlabel("Эпоха")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.22)
    ax_loss.legend(loc="upper right", frameon=True)

    ax_mae.plot(history["epoch"], history["val_mae"], color=green, linewidth=2.4, marker="o", markersize=4.5, label="Валидация")
    best_mae = history.loc[history["val_mae"].idxmin()]
    ax_mae.scatter(best_mae["epoch"], best_mae["val_mae"], s=110, color="#facc15", edgecolor="#7a4b00", linewidth=1.0, zorder=5)
    ax_mae.axvline(best_mae["epoch"], color=gray, linestyle="--", linewidth=1.1, alpha=0.55)
    ax_mae.annotate(
        f"min MAE={best_mae['val_mae']:.3f}",
        xy=(best_mae["epoch"], best_mae["val_mae"]),
        xytext=(best_mae["epoch"] + 0.7, best_mae["val_mae"] + 0.025),
        arrowprops={"arrowstyle": "->", "color": gray, "lw": 1.0},
        fontsize=9,
        ha="left",
        va="center",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d9e2ec", "alpha": 0.95},
    )
    ax_mae.set_title("б) Средняя абсолютная ошибка")
    ax_mae.set_xlabel("Эпоха")
    ax_mae.set_ylabel("MAE")
    ax_mae.grid(True, alpha=0.22)
    ax_mae.legend(loc="upper right", frameon=True)

    for ax in axes:
        ax.set_xlim(0.5, float(history["epoch"].max()) + 0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=3)

    fig.text(
        0.5,
        0.018,
        "ResNet-encoder + LSTM, профиль balanced, окно 1024; обучающая MAE не логировалась в MLflow",
        ha="center",
        va="center",
        fontsize=10,
        color="#52616b",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.17)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    history = load_history()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    history.to_csv(CSV_PATH, index=False, encoding="utf-8")
    draw(history)
    print(f"Saved: {OUT_PATH}")
    print(f"Saved: {CSV_PATH}")
    print(history.to_string(index=False))


if __name__ == "__main__":
    main()
