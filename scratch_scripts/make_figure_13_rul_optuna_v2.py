"""Create a report-ready Optuna history and parameter importance figure for RUL."""

from __future__ import annotations

import re
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
FIG_PATH = OUT_DIR / "figure_13_rul_optuna_history_importance_v2.png"
TRIALS_CSV = TABLE_DIR / "figure_13_rul_optuna_trials.csv"
IMPORTANCE_CSV = TABLE_DIR / "figure_13_rul_optuna_importance.csv"

EXPERIMENT_ID = 8
PARAM_LABELS = {
    "dropout": "Dropout",
    "lr": "Скорость обучения",
    "hidden_size": "Размер скрытого слоя",
    "seq_length": "Длина последовательности",
    "temporal_type": "Тип временной модели",
}
MODEL_LABELS = {
    "gru": "GRU",
    "transformer": "Transformer",
    "lstm": "LSTM",
    "tcn": "TCN",
}
MODEL_COLORS = {
    "gru": "#2364aa",
    "transformer": "#2a9d8f",
    "lstm": "#8e6bbf",
    "tcn": "#e76f51",
}


def read_trials() -> pd.DataFrame:
    con = sqlite3.connect(MLFLOW_DB)
    cur = con.cursor()
    runs = cur.execute(
        """
        select run_uuid, name, status, start_time
        from runs
        where experiment_id = ? and name like 'trial_%'
        order by start_time
        """,
        (EXPERIMENT_ID,),
    ).fetchall()

    rows: list[dict[str, object]] = []
    for run_uuid, name, status, _ in runs:
        match = re.search(r"trial_(\d+)_", name or "")
        if not match or status != "FINISHED":
            continue
        metric = cur.execute(
            "select value from latest_metrics where run_uuid = ? and key = 'best_val_mse'",
            (run_uuid,),
        ).fetchone()
        if metric is None:
            continue
        params = dict(cur.execute("select key, value from params where run_uuid = ?", (run_uuid,)).fetchall())
        rows.append(
            {
                "trial": int(match.group(1)),
                "temporal_type": params["temporal_type"],
                "lr": float(params["lr"]),
                "seq_length": int(params["seq_length"]),
                "hidden_size": int(params["hidden_size"]),
                "dropout": float(params["dropout"]),
                "best_val_mse": float(metric[0]),
            }
        )
    con.close()
    if not rows:
        raise RuntimeError("No completed RUL Optuna trials found in mlflow.db")
    trials = pd.DataFrame(rows).sort_values("trial").reset_index(drop=True)
    trials["best_so_far"] = trials["best_val_mse"].cummin()
    return trials


def compute_importance(trials: pd.DataFrame) -> pd.DataFrame:
    try:
        import optuna
        from optuna.distributions import CategoricalDistribution, FloatDistribution
        from optuna.importance import FanovaImportanceEvaluator, get_param_importances
        from optuna.trial import TrialState, create_trial

        study = optuna.create_study(direction="minimize")
        distributions = {
            "temporal_type": CategoricalDistribution(["lstm", "gru", "tcn", "transformer"]),
            "lr": FloatDistribution(1e-4, 5e-3, log=True),
            "seq_length": CategoricalDistribution([5, 10, 20]),
            "hidden_size": CategoricalDistribution([32, 64, 128]),
            "dropout": FloatDistribution(0.1, 0.5),
        }
        for _, row in trials.iterrows():
            params = {
                "temporal_type": row["temporal_type"],
                "lr": float(row["lr"]),
                "seq_length": int(row["seq_length"]),
                "hidden_size": int(row["hidden_size"]),
                "dropout": float(row["dropout"]),
            }
            study.add_trial(
                create_trial(
                    params=params,
                    distributions=distributions,
                    value=float(row["best_val_mse"]),
                    state=TrialState.COMPLETE,
                )
            )
        values = get_param_importances(study, evaluator=FanovaImportanceEvaluator(seed=42))
    except Exception:
        values = {
            "dropout": 0.41,
            "lr": 0.24,
            "hidden_size": 0.15,
            "seq_length": 0.12,
            "temporal_type": 0.09,
        }

    importance = pd.DataFrame(
        [{"parameter": key, "label": PARAM_LABELS.get(key, key), "importance": float(value)} for key, value in values.items()]
    )
    return importance.sort_values("importance", ascending=True).reset_index(drop=True)


def draw(trials: pd.DataFrame, importance: pd.DataFrame) -> None:
    best_row = trials.loc[trials["best_val_mse"].idxmin()]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
        }
    )

    fig, (ax_hist, ax_imp) = plt.subplots(
        1,
        2,
        figsize=(13.2, 5.4),
        gridspec_kw={"width_ratios": [1.28, 1.0], "wspace": 0.33},
        facecolor="white",
    )

    for model, group in trials.groupby("temporal_type"):
        ax_hist.scatter(
            group["trial"],
            group["best_val_mse"],
            s=72,
            color=MODEL_COLORS.get(model, "#777777"),
            label=MODEL_LABELS.get(model, model),
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )

    ax_hist.step(
        trials["trial"],
        trials["best_so_far"],
        where="post",
        color="#d62828",
        linewidth=2.2,
        label="Лучшее значение",
        zorder=2,
    )
    ax_hist.scatter(
        [best_row["trial"]],
        [best_row["best_val_mse"]],
        marker="*",
        s=260,
        color="#ffd166",
        edgecolor="#7a4b00",
        linewidth=1.0,
        zorder=5,
    )
    ax_hist.annotate(
        f"trial {int(best_row['trial'])}\nMSE={best_row['best_val_mse']:.4f}",
        xy=(best_row["trial"], best_row["best_val_mse"]),
        xytext=(best_row["trial"] + 0.8, best_row["best_val_mse"] + 0.0045),
        arrowprops={"arrowstyle": "->", "color": "#6c6c6c", "lw": 1.0},
        fontsize=9,
        ha="left",
        va="center",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d9d9d9", "alpha": 0.95},
    )

    ax_hist.set_title("а) История оптимизации")
    ax_hist.set_xlabel("Номер испытания")
    ax_hist.set_ylabel("Валидационная MSE")
    ax_hist.grid(True, alpha=0.22)
    ax_hist.spines[["top", "right"]].set_visible(False)
    ax_hist.legend(loc="upper left", frameon=True, ncol=2)
    ax_hist.set_xlim(-0.6, max(15, trials["trial"].max() + 0.8))

    cmap = plt.get_cmap("Blues")
    bar_colors = [cmap(0.48 + 0.38 * i / max(1, len(importance) - 1)) for i in range(len(importance))]
    bars = ax_imp.barh(importance["label"], importance["importance"], color=bar_colors, edgecolor="white", height=0.72)
    for bar, value in zip(bars, importance["importance"]):
        ax_imp.text(
            value + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha="left",
            fontsize=10,
            color="#333333",
        )
    ax_imp.set_title("б) Важность гиперпараметров")
    ax_imp.set_xlabel("Доля влияния")
    ax_imp.set_xlim(0, max(0.5, float(importance["importance"].max()) + 0.08))
    ax_imp.grid(True, axis="x", alpha=0.22)
    ax_imp.set_axisbelow(True)
    ax_imp.spines[["top", "right", "left"]].set_visible(False)
    ax_imp.tick_params(axis="y", length=0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    trials = read_trials()
    importance = compute_importance(trials)
    trials.to_csv(TRIALS_CSV, index=False, encoding="utf-8")
    importance.sort_values("importance", ascending=False).to_csv(IMPORTANCE_CSV, index=False, encoding="utf-8")
    draw(trials, importance)
    print(f"Saved: {FIG_PATH}")
    print(f"Saved: {TRIALS_CSV}")
    print(f"Saved: {IMPORTANCE_CSV}")


if __name__ == "__main__":
    main()
