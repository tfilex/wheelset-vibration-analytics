"""Build revised Figure 31 for the VKR RUL comparison.

The figure keeps the historical family-level PHM_norm values, but avoids
presenting them as the final demo_best/rul model list.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "reports" / "figures" / "summary" / "figure_31_family_phm_norm_comparison.png"
TABLE_PATH = ROOT / "reports" / "tables_for_vkr" / "figure_31_family_phm_norm_comparison.csv"


MODELS = [
    {
        "family": "Базовая",
        "model": "ResNet+LSTM",
        "phm_norm": 0.604,
        "kind": "Одиночная модель",
        "color": "#8D9A9E",
    },
    {
        "family": "v3_rnn",
        "model": "ImprovedTransformer\nFrozen",
        "phm_norm": 0.712,
        "kind": "Одиночная модель",
        "color": "#8C2D3E",
    },
    {
        "family": "v4_tcn",
        "model": "BiTCN",
        "phm_norm": 0.654,
        "kind": "Одиночная модель",
        "color": "#5D7F4F",
    },
    {
        "family": "v5_odd",
        "model": "Conformer",
        "phm_norm": 0.689,
        "kind": "Одиночная модель",
        "color": "#7B4F7A",
    },
    {
        "family": "Ансамбль",
        "model": "0,52 IT + 0,28 Conformer\n+ 0,20 BiTCN",
        "phm_norm": 0.728,
        "kind": "Исследовательский ансамбль",
        "color": "#333333",
    },
]


def comma(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def main() -> None:
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TABLE_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["family", "model", "kind", "phm_norm"])
        for item in MODELS:
            writer.writerow(
                [
                    item["family"],
                    item["model"].replace("\n", " "),
                    item["kind"],
                    item["phm_norm"],
                ]
            )

    labels = [f"{item['family']}\n{item['model']}" for item in MODELS]
    values = [item["phm_norm"] for item in MODELS]
    colors = [item["color"] for item in MODELS]
    x = list(range(len(MODELS)))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(11.8, 5.8), constrained_layout=True)
    ax.grid(axis="y", color="#DDD7D0", linewidth=0.8, alpha=0.75, zorder=0)
    ax.set_axisbelow(True)

    bars = ax.bar(
        x,
        values,
        width=0.62,
        color=colors,
        edgecolor="#222222",
        linewidth=0.8,
        zorder=3,
    )
    bars[-1].set_linewidth(1.6)
    bars[-1].set_hatch("//")

    best_single = 0.712
    ax.axhline(
        best_single,
        color="#6A6A6A",
        linewidth=1.2,
        linestyle=(0, (4, 3)),
        zorder=2,
    )

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.007,
            comma(value),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold" if value == max(values) else "normal",
            color="#222222",
        )

    ax.text(
        3.85,
        best_single + 0.004,
        "лучшая одиночная модель",
        ha="right",
        va="bottom",
        fontsize=10,
        color="#555555",
    )

    ax.set_ylabel("PHM_norm")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.56, 0.75)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.3f}".replace(".", ",")))

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#A8A8A8")
    ax.spines["bottom"].set_color("#A8A8A8")

    single_proxy = plt.Rectangle((0, 0), 1, 1, facecolor="#8C2D3E", edgecolor="#222222")
    ensemble_proxy = plt.Rectangle((0, 0), 1, 1, facecolor="#333333", edgecolor="#222222", hatch="//")
    line_proxy = plt.Line2D([0], [0], color="#6A6A6A", linestyle=(0, (4, 3)), linewidth=1.2)
    ax.legend(
        [single_proxy, ensemble_proxy, line_proxy],
        ["Одиночные модели", "Ансамбль", "Уровень лучшей одиночной модели"],
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        edgecolor="#DDDDDD",
        fontsize=10,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(OUT_PATH)
    print(TABLE_PATH)


if __name__ == "__main__":
    main()
