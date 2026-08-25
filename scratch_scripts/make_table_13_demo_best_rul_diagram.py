from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter


TABLE_PATH = Path("reports/tables_for_vkr/table_13_demo_best_rul_top5.csv")
OUT_PATH = Path("reports/figures/summary/figure_demo_best_rul_top5_r2_mae.png")
OUT_V2_PATH = Path("reports/figures/summary/figure_demo_best_rul_top5_r2_mae_v2.png")


def comma(value: float, digits: int = 3, signed: bool = False) -> str:
    template = f"{{:{'+' if signed else ''}.{digits}f}}"
    return template.format(value).replace(".", ",")


def main() -> None:
    df = pd.read_csv(TABLE_PATH).sort_values("Test R2", ascending=True)

    labels = df["Модель"].tolist()
    y = range(len(df))

    # Palette deliberately avoids the yellow/blue pairing.
    r2_color = "#8C2D3E"
    r2_highlight = "#5F1F2B"
    mae_color = "#3F7D55"
    mae_highlight = "#24563A"
    grid_color = "#D9D2CC"
    text_color = "#222222"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), constrained_layout=True)

    r2_values = df["Test R2"].to_numpy()
    r2_best = r2_values.max()
    r2_colors = [r2_highlight if value == r2_best else r2_color for value in r2_values]
    axes[0].barh(y, r2_values, color=r2_colors, edgecolor="#4F1A25", linewidth=0.7)
    axes[0].axvline(0, color="#4A4A4A", linewidth=1.0)
    axes[0].set_yticks(list(y), labels)
    axes[0].set_xlabel("Test R²")
    axes[0].set_title("Коэффициент детерминации")
    axes[0].grid(axis="x", color=grid_color, linewidth=0.8, alpha=0.7)
    axes[0].set_axisbelow(True)
    axes[0].set_xlim(min(-0.48, r2_values.min() - 0.05), 0.04)
    for pos, value in zip(y, r2_values):
        axes[0].text(
            value - 0.012,
            pos,
            comma(value, signed=True),
            va="center",
            ha="right",
            color=text_color,
            fontsize=10,
        )

    mae_values = df["MAE"].to_numpy()
    mae_best = mae_values.min()
    mae_colors = [mae_highlight if value == mae_best else mae_color for value in mae_values]
    axes[1].barh(y, mae_values, color=mae_colors, edgecolor="#244C36", linewidth=0.7)
    axes[1].set_yticks(list(y), labels)
    axes[1].set_xlabel("MAE")
    axes[1].set_title("Средняя абсолютная ошибка")
    axes[1].grid(axis="x", color=grid_color, linewidth=0.8, alpha=0.7)
    axes[1].set_axisbelow(True)
    axes[1].set_xlim(0, max(mae_values) * 1.18)
    for pos, value in zip(y, mae_values):
        axes[1].text(
            value + 0.006,
            pos,
            comma(value),
            va="center",
            ha="left",
            color=text_color,
            fontsize=10,
        )

    for ax, letter in zip(axes, ["а)", "б)"]):
        ax.text(
            -0.08,
            1.04,
            letter,
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            va="bottom",
            color=text_color,
        )
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _pos: f"{value:.2f}".replace(".", ","))
        )
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#999999")
        ax.spines["bottom"].set_color("#999999")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=240, bbox_inches="tight")
    fig.savefig(OUT_V2_PATH, dpi=240, bbox_inches="tight")
    plt.close(fig)

    print(OUT_PATH)
    print(OUT_V2_PATH)


if __name__ == "__main__":
    main()
