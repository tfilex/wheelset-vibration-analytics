from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "figures" / "summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ARCH_PATH = OUT_DIR / "presentation_architecture_hybrid_cascade.png"
TRANSFER_PATH = OUT_DIR / "presentation_architecture_transfer_protocol.png"


COLORS = {
    "blue": "#0A2A5E",
    "green": "#15803D",
    "light_green": "#DDEEDD",
    "red": "#B13A4A",
    "purple": "#6C4D8F",
    "gray": "#EEF1F4",
    "dark": "#1F2933",
    "line": "#65717C",
    "orange": "#C86B3C",
    "cream": "#FFF4D7",
}


def add_box(
    ax,
    xy,
    wh,
    text,
    fc,
    ec="#65717C",
    lw=1.2,
    size=11,
    weight="normal",
    color=None,
    radius=0.06,
    ha="center",
):
    color = color or COLORS["dark"]
    x, y = xy
    w, h = wh
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.018,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha=ha,
        va="center",
        fontsize=size,
        color=color,
        fontweight=weight,
        linespacing=1.12,
    )
    return box


def add_arrow(ax, start, end, color=None, lw=1.35, style="-|>", rad=0):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=12,
        linewidth=lw,
        color=color or COLORS["line"],
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)
    return arrow


def add_cnn_stack(ax, x, y, label):
    widths = [0.48, 0.43, 0.38, 0.33, 0.28]
    heights = [0.78, 0.70, 0.62, 0.54, 0.46]
    for i, (w, h) in enumerate(zip(widths, heights)):
        dx = i * 0.18
        dy = i * 0.035
        front = Rectangle(
            (x + dx, y + dy),
            w,
            h,
            facecolor="#D06F45",
            edgecolor="#804225",
            linewidth=1.0,
        )
        side = Polygon(
            [
                (x + dx + w, y + dy),
                (x + dx + w + 0.08, y + dy + 0.08),
                (x + dx + w + 0.08, y + dy + h + 0.08),
                (x + dx + w, y + dy + h),
            ],
            facecolor="#F0B177",
            edgecolor="#804225",
            linewidth=0.8,
        )
        top = Polygon(
            [
                (x + dx, y + dy + h),
                (x + dx + 0.08, y + dy + h + 0.08),
                (x + dx + w + 0.08, y + dy + h + 0.08),
                (x + dx + w, y + dy + h),
            ],
            facecolor="#FFD89A",
            edgecolor="#804225",
            linewidth=0.8,
        )
        ax.add_patch(side)
        ax.add_patch(top)
        ax.add_patch(front)
        ax.text(
            x + dx + w / 2,
            y + dy - 0.11,
            f"CNN {i + 1}",
            ha="center",
            va="top",
            fontsize=8,
            color=COLORS["dark"],
        )
    ax.text(
        x + 0.48,
        y + 1.05,
        label,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=COLORS["dark"],
    )


def add_spectrogram(ax, extent, title, cmap="viridis"):
    x0, x1, y0, y1 = extent
    rng = np.random.default_rng(42)
    base = rng.normal(0, 0.12, (80, 80))
    yy = np.linspace(0, 1, 80)[:, None]
    xx = np.linspace(0, 1, 80)[None, :]
    data = base + 0.8 * np.exp(-((yy - 0.35) ** 2) / 0.01) * (0.35 + 0.65 * np.sin(18 * xx) ** 2)
    data += 0.55 * np.exp(-((yy - 0.66) ** 2) / 0.004) * (0.25 + 0.75 * np.cos(11 * xx) ** 2)
    ax.imshow(data, extent=extent, origin="lower", cmap=cmap, aspect="auto")
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#2C7A7B", linewidth=1.6))
    ax.text((x0 + x1) / 2, y0 - 0.12, title, ha="center", va="top", fontsize=9, color=COLORS["dark"])


def make_hybrid_architecture() -> None:
    fig, ax = plt.subplots(figsize=(16, 8), facecolor="white")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(
        0.6,
        7.55,
        "Гибридная архитектура предиктивной вибродиагностики",
        fontsize=22,
        fontweight="bold",
        color=COLORS["blue"],
    )

    add_spectrogram(ax, (0.8, 2.15, 4.65, 6.55), "STFT-образ\nCWRU")
    add_spectrogram(ax, (0.8, 2.15, 1.65, 3.55), "CWT-окна\nXJTU-SY")

    add_cnn_stack(ax, 2.9, 4.65, "ResNet-18 encoder")
    add_cnn_stack(ax, 2.9, 1.65, "RUL encoder")

    add_box(
        ax,
        (4.58, 3.70),
        (2.10, 0.70),
        "инициализация\nиз CWRU-энкодера",
        COLORS["light_green"],
        ec="#5B8C61",
        size=10,
        weight="bold",
    )

    add_arrow(ax, (2.15, 5.6), (2.88, 5.45))
    add_arrow(ax, (2.15, 2.6), (2.88, 2.45))
    add_arrow(ax, (4.15, 5.45), (5.15, 5.45))
    add_arrow(ax, (4.15, 2.45), (5.15, 2.45))
    add_arrow(ax, (4.65, 4.72), (4.65, 3.15), color="#5B8C61", lw=1.5, rad=-0.18)
    ax.text(
        4.92,
        3.34,
        "частичная\nдонастройка\nдля RUL",
        ha="left",
        va="center",
        fontsize=9,
        color="#2F6F3E",
        fontweight="bold",
    )

    add_box(ax, (5.35, 5.05), (1.55, 0.80), "эмбеддинг\nизображения", COLORS["gray"], size=10)
    add_box(ax, (5.35, 2.05), (1.55, 0.80), "эмбеддинги\n16 окон", COLORS["gray"], size=10)

    add_arrow(ax, (6.9, 5.45), (7.65, 5.45))
    add_arrow(ax, (6.9, 2.45), (7.65, 2.45))

    add_box(ax, (7.65, 4.95), (2.05, 1.00), "Classification head\nMLP + Softmax", "#E7EEF8", ec="#5D7FAE", size=11, weight="bold")
    add_box(ax, (7.65, 1.85), (2.05, 1.20), "Темпоральный блок\nLSTM / GRU /\nTransformer / TCN", "#EFE8F6", ec="#7B4F7A", size=11, weight="bold")

    add_arrow(ax, (9.7, 5.45), (11.0, 5.45))
    add_arrow(ax, (9.7, 2.45), (11.0, 2.45))

    add_box(ax, (11.0, 4.95), (1.9, 1.0), "10 классов\nсостояния", "#F7E6EA", ec=COLORS["red"], size=12, weight="bold")
    add_box(ax, (11.0, 1.85), (1.9, 1.2), "RUL\n1,0 → 0,0", "#E4F4EA", ec=COLORS["green"], size=13, weight="bold")

    add_arrow(ax, (12.9, 2.45), (14.0, 2.45))
    add_box(ax, (14.0, 1.85), (1.45, 1.2), "Health Index\nстатус\nпробег", COLORS["cream"], ec="#B08B2E", size=10, weight="bold")

    ax.text(0.65, 0.65, "а) классификационная ветвь", fontsize=11, fontweight="bold", color=COLORS["red"])
    ax.text(0.65, 0.35, "б) регрессионная ветвь RUL", fontsize=11, fontweight="bold", color=COLORS["green"])
    ax.text(
        6.8,
        0.43,
        "CNN-энкодер переносится из классификации в RUL и донастраивается под траектории деградации",
        fontsize=11,
        color=COLORS["dark"],
        ha="center",
    )

    fig.savefig(ARCH_PATH, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def make_transfer_protocol() -> None:
    fig, ax = plt.subplots(figsize=(16, 8), facecolor="white")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(
        0.6,
        7.55,
        "Протокол обучения и переноса признаков",
        fontsize=22,
        fontweight="bold",
        color=COLORS["blue"],
    )

    stages = [
        {
            "x": 0.8,
            "title": "1. CWRU pretrain",
            "body": "STFT-спектрограммы\nResNet-18\n10 классов дефектов",
            "color": "#E7EEF8",
            "edge": "#5D7FAE",
        },
        {
            "x": 4.25,
            "title": "2. XJTU-SY adaptation",
            "body": "одиночная CWT-скалограмма\nsingle-frame RUL\nсохранение encoder",
            "color": "#E4F4EA",
            "edge": "#15803D",
        },
        {
            "x": 7.7,
            "title": "3. Frozen HPO",
            "body": "CNN заморожен\nOptuna TPE\nпоиск верхней части",
            "color": "#FFF4D7",
            "edge": "#B08B2E",
        },
        {
            "x": 11.15,
            "title": "4. Final inference",
            "body": "16 окон → temporal head\ncheckpoint demo_best/rul\nRUL + Health Index",
            "color": "#F7E6EA",
            "edge": "#B13A4A",
        },
    ]

    for i, st in enumerate(stages):
        add_box(ax, (st["x"], 4.35), (2.65, 1.85), f"{st['title']}\n\n{st['body']}", st["color"], ec=st["edge"], size=11, weight="bold")
        add_box(ax, (st["x"] + 0.22, 2.55), (2.21, 0.82), ["классификационные\nпризнаки", "адаптация\nк деградации", "устойчивый\nпоиск HPO", "демо-инференс\nбез переобучения"][i], "white", ec=st["edge"], size=10)
        if i < len(stages) - 1:
            add_arrow(ax, (st["x"] + 2.65, 5.27), (stages[i + 1]["x"], 5.27), color=COLORS["line"], lw=1.6)

    add_box(
        ax,
        (3.55, 1.05),
        (8.9, 0.78),
        "ограниченный набор run-to-failure траекторий ⇒ осторожный выбор режима обучения и контроль вырождения",
        COLORS["gray"],
        ec="#B7C0C9",
        size=12,
        weight="bold",
    )

    ax.text(1.1, 0.45, "а) предобучение энкодера", fontsize=11, fontweight="bold", color="#5D7FAE")
    ax.text(5.1, 0.45, "б) адаптация признаков", fontsize=11, fontweight="bold", color=COLORS["green"])
    ax.text(8.65, 0.45, "в) подбор временной головы", fontsize=11, fontweight="bold", color="#8A6A20")
    ax.text(12.15, 0.45, "г) финальный инференс", fontsize=11, fontweight="bold", color=COLORS["red"])

    fig.savefig(TRANSFER_PATH, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    make_hybrid_architecture()
    make_transfer_protocol()
    print(ARCH_PATH)
    print(TRANSFER_PATH)


if __name__ == "__main__":
    main()
