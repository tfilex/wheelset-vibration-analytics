"""Generate report-ready Figures 17 and 18 for the baseline ResNet+LSTM RUL model.

The repository does not store per-epoch numeric history for this baseline run,
so the script uses the saved experiment PNGs as sources and wraps them in a
consistent VKR layout with Russian titles and captions.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR = ROOT / "reports" / "figures" / "summary"
BASELINE_DIR = (
    SUMMARY_DIR
    / "train_three_models_2_0_nonfrozen"
    / "balanced"
    / "ws1024"
)

FIGURE_SPECS = [
    {
        "source": BASELINE_DIR / "lstm_ws1024_learning_curves_balanced.png",
        "target": SUMMARY_DIR / "figure_17_baseline_resnet_lstm_learning_curves.png",
        "title": "Рисунок 17. Кривые обучения базовой модели ResNet+LSTM",
        "subtitle": "Функция потерь Huber и MAE на обучающей и валидационной выборках",
        "caption": "Базовая модель: ResNet-encoder + LSTM, профиль balanced, окно 1024.",
    },
    {
        "source": BASELINE_DIR / "lstm_ws1024_rul_prediction_balanced.png",
        "target": SUMMARY_DIR / "figure_18_baseline_resnet_lstm_rul_prediction.png",
        "title": "Рисунок 18. Истинная и предсказанная кривые RUL",
        "subtitle": "Тестовая траектория XJTU-SY; серая зона обозначает область ограничения метки RUL > 0,8",
        "caption": "Метка RUL ограничивается сверху на уровне 0,8 для раннего участка деградации.",
        "add_cap_legend": True,
    },
]


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def centered_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font, fill: str) -> None:
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = left + (right - left - width) // 2
    y = top + (bottom - top - height) // 2
    draw.text((x, y), text, font=font, fill=fill)


def add_cap_zone(canvas: Image.Image, image_box: tuple[int, int, int, int]) -> None:
    left, top, right, bottom = image_box
    width = right - left
    height = bottom - top

    # Approximate the Matplotlib axes area in the saved source figure.
    plot_left = left + int(width * 0.075)
    plot_right = left + int(width * 0.955)
    plot_top = top + int(height * 0.115)
    plot_bottom = top + int(height * 0.865)
    cap_bottom = plot_top + int((plot_bottom - plot_top) * 0.20)

    overlay = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        (plot_left, plot_top, plot_right, cap_bottom),
        fill=(176, 190, 197, 78),
        outline=(120, 144, 156, 150),
        width=2,
    )
    canvas.alpha_composite(overlay)


def add_cap_legend(draw: ImageDraw.ImageDraw, x: int, y: int, font) -> None:
    draw.rounded_rectangle((x, y, x + 430, y + 44), radius=10, fill="#F5F5F5", outline="#B0BEC5", width=1)
    draw.rectangle((x + 18, y + 14, x + 58, y + 30), fill="#D9DDE3", outline="#9EA7B3")
    draw.text((x + 72, y + 10), "область RUL > 0,8", font=font, fill="#37474F")


def render_figure(source: Path, target: Path, title: str, subtitle: str, caption: str, add_cap: bool = False) -> Path:
    if not source.exists():
        raise FileNotFoundError(source)

    image = Image.open(source).convert("RGBA")
    max_width = 2500
    if image.width > max_width:
        ratio = max_width / image.width
        image = image.resize((max_width, int(image.height * ratio)), Image.Resampling.LANCZOS)

    margin_x = 95
    top = 145
    bottom = 95
    canvas_width = image.width + margin_x * 2
    canvas_height = image.height + top + bottom

    canvas = Image.new("RGBA", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(42, bold=True)
    subtitle_font = load_font(25)
    caption_font = load_font(24)
    legend_font = load_font(22)

    centered_text(draw, (0, 22, canvas_width, 80), title, title_font, "#1F2933")
    centered_text(draw, (0, 82, canvas_width, 125), subtitle, subtitle_font, "#455A64")

    image_x = margin_x
    image_y = top
    draw.rounded_rectangle(
        (image_x - 14, image_y - 14, image_x + image.width + 14, image_y + image.height + 14),
        radius=10,
        fill="#FFFFFF",
        outline="#CFD8DC",
        width=2,
    )
    canvas.alpha_composite(image, (image_x, image_y))

    if add_cap:
        add_cap_zone(canvas, (image_x, image_y, image_x + image.width, image_y + image.height))
        draw = ImageDraw.Draw(canvas)
        add_cap_legend(draw, image_x + 25, image_y + 25, legend_font)

    centered_text(
        draw,
        (margin_x, image_y + image.height + 28, canvas_width - margin_x, canvas_height - 24),
        caption,
        caption_font,
        "#37474F",
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(target, dpi=(150, 150), quality=95)
    return target


def main() -> None:
    for spec in FIGURE_SPECS:
        output = render_figure(
            source=spec["source"],
            target=spec["target"],
            title=spec["title"],
            subtitle=spec["subtitle"],
            caption=spec["caption"],
            add_cap=spec.get("add_cap_legend", False),
        )
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
