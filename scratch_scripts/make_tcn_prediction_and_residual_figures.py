"""Generate compact VKR figures for all TCN-family models.

Creates two figures from saved v4_tcn experiment PNGs:
1. true vs predicted RUL curves;
2. residual plots.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR = ROOT / "reports" / "figures" / "summary"
SOURCE_DIR = SUMMARY_DIR / "train_rul_hybrid_v4_tcn" / "balanced" / "ws2048"

MODEL_SPECS = [
    (
        "А. TCN",
        "tcn_ws2048_rul_prediction_smoothed_train_rul_hybrid_v4_tcn_profilebalanced_trials30_epochs25_featurecache_on.png",
        "tcn_ws2048_residuals_smoothed_train_rul_hybrid_v4_tcn_profilebalanced_trials30_epochs25_featurecache_on.png",
    ),
    (
        "Б. TCN-Attention",
        "tcna_ws2048_rul_prediction_smoothed_train_rul_hybrid_v4_tcn_profilebalanced_trials30_epochs25_featurecache_on.png",
        "tcna_ws2048_residuals_smoothed_train_rul_hybrid_v4_tcn_profilebalanced_trials30_epochs25_featurecache_on.png",
    ),
    (
        "В. BiTCN",
        "tcn_bi_ws2048_rul_prediction_smoothed_train_rul_hybrid_v4_tcn_profilebalanced_trials30_epochs25_featurecache_on.png",
        "tcn_bi_ws2048_residuals_smoothed_train_rul_hybrid_v4_tcn_profilebalanced_trials30_epochs25_featurecache_on.png",
    ),
    (
        "Г. MS-TCN",
        "tcn_ms_ws2048_rul_prediction_smoothed_train_rul_hybrid_v4_tcn_profilebalanced_trials30_epochs25_featurecache_on.png",
        "tcn_ms_ws2048_residuals_smoothed_train_rul_hybrid_v4_tcn_profilebalanced_trials30_epochs25_featurecache_on.png",
    ),
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
    draw.text((left + (right - left - width) // 2, top + (bottom - top - height) // 2), text, font=font, fill=fill)


def crop_plot_area(image: Image.Image, kind: str) -> Image.Image:
    width, height = image.size
    if kind == "residuals":
        left = int(width * 0.035)
        top = int(height * 0.055)
        right = int(width * 0.985)
        bottom = int(height * 0.94)
    else:
        left = int(width * 0.035)
        top = int(height * 0.055)
        right = int(width * 0.985)
        bottom = int(height * 0.94)
    return image.crop((left, top, right, bottom))


def render_panel(source: Path, label: str, size: tuple[int, int], label_font, kind: str) -> Image.Image:
    if not source.exists():
        raise FileNotFoundError(source)
    img = Image.open(source).convert("RGBA")
    img = crop_plot_area(img, kind=kind)
    img.thumbnail((size[0] - 34, size[1] - 76), Image.Resampling.LANCZOS)

    panel = Image.new("RGBA", size, "white")
    draw = ImageDraw.Draw(panel)
    draw.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=12, fill="#FFFFFF", outline="#CFD8DC", width=2)
    draw.text((22, 15), label, font=label_font, fill="#263238")
    panel.alpha_composite(img, ((size[0] - img.width) // 2, 58))
    return panel


def render_figure(title: str, subtitle: str, output_path: Path, source_index: int, kind: str) -> Path:
    title_font = load_font(38, bold=True)
    subtitle_font = load_font(23)
    label_font = load_font(22, bold=True)

    panel_size = (1160, 430)
    gap = 30
    margin_x = 70
    title_h = 105
    bottom_h = 45

    canvas_w = margin_x * 2 + 2 * panel_size[0] + gap
    canvas_h = title_h + 2 * panel_size[1] + gap + bottom_h
    canvas = Image.new("RGBA", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    centered_text(draw, (0, 20, canvas_w, 62), title, title_font, "#1F2933")
    centered_text(draw, (0, 64, canvas_w, 98), subtitle, subtitle_font, "#455A64")

    positions = [
        (margin_x, title_h),
        (margin_x + panel_size[0] + gap, title_h),
        (margin_x, title_h + panel_size[1] + gap),
        (margin_x + panel_size[0] + gap, title_h + panel_size[1] + gap),
    ]

    for (label, pred_name, residual_name), (x, y) in zip(MODEL_SPECS, positions):
        filename = pred_name if source_index == 1 else residual_name
        panel = render_panel(SOURCE_DIR / filename, label, panel_size, label_font, kind=kind)
        canvas.alpha_composite(panel, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output_path, dpi=(150, 150), quality=95)
    return output_path


def main() -> None:
    outputs = [
        render_figure(
            title="Сравнение истинных и предсказанных RUL-кривых TCN-моделей",
            subtitle="Семейство v4_tcn, профиль balanced, окно 2048",
            output_path=SUMMARY_DIR / "figure_tcn_true_vs_predicted_all_models.png",
            source_index=1,
            kind="prediction",
        ),
        render_figure(
            title="Сравнение остатков предсказания TCN-моделей",
            subtitle="Residuals для семейства v4_tcn, профиль balanced, окно 2048",
            output_path=SUMMARY_DIR / "figure_tcn_residuals_all_models.png",
            source_index=2,
            kind="residuals",
        ),
    ]
    for output in outputs:
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
