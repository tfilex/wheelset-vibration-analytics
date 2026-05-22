"""Generate Figure 19: v3_rnn model prediction comparison.

The saved experiments contain model-level PNG plots, not raw prediction series.
This script composes the report figure from those validated artifacts so the
result remains reproducible without rerunning inference.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR = ROOT / "reports" / "figures" / "summary"
SOURCE_DIR = SUMMARY_DIR / "train_rul_hybrid_v3_rnn" / "balanced" / "ws1024"
OUT_PATH = SUMMARY_DIR / "figure_19_v3_rnn_rul_predictions_comparison_v2.png"

PANELS = [
    (
        "А. ImprovedTransformer (frozen HPO)",
        SOURCE_DIR / "transformer_improved_ws1024_rul_prediction_smoothed_train_rul_hybrid_v3_rnn_profilebalanced_trials30_epochs10_featurecache_on_frozen.png",
    ),
    (
        "Б. BiLSTM",
        SOURCE_DIR / "bilstm_ws1024_rul_prediction_smoothed_train_rul_hybrid_v3_rnn_profilebalanced_trials30_epochs10_featurecache_on.png",
    ),
    (
        "В. BiGRU",
        SOURCE_DIR / "bigru_ws1024_rul_prediction_smoothed_train_rul_hybrid_v3_rnn_profilebalanced_trials30_epochs10_featurecache_on.png",
    ),
    (
        "Г. LSTM-Attention",
        SOURCE_DIR / "lstm_attn_ws1024_rul_prediction_smoothed_train_rul_hybrid_v3_rnn_profilebalanced_trials30_epochs10_featurecache_on.png",
    ),
    (
        "Д. GRU-Attention",
        SOURCE_DIR / "gru_attn_ws1024_rul_prediction_smoothed_train_rul_hybrid_v3_rnn_profilebalanced_trials30_epochs10_featurecache_on.png",
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


def crop_plot_area(image: Image.Image) -> Image.Image:
    # The source Matplotlib figures have generous margins. This crop keeps axes,
    # legend and labels while removing excess whitespace around the image.
    width, height = image.size
    left = int(width * 0.035)
    top = int(height * 0.055)
    right = int(width * 0.985)
    bottom = int(height * 0.94)
    return image.crop((left, top, right, bottom))


def render_panel(source: Path, label: str, size: tuple[int, int], label_font) -> Image.Image:
    if not source.exists():
        raise FileNotFoundError(source)

    img = Image.open(source).convert("RGBA")
    img = crop_plot_area(img)
    img.thumbnail((size[0] - 34, size[1] - 76), Image.Resampling.LANCZOS)

    panel = Image.new("RGBA", size, "white")
    draw = ImageDraw.Draw(panel)
    draw.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=12, fill="#FFFFFF", outline="#CFD8DC", width=2)
    draw.text((22, 15), label, font=label_font, fill="#263238")
    panel.alpha_composite(img, ((size[0] - img.width) // 2, 58))
    return panel


def main() -> None:
    title_font = load_font(38, bold=True)
    subtitle_font = load_font(24)
    label_font = load_font(22, bold=True)

    panel_size = (850, 390)
    gap = 30
    margin_x = 70
    title_h = 105
    bottom_h = 45

    canvas_w = margin_x * 2 + 3 * panel_size[0] + 2 * gap
    canvas_h = title_h + 2 * panel_size[1] + gap + bottom_h
    canvas = Image.new("RGBA", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    centered_text(
        draw,
        (0, 20, canvas_w, 62),
        "Сравнение предсказаний моделей семейства v3_rnn",
        title_font,
        "#1F2933",
    )
    centered_text(
        draw,
        (0, 64, canvas_w, 98),
        "Одна тестовая траектория XJTU-SY: истинная и предсказанная кривые RUL",
        subtitle_font,
        "#455A64",
    )

    positions = [
        (margin_x, title_h),
        (margin_x + panel_size[0] + gap, title_h),
        (margin_x + 2 * (panel_size[0] + gap), title_h),
        (margin_x + (panel_size[0] + gap) // 2, title_h + panel_size[1] + gap),
        (margin_x + (panel_size[0] + gap) // 2 + panel_size[0] + gap, title_h + panel_size[1] + gap),
    ]

    for (label, source), (x, y) in zip(PANELS, positions):
        panel = render_panel(source, label, panel_size, label_font)
        canvas.alpha_composite(panel, (x, y))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(OUT_PATH, dpi=(150, 150), quality=95)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
