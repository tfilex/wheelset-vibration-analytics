"""Create a readable v3_rnn RUL prediction comparison figure from saved PNG artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR = ROOT / "reports" / "figures" / "summary"
SOURCE_DIR = SUMMARY_DIR / "train_rul_hybrid_v3_rnn" / "balanced" / "ws1024"
OUT_PATH = SUMMARY_DIR / "figure_19_v3_rnn_rul_predictions_comparison_readable.png"

PANELS = [
    (
        "а) ImprovedTransformer (frozen HPO)",
        SOURCE_DIR / "transformer_improved_ws1024_rul_prediction_smoothed_train_rul_hybrid_v3_rnn_profilebalanced_trials30_epochs10_featurecache_on_frozen.png",
    ),
    (
        "б) BiLSTM",
        SOURCE_DIR / "bilstm_ws1024_rul_prediction_smoothed_train_rul_hybrid_v3_rnn_profilebalanced_trials30_epochs10_featurecache_on.png",
    ),
    (
        "в) BiGRU",
        SOURCE_DIR / "bigru_ws1024_rul_prediction_smoothed_train_rul_hybrid_v3_rnn_profilebalanced_trials30_epochs10_featurecache_on.png",
    ),
    (
        "г) LSTM-Attention",
        SOURCE_DIR / "lstm_attn_ws1024_rul_prediction_smoothed_train_rul_hybrid_v3_rnn_profilebalanced_trials30_epochs10_featurecache_on.png",
    ),
    (
        "д) GRU-Attention",
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


def find_plot_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    arr = np.asarray(image.convert("RGB"))
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    # Matplotlib source figures use a large light-gray axes background.
    mask = (mx > 205) & (mx < 245) & ((mx - mn) < 8)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        width, height = image.size
        return (int(width * 0.03), int(height * 0.10), int(width * 0.98), int(height * 0.96))

    row_counts = np.bincount(ys, minlength=arr.shape[0])
    col_counts = np.bincount(xs, minlength=arr.shape[1])
    rows = np.where(row_counts > arr.shape[1] * 0.25)[0]
    cols = np.where(col_counts > arr.shape[0] * 0.15)[0]
    if len(rows) == 0 or len(cols) == 0:
        width, height = image.size
        return (int(width * 0.03), int(height * 0.10), int(width * 0.98), int(height * 0.96))

    width, height = image.size
    left = max(0, int(cols.min()) - 260)
    top = max(0, int(rows.min()) - 42)
    right = min(width, int(cols.max()) + 68)
    bottom = min(height, int(rows.max()) + 138)
    return (left, top, right, bottom)


def render_panel(label: str, source: Path, size: tuple[int, int], label_font: ImageFont.ImageFont) -> Image.Image:
    if not source.exists():
        raise FileNotFoundError(source)

    source_img = Image.open(source).convert("RGBA")
    crop = source_img.crop(find_plot_bbox(source_img))
    crop.thumbnail((size[0] - 64, size[1] - 126), Image.Resampling.LANCZOS)

    panel = Image.new("RGBA", size, "white")
    draw = ImageDraw.Draw(panel)
    draw.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=8, fill="#FFFFFF", outline="#CBD5E1", width=2)
    draw.rectangle((1, 1, size[0] - 2, 64), fill="#F8FAFC")
    draw.line((0, 64, size[0], 64), fill="#E2E8F0", width=2)
    draw.text((28, 17), label, font=label_font, fill="#111827")
    panel.alpha_composite(crop, ((size[0] - crop.width) // 2, 84))
    return panel


def draw_legend(draw: ImageDraw.ImageDraw, x: int, y: int, font: ImageFont.ImageFont) -> None:
    draw.rounded_rectangle((x, y, x + 690, y + 48), radius=8, fill="#FFFFFF", outline="#CBD5E1", width=1)
    draw.line((x + 28, y + 17, x + 96, y + 17), fill="#2563EB", width=5)
    draw.text((x + 112, y + 5), "Истинный RUL", font=font, fill="#1F2937")
    draw.line((x + 352, y + 17, x + 420, y + 17), fill="#DC2626", width=5)
    draw.text((x + 436, y + 5), "Предсказанный RUL", font=font, fill="#1F2937")


def main() -> None:
    title_font = load_font(42, bold=True)
    legend_font = load_font(24)
    label_font = load_font(32, bold=True)

    panel_size = (1850, 805)
    gap_x = 72
    gap_y = 58
    margin_x = 95
    top_h = 112
    bottom_h = 42

    canvas_w = margin_x * 2 + 2 * panel_size[0] + gap_x
    canvas_h = top_h + 3 * panel_size[1] + 2 * gap_y + bottom_h
    canvas = Image.new("RGBA", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text((margin_x, 30), "Сравнение предсказаний моделей v3_rnn", font=title_font, fill="#111827")
    draw_legend(draw, canvas_w - margin_x - 690, 27, legend_font)

    positions = [
        (margin_x, top_h),
        (margin_x + panel_size[0] + gap_x, top_h),
        (margin_x, top_h + panel_size[1] + gap_y),
        (margin_x + panel_size[0] + gap_x, top_h + panel_size[1] + gap_y),
        (margin_x + (panel_size[0] + gap_x) // 2, top_h + 2 * (panel_size[1] + gap_y)),
    ]

    for (label, source), (x, y) in zip(PANELS, positions):
        panel = render_panel(label, source, panel_size, label_font)
        canvas.alpha_composite(panel, (x, y))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(OUT_PATH, dpi=(300, 300), quality=95)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
