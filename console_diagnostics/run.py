"""Run RUL diagnostics without the Streamlit UI."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr
from io import StringIO
import logging
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)

from src.demo.mock_data import TEST_BEARINGS  # noqa: E402
from src.utils.health_index import (  # noqa: E402
    compute_hi,
    compute_slope,
    format_rul_display,
    get_status,
    rul_to_km,
)


ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "blue": "\033[94m",
    "green": "\033[92m",
    "orange": "\033[93m",
    "red": "\033[91m",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline RUL/Health Index diagnostics for XJTU-SY bearings."
    )
    parser.add_argument(
        "--bearing",
        default=TEST_BEARINGS[0],
        choices=TEST_BEARINGS,
        help="Test bearing to process.",
    )
    parser.add_argument("--output-steps", type=int, default=100)
    parser.add_argument("--anchor-points", type=int, default=16)
    parser.add_argument("--window", type=int, default=10, help="HI smoothing window.")
    parser.add_argument("--trend", type=int, default=20, help="HI slope window.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="Colorize console status output.",
    )
    return parser.parse_args()


def use_color(mode: str) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    return sys.stdout.isatty()


def colorize(text: str, color: str, enabled: bool, bold: bool = False) -> str:
    if not enabled:
        return text
    prefix = ANSI["bold"] if bold else ""
    return f"{prefix}{ANSI.get(color, '')}{text}{ANSI['reset']}"


def save_figure(df: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["step"], df["true_rul"], label="True RUL", color="#2ca02c", lw=2)
    ax.plot(df["step"], df["pred_rul"], label="Pred RUL", color="#d62728", lw=2)
    ax.plot(df["step"], df["hi"], label="Health Index", color="#1f77b4", lw=2)
    for threshold, label, color in [
        (0.85, "Норма", "orange"),
        (0.60, "Контроль", "red"),
        (0.35, "Авария", "darkred"),
    ]:
        ax.axhline(threshold, ls="--", lw=1, color=color, alpha=0.7)
        ax.text(df["step"].min(), threshold + 0.01, label, color=color, fontsize=9)
    ax.set_title("Console RUL diagnostics")
    ax.set_xlabel("Шаг наблюдения")
    ax.set_ylabel("Нормированное значение")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    colors_enabled = use_color(args.color)
    with redirect_stderr(StringIO()):
        from src.prediction.demo_inference import XJTU_BEARING_DIRS, predict_rul_series

        history, metadata = predict_rul_series(
            args.bearing,
            output_steps=args.output_steps,
            anchor_points=args.anchor_points,
        )

    hi = compute_hi(history["pred_rul"].to_numpy(), window=args.window)
    slope = compute_slope(hi, n=args.trend)
    rul_km, sigma_km = rul_to_km(float(hi[-1]), slope)
    level, name, color = get_status(float(hi[-1]))

    output = history.copy()
    output["hi"] = hi
    output["status_level"] = level
    output["status_name"] = name
    output["rul_km"] = rul_km
    output["sigma_km"] = sigma_km

    safe_bearing = args.bearing.lower()
    results_path = PROJECT_ROOT / args.results_dir / f"offline_rul_diagnostics_{safe_bearing}.csv"
    figure_path = PROJECT_ROOT / args.figures_dir / f"offline_rul_hi_{safe_bearing}.png"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(results_path, index=False)
    save_figure(output, figure_path)

    status_text = f"{level}. {name}"
    print(colorize("Console RUL diagnostics", "blue", colors_enabled, bold=True))
    print(f"Bearing: {args.bearing}")
    print(f"Data: {XJTU_BEARING_DIRS[args.bearing]}")
    print(f"Model: {metadata['model_name']}")
    print(f"Checkpoint: {metadata['checkpoint_path']}")
    print(f"Health Index: {hi[-1]:.3f}")
    print(f"Slope: {slope:.6f}")
    print(f"RUL: {format_rul_display(rul_km, sigma_km)}")
    print(f"Status: {colorize(status_text, color, colors_enabled, bold=True)}")
    print(f"CSV: {results_path}")
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
