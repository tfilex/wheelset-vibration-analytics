from argparse import Namespace
import sys

import pandas as pd

from console_diagnostics import run
from src.prediction import demo_inference as demo


def test_use_color_modes(monkeypatch):
    assert run.use_color("always") is True
    assert run.use_color("never") is False
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert run.use_color("auto") is True
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    assert run.use_color("auto") is False


def test_colorize_respects_enabled_flag():
    assert run.colorize("OK", "green", enabled=False, bold=True) == "OK"
    colored = run.colorize("OK", "green", enabled=True, bold=True)
    assert "\033[1m" in colored
    assert "\033[92m" in colored
    assert colored.endswith(run.ANSI["reset"])


def test_save_figure_writes_png(tmp_path):
    df = pd.DataFrame(
        {
            "step": [1, 2, 3],
            "true_rul": [1.0, 0.7, 0.4],
            "pred_rul": [0.9, 0.6, 0.3],
            "hi": [0.95, 0.65, 0.35],
        }
    )
    figure_path = tmp_path / "figures" / "diagnostics.png"

    run.save_figure(df, figure_path)

    assert figure_path.exists()
    assert figure_path.stat().st_size > 0


def test_main_writes_console_csv_and_figure(monkeypatch, tmp_path, capsys):
    history = pd.DataFrame(
        {
            "step": [1.0, 2.0, 3.0, 4.0],
            "true_rul": [1.0, 0.8, 0.6, 0.4],
            "pred_rul": [0.95, 0.75, 0.55, 0.35],
        }
    )

    def fake_predict_rul_series(bearing, output_steps, anchor_points):
        assert bearing == "Bearing1_3"
        assert output_steps == 4
        assert anchor_points == 2
        return history, {
            "model_name": "CNN + TEST",
            "checkpoint_path": "models/demo_best/rul/test.pth",
        }

    monkeypatch.setattr(demo, "predict_rul_series", fake_predict_rul_series)
    monkeypatch.setattr(demo, "XJTU_BEARING_DIRS", {"Bearing1_3": tmp_path / "bearing"})
    monkeypatch.setattr(run, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--bearing",
            "Bearing1_3",
            "--output-steps",
            "4",
            "--anchor-points",
            "2",
            "--window",
            "2",
            "--trend",
            "2",
            "--results-dir",
            "out/results",
            "--figures-dir",
            "out/figures",
            "--color",
            "never",
        ],
    )

    run.main()

    stdout = capsys.readouterr().out
    assert "Console RUL diagnostics" in stdout
    assert "Bearing: Bearing1_3" in stdout
    assert "Model: CNN + TEST" in stdout

    csv_path = tmp_path / "out/results/offline_rul_diagnostics_bearing1_3.csv"
    figure_path = tmp_path / "out/figures/offline_rul_hi_bearing1_3.png"
    assert csv_path.exists()
    assert figure_path.exists()
    written = pd.read_csv(csv_path)
    assert {"hi", "status_level", "status_name", "rul_km", "sigma_km"}.issubset(written.columns)
