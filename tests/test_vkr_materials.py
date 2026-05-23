from pathlib import Path

import pandas as pd

from scratch_scripts.make_vkr_materials import (
    build_materials,
    collect_metrics,
    select_best_by_family_mode,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_collect_metrics_normalizes_mixed_summary_formats(tmp_path):
    summary_dir = tmp_path / "reports" / "figures" / "summary"

    _write_csv(
        summary_dir
        / "train_rul_hybrid_v3_rnn"
        / "balanced"
        / "summary_metrics_train_rul_hybrid_v3_rnn_profilebalanced.csv",
        [
            {
                "mode": "balanced",
                "final_fit_mode": "frozen",
                "window_size": 1024,
                "temporal_type": "transformer_improved_frozen",
                "best_val_mse": 0.01,
                "test_mse": 0.02,
                "test_mae": 0.1,
                "inference_ms_per_sample": 0.2,
            },
            {
                "mode": "balanced",
                "final_fit_mode": "finetune",
                "window_size": 1024,
                "temporal_type": "transformer_improved_finetune",
                "best_val_mse": 0.02,
                "test_mse": 0.03,
                "test_mae": 0.2,
                "inference_ms_per_sample": 4.0,
            },
        ],
    )
    _write_csv(
        summary_dir
        / "train_three_models_2_frozen"
        / "balanced"
        / "summary_metrics.csv",
        [
            {
                "mode": "balanced",
                "window_size": 1024,
                "temporal_type": "gru",
                "best_val_mse": 0.04,
                "test_mse": 0.05,
                "test_mae": 0.3,
                "inference_ms_per_sample": 0.4,
            }
        ],
    )

    metrics = collect_metrics(summary_dir)

    assert set(metrics["family"]) == {"v3_rnn", "v2_frozen"}
    assert set(metrics["profile"]) == {"balanced"}
    assert metrics.loc[metrics["family"] == "v2_frozen", "final_fit_mode"].iloc[0] == "frozen"
    assert metrics.loc[
        metrics["temporal_type"] == "transformer_improved_frozen", "model_name"
    ].iloc[0] == "transformer_improved"


def test_build_materials_exports_tables_and_figures(tmp_path):
    root = tmp_path
    summary_dir = root / "reports" / "figures" / "summary"

    _write_csv(
        summary_dir
        / "train_rul_hybrid_v4_tcn"
        / "balanced"
        / "summary_metrics_train_rul_hybrid_v4_tcn_profilebalanced.csv",
        [
            {
                "mode": "balanced",
                "final_fit_mode": "frozen",
                "artifact_suffix": "v4",
                "window_size": 2048,
                "temporal_type": "tcn_frozen",
                "best_val_mse": 0.02,
                "test_mse": 0.03,
                "test_mae": 0.1,
                "test_rmse": 0.17,
                "test_r2": 0.4,
                "test_phm_score": 10.0,
                "inference_ms_per_sample": 0.05,
                "inference_samples_per_sec": 20000,
            },
            {
                "mode": "balanced",
                "final_fit_mode": "finetune",
                "artifact_suffix": "v4",
                "window_size": 2048,
                "temporal_type": "tcn_finetune",
                "best_val_mse": 0.01,
                "test_mse": 0.02,
                "test_mae": 0.1,
                "test_rmse": 0.14,
                "test_r2": 0.5,
                "test_phm_score": 9.0,
                "inference_ms_per_sample": 9.0,
                "inference_samples_per_sec": 111,
            },
        ],
    )

    outputs = build_materials(root=root, profile="balanced", skip_xlsx=False)
    best = select_best_by_family_mode(collect_metrics(summary_dir), profile="balanced")

    assert len(best) == 2
    assert root.joinpath("reports", "tables_for_vkr", "vkr_model_metrics_all.csv").exists()
    assert root.joinpath(
        "reports", "tables_for_vkr", "vkr_best_models_by_family_mode.md"
    ).exists()
    assert root.joinpath("reports", "tables_for_vkr", "vkr_model_metrics.xlsx").exists()
    assert root.joinpath(
        "reports", "figures", "summary", "figure_2_18_vkr_best_models_by_family_mode.png"
    ).exists()
    assert root.joinpath(
        "reports", "figures", "summary", "figure_2_19_vkr_accuracy_speed_tradeoff.png"
    ).exists()
    assert len(outputs) == 6
