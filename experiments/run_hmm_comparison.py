"""Train and evaluate an HMM baseline for XJTU-SY RUL prediction."""

from __future__ import annotations

from pathlib import Path
import re
import sys

import mlflow
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.stat_features import extract_stat_features  # noqa: E402
from src.models.hmm_baseline import HMMRULPredictor  # noqa: E402


DATA_ROOT = PROJECT_ROOT / "data/raw/XJTU-SY"
RESULT_PATH = PROJECT_ROOT / "results/hmm_vs_nn_comparison.csv"
WINDOW_SIZE = 2048
MAX_FILES_PER_BEARING = 160
TRAIN_BEARINGS = (
    "35Hz12kN/Bearing1_1",
    "35Hz12kN/Bearing1_2",
    "37.5Hz11kN/Bearing2_1",
    "37.5Hz11kN/Bearing2_2",
    "40Hz10kN/Bearing3_1",
    "40Hz10kN/Bearing3_2",
)
TEST_BEARINGS = (
    "35Hz12kN/Bearing1_3",
    "35Hz12kN/Bearing1_4",
    "37.5Hz11kN/Bearing2_5",
    "40Hz10kN/Bearing3_3",
)


def natural_key(path: Path) -> int:
    digits = re.sub(r"\D", "", path.name)
    return int(digits) if digits else 0


def bearing_files(bearing_dir: Path) -> list[Path]:
    files = sorted(bearing_dir.glob("*.csv"), key=natural_key)
    if len(files) <= MAX_FILES_PER_BEARING:
        return files
    indices = np.linspace(0, len(files) - 1, MAX_FILES_PER_BEARING, dtype=int)
    return [files[int(index)] for index in indices]


def file_features(csv_path: Path) -> dict[str, float]:
    df = pd.read_csv(csv_path)
    h_signal = df.iloc[:WINDOW_SIZE, 0].to_numpy(dtype=float)
    v_signal = df.iloc[:WINDOW_SIZE, 1].to_numpy(dtype=float)
    h_features = {f"h_{key}": value for key, value in extract_stat_features(h_signal).items()}
    v_features = {f"v_{key}": value for key, value in extract_stat_features(v_signal).items()}
    return {**h_features, **v_features}


def load_bearing_sequence(relative_dir: str) -> tuple[np.ndarray, np.ndarray]:
    bearing_dir = DATA_ROOT / relative_dir
    files = bearing_files(bearing_dir)
    if not files:
        raise FileNotFoundError(f"No CSV files found in {bearing_dir}")

    rows = [file_features(path) for path in files]
    features = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(features) <= 1:
        true_rul = np.array([0.0], dtype=float)
    else:
        true_rul = np.linspace(1.0, 0.0, len(features), dtype=float)
    return features.to_numpy(dtype=float), true_rul


def load_sequences(relative_dirs: tuple[str, ...]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    sequences: list[np.ndarray] = []
    ruls: list[np.ndarray] = []
    for relative_dir in relative_dirs:
        features, true_rul = load_bearing_sequence(relative_dir)
        sequences.append(features)
        ruls.append(true_rul)
    return sequences, ruls


def scale_sequences(
    train_sequences: list[np.ndarray],
    test_sequences: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    scaler = StandardScaler()
    scaler.fit(np.vstack(train_sequences))
    train_scaled = [scaler.transform(sequence) for sequence in train_sequences]
    test_scaled = [scaler.transform(sequence) for sequence in test_sequences]
    return train_scaled, test_scaled


def main() -> None:
    train_sequences, _ = load_sequences(TRAIN_BEARINGS)
    test_sequences, test_ruls = load_sequences(TEST_BEARINGS)
    train_scaled, test_scaled = scale_sequences(train_sequences, test_sequences)

    predictor = HMMRULPredictor(n_components=4, covariance_type="full", n_iter=200)
    predictor.fit(train_scaled)
    metrics = predictor.evaluate(test_scaled, test_ruls)

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(
        [
            {
                "model": "HMM (K=4)",
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
                "notes": "statistical features baseline",
            },
            {
                "model": "ImprovedTransformer (из ВКР)",
                "rmse": np.nan,
                "mae": np.nan,
                "r2": 0.105,
                "notes": "reference value from saved VKR materials",
            },
            {
                "model": "Ensemble ImprovedTransformer+Conformer+BiTCN (из ВКР)",
                "rmse": np.nan,
                "mae": np.nan,
                "r2": 0.121,
                "notes": "reference value from saved VKR materials",
            },
        ]
    )
    comparison.to_csv(RESULT_PATH, index=False)

    mlflow.set_experiment("HMM_RUL_baseline")
    with mlflow.start_run(run_name="HMM_baseline_K4"):
        mlflow.log_params(
            {
                "n_components": predictor.n_components,
                "covariance_type": predictor.covariance_type,
                "window_size": WINDOW_SIZE,
                "max_files_per_bearing": MAX_FILES_PER_BEARING,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(RESULT_PATH))

    print(comparison.to_string(index=False))
    print(f"Saved: {RESULT_PATH}")


if __name__ == "__main__":
    main()

