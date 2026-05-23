"""Generate One-vs-Rest ROC-AUC curves for the saved CWRU classifier."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.data_loader import CWRUDataset  # noqa: E402
from src.classification.model import get_model  # noqa: E402
from src.evaluation.roc_analysis import plot_roc_ovr  # noqa: E402


RANDOM_SEED = 13
DATA_DIR = PROJECT_ROOT / "data/raw/CWRU"
CLASS_NAMES = [
    "Normal",
    "InnerRace_007",
    "InnerRace_014",
    "InnerRace_021",
    "Ball_007",
    "Ball_014",
    "Ball_021",
    "OuterRace_007",
    "OuterRace_014",
    "OuterRace_021",
]


def find_classifier_checkpoint() -> Path:
    """Return the best available classification checkpoint."""
    preferred = [
        PROJECT_ROOT / "models/demo_best/classification/cwru_classifier.pth",
        PROJECT_ROOT / "models/demo_best/cwru_classifier.pth",
        PROJECT_ROOT / "models/cnn/best_resnet18.pth",
    ]
    existing = [path for path in preferred if path.exists()]
    if existing:
        return existing[0]

    candidates = [
        path
        for path in (PROJECT_ROOT / "models/cnn").glob("*.pth")
        if "rul" not in path.name.lower()
    ]
    if not candidates:
        raise FileNotFoundError("No CWRU classifier checkpoint found")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def build_test_loader(batch_size: int = 128) -> DataLoader:
    dataset = CWRUDataset(data_dir=str(DATA_DIR))
    if len(dataset) == 0:
        raise RuntimeError(f"CWRU dataset is empty: {DATA_DIR}")

    labels = dataset.labels
    indices = list(range(len(dataset)))
    _, temp_idx, _, temp_labels = train_test_split(
        indices,
        labels,
        train_size=0.70,
        stratify=labels,
        random_state=RANDOM_SEED,
    )
    _, test_idx, _, _ = train_test_split(
        temp_idx,
        temp_labels,
        test_size=0.5,
        stratify=temp_labels,
        random_state=RANDOM_SEED,
    )
    return DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)


def load_classifier(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("model_name", "resnet18")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model = get_model(model_name, num_classes=len(CLASS_NAMES), in_channels=1)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def collect_probabilities(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    y_true: list[np.ndarray] = []
    y_prob: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, labels in loader:
            logits = model(inputs.to(device))
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            y_prob.append(probabilities)
            y_true.append(labels.numpy())
    return np.concatenate(y_true), np.vstack(y_prob)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = find_classifier_checkpoint()
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")

    loader = build_test_loader()
    model = load_classifier(checkpoint_path, device)
    y_true, y_prob = collect_probabilities(model, loader, device)
    auc_values = plot_roc_ovr(
        y_true,
        y_prob,
        CLASS_NAMES,
        save_path=PROJECT_ROOT / "figures/roc_auc_ovr.png",
    )

    print(f"{'Класс':<24} AUC")
    for class_name in CLASS_NAMES:
        value = auc_values[class_name]
        text_value = "nan" if np.isnan(value) else f"{value:.4f}"
        print(f"{class_name:<24} {text_value}")
    print(f"{'Macro-average':<24} {auc_values['macro_auc']:.4f}")


if __name__ == "__main__":
    main()

