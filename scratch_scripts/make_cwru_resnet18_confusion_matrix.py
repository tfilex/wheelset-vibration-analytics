"""Generate a clean 10x10 confusion matrix for ResNet-18 on CWRU test split."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src" / "classification"))

from data_loader import CWRUDataset  # noqa: E402
from model import get_model  # noqa: E402


RANDOM_SEED = 13
DATA_DIR = ROOT / "data" / "raw" / "CWRU"
CHECKPOINT = ROOT / "models" / "demo_best" / "classification" / "cwru_classifier.pth"
OUT_DIR = ROOT / "reports" / "figures" / "summary"
TABLE_DIR = ROOT / "reports" / "tables_for_vkr"
FIG_PATH = OUT_DIR / "cwru_resnet18_confusion_matrix_10x10.png"
CSV_PATH = TABLE_DIR / "cwru_resnet18_confusion_matrix_10x10.csv"
METRICS_PATH = TABLE_DIR / "cwru_resnet18_confusion_matrix_metrics.csv"

CLASS_NAMES = [
    "Normal",
    "IR_007", "IR_014", "IR_021",
    "Ball_007", "Ball_014", "Ball_021",
    "OR_007", "OR_014", "OR_021",
]


def load_test_subset() -> Subset:
    dataset = CWRUDataset(data_dir=str(DATA_DIR))
    labels = dataset.labels
    indices = list(range(len(dataset)))
    train_idx, temp_idx, _, temp_labels = train_test_split(
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
    return Subset(dataset, test_idx)


def load_model(device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    model_name = checkpoint.get("model_name", "resnet18")
    num_classes = int(checkpoint.get("num_classes", 10))
    model = get_model(model_name, num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int]]:
    preds: list[int] = []
    labels: list[int] = []
    with torch.no_grad():
        for inputs, target in loader:
            inputs = inputs.to(device)
            output = model(inputs)
            pred = output.argmax(dim=1).cpu().numpy().tolist()
            preds.extend(pred)
            labels.extend(target.numpy().tolist())
    return labels, preds


def save_confusion_matrix(labels: list[int], preds: list[int]) -> None:
    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASS_NAMES))))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n{cm_pct[i, j]:.1f}%" if cm[i, j] else ""

    accuracy = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig, ax = plt.subplots(figsize=(12, 10), facecolor="white")
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Количество образцов"},
        ax=ax,
    )
    ax.set_title(f"Матрица ошибок ResNet-18 на тестовой выборке CWRU\nAccuracy={accuracy:.3f}, Macro-F1={f1_macro:.3f}")
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(CSV_PATH, encoding="utf-8")
    pd.DataFrame(
        [{"test_samples": len(labels), "accuracy": accuracy, "macro_f1": f1_macro}]
    ).to_csv(METRICS_PATH, index=False)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_subset = load_test_subset()
    loader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=0)
    model = load_model(device)
    labels, preds = predict(model, loader, device)
    save_confusion_matrix(labels, preds)
    print(f"Saved: {FIG_PATH}")
    print(f"Saved: {CSV_PATH}")
    print(f"Saved: {METRICS_PATH}")


if __name__ == "__main__":
    main()
