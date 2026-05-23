"""Run ResNet-18 inference on CWRU windows grouped by source .mat file.

The script does not train anything. It preserves the source-file id for every
window, reports the available file-level split, and writes a confusion matrix
plus scalar metrics for the trained classifier checkpoint.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import scipy.signal
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, Subset


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src" / "classification"))

from model import get_model  # noqa: E402


DATA_DIR = ROOT / "data" / "raw" / "CWRU"
CHECKPOINT = ROOT / "models" / "cnn" / "best_resnet18.pth"
OUT_DIR = ROOT / "reports" / "figures" / "summary"
TABLE_DIR = ROOT / "reports" / "tables_for_vkr"

FIG_PATH = OUT_DIR / "cwru_resnet18_confusion_matrix_file_grouped.png"
CM_CSV_PATH = TABLE_DIR / "cwru_resnet18_confusion_matrix_file_grouped.csv"
METRICS_PATH = TABLE_DIR / "cwru_resnet18_file_grouped_metrics.csv"
FILES_PATH = TABLE_DIR / "cwru_resnet18_file_grouped_files.csv"

WINDOW_SIZE = 1024
BATCH_SIZE = 64

CLASS_NAMES = [
    "Normal",
    "IR_007",
    "IR_014",
    "IR_021",
    "Ball_007",
    "Ball_014",
    "Ball_021",
    "OR_007",
    "OR_014",
    "OR_021",
]


@dataclass(frozen=True)
class SourceFile:
    path: Path
    rel_path: str
    label: int
    class_name: str


class CWRUFileWindowDataset(Dataset):
    """CWRU windows with file provenance retained for grouped evaluation."""

    def __init__(self, files: list[SourceFile], window_size: int = WINDOW_SIZE) -> None:
        self.files = files
        self.window_size = window_size
        self.spectrograms: list[np.ndarray] = []
        self.labels: list[int] = []
        self.source_files: list[str] = []
        self._load_data()

    def _load_data(self) -> None:
        for source in self.files:
            mat_data = scipy.io.loadmat(source.path)
            signal = None
            for key, value in mat_data.items():
                if not key.startswith("__") and key.endswith("_DE_time"):
                    signal = value.flatten()
                    break

            if signal is None:
                raise ValueError(f"No *_DE_time signal found in {source.path}")

            num_windows = len(signal) // self.window_size
            for i in range(num_windows):
                start_idx = i * self.window_size
                window = signal[start_idx : start_idx + self.window_size]
                _, _, zxx = scipy.signal.stft(window, nperseg=256, noverlap=128)
                self.spectrograms.append((np.abs(zxx) ** 2).astype(np.float32))
                self.labels.append(source.label)
                self.source_files.append(source.rel_path)

    def __len__(self) -> int:
        return len(self.spectrograms)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.spectrograms[idx], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def discover_source_files(data_dir: Path) -> list[SourceFile]:
    sources: list[SourceFile] = []
    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        if not class_dir.name or not class_dir.name[0].isdigit():
            continue
        label = int(class_dir.name[0])
        class_name = class_dir.name.split("_", 1)[1] if "_" in class_dir.name else class_dir.name
        for path in sorted(class_dir.glob("*.mat")):
            sources.append(
                SourceFile(
                    path=path,
                    rel_path=str(path.relative_to(data_dir)),
                    label=label,
                    class_name=class_name,
                )
            )
    return sources


def select_grouped_test_files(files: list[SourceFile]) -> tuple[list[SourceFile], str]:
    """Select file-level test groups.

    With the current repository data there is only one .mat file per class, so a
    separate unseen-file holdout cannot be constructed from local files. In that
    case all available files are evaluated as file groups and the limitation is
    written to the metrics table.
    """

    counts = pd.Series([f.label for f in files]).value_counts().to_dict()
    if all(counts.get(label, 0) >= 3 for label in range(len(CLASS_NAMES))):
        selected: list[SourceFile] = []
        for label in range(len(CLASS_NAMES)):
            class_files = [f for f in files if f.label == label]
            selected.extend(class_files[-3:])
        return selected, "held_out_last_3_files_per_class"

    return files, "all_available_files_one_file_per_class"


def load_model(device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    model_name = checkpoint.get("model_name", "resnet18")
    num_classes = int(checkpoint.get("num_classes", len(CLASS_NAMES)))
    model = get_model(model_name, num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int]]:
    labels: list[int] = []
    preds: list[int] = []
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs.to(device))
            preds.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
            labels.extend(targets.numpy().tolist())
    return labels, preds


def save_outputs(
    labels: list[int],
    preds: list[int],
    dataset: CWRUFileWindowDataset,
    selected_files: list[SourceFile],
    split_mode: str,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASS_NAMES))))
    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100
    annot = np.empty_like(cm, dtype=object)
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            annot[row, col] = f"{cm[row, col]}\n{cm_pct[row, col]:.1f}%" if cm[row, col] else ""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
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
        cbar_kws={"label": "Количество окон"},
        ax=ax,
    )
    ax.set_title(f"ResNet-18: grouped inference по .mat-файлам\nAccuracy={accuracy:.3f}, Macro-F1={macro_f1:.3f}")
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(CM_CSV_PATH, encoding="utf-8")

    file_count_by_class = pd.Series([f.label for f in selected_files]).value_counts().sort_index()
    limitation = (
        "Only one .mat file per class is available locally; the saved checkpoint "
        "was trained with a segment-level split, so this is not an unseen-file "
        "holdout metric."
        if split_mode == "all_available_files_one_file_per_class"
        else "File-level holdout constructed from available source files."
    )
    pd.DataFrame(
        [
            {
                "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
                "data_dir": str(DATA_DIR.relative_to(ROOT)),
                "split_mode": split_mode,
                "source_files": len(selected_files),
                "test_windows": len(labels),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "limitation": limitation,
            }
        ]
    ).to_csv(METRICS_PATH, index=False, encoding="utf-8")

    file_rows = []
    for source in selected_files:
        file_rows.append(
            {
                "split": "test",
                "label": source.label,
                "class_name": CLASS_NAMES[source.label],
                "source_file": source.rel_path,
                "files_in_class": int(file_count_by_class.get(source.label, 0)),
                "windows": dataset.source_files.count(source.rel_path),
            }
        )
    pd.DataFrame(file_rows).to_csv(FILES_PATH, index=False, encoding="utf-8")


def main() -> None:
    files = discover_source_files(DATA_DIR)
    if not files:
        raise RuntimeError(f"No CWRU .mat files found in {DATA_DIR}")

    selected_files, split_mode = select_grouped_test_files(files)
    dataset = CWRUFileWindowDataset(selected_files)
    if len(dataset) == 0:
        raise RuntimeError("No windows were produced from selected CWRU files")

    loader = DataLoader(Subset(dataset, list(range(len(dataset)))), batch_size=BATCH_SIZE, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    labels, preds = predict(model, loader, device)
    save_outputs(labels, preds, dataset, selected_files, split_mode)

    print(f"Split mode: {split_mode}")
    print(f"Source files: {len(selected_files)}")
    print(f"Test windows: {len(labels)}")
    print(f"Accuracy: {accuracy_score(labels, preds):.6f}")
    print(f"Macro-F1: {f1_score(labels, preds, average='macro'):.6f}")
    print(f"Saved: {FIG_PATH}")
    print(f"Saved: {CM_CSV_PATH}")
    print(f"Saved: {METRICS_PATH}")
    print(f"Saved: {FILES_PATH}")


if __name__ == "__main__":
    main()
