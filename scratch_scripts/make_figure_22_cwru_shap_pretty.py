"""Create report-ready SHAP maps for four CWRU classes."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src" / "classification"))

from data_loader import CWRUDataset  # noqa: E402
from model import get_model  # noqa: E402


DATA_DIR = ROOT / "data" / "raw" / "CWRU"
CHECKPOINT = ROOT / "models" / "cnn" / "best_resnet18.pth"
OUT_DIR = ROOT / "reports" / "figures" / "summary"
OUT_PATH = OUT_DIR / "figure_22_cwru_resnet18_shap_four_classes_pretty.png"

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

EXPLAIN_CLASSES = [
    (0, "а) Исправное состояние"),
    (1, "б) Дефект внутреннего кольца"),
    (4, "в) Дефект тела качения"),
    (7, "г) Дефект наружного кольца"),
]


def load_model(device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    model = get_model(
        checkpoint.get("model_name", "resnet18"),
        num_classes=int(checkpoint.get("num_classes", len(CLASS_NAMES))),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def select_indices(dataset: CWRUDataset) -> tuple[list[int], list[int]]:
    labels = np.asarray(dataset.labels)
    explain_indices: list[int] = []
    for label, _ in EXPLAIN_CLASSES:
        class_indices = np.flatnonzero(labels == label)
        if len(class_indices) == 0:
            raise RuntimeError(f"No samples for class {label}")
        explain_indices.append(int(class_indices[len(class_indices) // 2]))

    background_indices: list[int] = []
    for label in range(len(CLASS_NAMES)):
        class_indices = np.flatnonzero(labels == label)
        if len(class_indices) == 0:
            continue
        picks = np.linspace(0, len(class_indices) - 1, min(2, len(class_indices)), dtype=int)
        background_indices.extend(int(class_indices[i]) for i in picks)
    return background_indices, explain_indices


def stack_dataset_items(dataset: CWRUDataset, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for idx in indices:
        x, y = dataset[idx]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)


def normalize_shap_values(raw_values: object) -> np.ndarray:
    if isinstance(raw_values, list):
        shap_all = np.stack(raw_values, axis=-1)
    else:
        shap_all = raw_values
    if isinstance(shap_all, torch.Tensor):
        shap_all = shap_all.detach().cpu().numpy()
    return np.asarray(shap_all)


def main() -> None:
    import shap

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CWRUDataset(data_dir=str(DATA_DIR))
    background_indices, explain_indices = select_indices(dataset)
    background, _ = stack_dataset_items(dataset, background_indices)
    explain_samples, explain_labels = stack_dataset_items(dataset, explain_indices)

    model = load_model(device)
    with torch.no_grad():
        logits = model(explain_samples.to(device))
        pred_labels = logits.argmax(dim=1).cpu().numpy()
        pred_probs = torch.softmax(logits, dim=1).cpu().numpy()

    explainer = shap.GradientExplainer(model, background.to(device))
    shap_raw = explainer.shap_values(explain_samples.to(device))
    shap_all = normalize_shap_values(shap_raw)

    explain_np = explain_samples.cpu().numpy()
    labels_np = explain_labels.cpu().numpy().astype(int)

    shap_maps: list[np.ndarray] = []
    specs: list[np.ndarray] = []
    for i, true_label in enumerate(labels_np):
        specs.append(explain_np[i, 0])
        if shap_all.ndim == 5:
            shap_map = shap_all[i, 0, :, :, true_label]
        elif shap_all.ndim == 4:
            shap_map = shap_all[i, 0]
        else:
            raise RuntimeError(f"Unexpected SHAP shape: {shap_all.shape}")
        shap_maps.append(shap_map)

    spec_values = np.concatenate([np.log1p(spec).ravel() for spec in specs])
    spec_vmin, spec_vmax = np.percentile(spec_values, [2, 99.5])
    shap_abs = np.concatenate([np.abs(m).ravel() for m in shap_maps])
    shap_vmax = float(np.percentile(shap_abs, 99))
    if shap_vmax <= 0:
        shap_vmax = float(np.max(shap_abs) or 1.0)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    fig = plt.figure(figsize=(14.2, 6.7), facecolor="white")
    gs = GridSpec(2, 5, width_ratios=[1, 1, 1, 1, 0.055], height_ratios=[1, 1], hspace=0.32, wspace=0.22)

    spec_im = None
    shap_im = None
    for col, ((true_label, panel_title), spec, shap_map) in enumerate(zip(EXPLAIN_CLASSES, specs, shap_maps)):
        ax_spec = fig.add_subplot(gs[0, col])
        spec_log = np.log1p(spec)
        spec_im = ax_spec.imshow(
            spec_log,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=spec_vmin,
            vmax=spec_vmax,
            interpolation="nearest",
        )
        prob = pred_probs[col, pred_labels[col]]
        ax_spec.set_title(f"{panel_title}\nпрогноз: {CLASS_NAMES[pred_labels[col]]}, p={prob:.2f}")
        ax_spec.set_xlabel("Временной бин")
        if col == 0:
            ax_spec.set_ylabel("Частотный бин")
        else:
            ax_spec.set_yticklabels([])

        ax_shap = fig.add_subplot(gs[1, col])
        shap_im = ax_shap.imshow(
            shap_map,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-shap_vmax,
            vmax=shap_vmax,
            interpolation="nearest",
        )
        ax_shap.contour(
            np.log1p(spec),
            levels=4,
            colors="black",
            linewidths=0.35,
            alpha=0.22,
        )
        ax_shap.set_xlabel("Временной бин")
        if col == 0:
            ax_shap.set_ylabel("Частотный бин")
        else:
            ax_shap.set_yticklabels([])

        for ax in (ax_spec, ax_shap):
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(length=2)

    ax_spec_cbar = fig.add_subplot(gs[0, 4])
    if spec_im is not None:
        cbar = fig.colorbar(spec_im, cax=ax_spec_cbar)
        cbar.set_label("log-мощность STFT")
        cbar.outline.set_visible(False)

    ax_shap_cbar = fig.add_subplot(gs[1, 4])
    if shap_im is not None:
        cbar = fig.colorbar(shap_im, cax=ax_shap_cbar)
        cbar.set_label("SHAP value")
        cbar.outline.set_visible(False)

    fig.text(0.012, 0.72, "Входная STFT-спектрограмма", rotation=90, va="center", ha="center", fontsize=11, color="#374151")
    fig.text(0.012, 0.28, "SHAP-карта истинного класса", rotation=90, va="center", ha="center", fontsize=11, color="#374151")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.055, right=0.93, top=0.94, bottom=0.10)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")
    print(f"SHAP shape: {shap_all.shape}")
    print("Predictions:", [CLASS_NAMES[i] for i in pred_labels.tolist()])


if __name__ == "__main__":
    main()
