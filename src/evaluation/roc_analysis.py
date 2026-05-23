"""One-vs-Rest ROC-AUC utilities for CWRU classification."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize


def compute_roc_ovr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
) -> dict[str, float]:
    """Compute per-class and macro ROC-AUC in One-vs-Rest mode.

    Classes absent from ``y_true`` are reported as ``nan`` and skipped in the
    fallback macro average.
    """
    labels = np.arange(len(class_names))
    y_true_arr = np.asarray(y_true, dtype=int).reshape(-1)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    if y_prob_arr.ndim != 2 or y_prob_arr.shape[1] != len(class_names):
        raise ValueError("y_prob must have shape (n_samples, n_classes)")

    y_bin = label_binarize(y_true_arr, classes=labels)
    result: dict[str, float] = {}
    valid_auc_values: list[float] = []
    for class_idx, class_name in enumerate(class_names):
        positives = int(y_bin[:, class_idx].sum())
        negatives = int(y_bin.shape[0] - positives)
        if positives == 0 or negatives == 0:
            result[class_name] = float("nan")
            continue
        class_auc = float(roc_auc_score(y_bin[:, class_idx], y_prob_arr[:, class_idx]))
        result[class_name] = class_auc
        valid_auc_values.append(class_auc)

    try:
        macro_auc = float(
            roc_auc_score(
                y_true_arr,
                y_prob_arr,
                labels=labels,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        macro_auc = float("nan")
    if not np.isfinite(macro_auc):
        macro_auc = float(np.nanmean(valid_auc_values)) if valid_auc_values else float("nan")
    result["macro_auc"] = macro_auc
    return result


def plot_roc_ovr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    save_path: str | Path = "figures/roc_auc_ovr.png",
) -> dict[str, float]:
    """Plot One-vs-Rest ROC curves and save them to disk."""
    labels = np.arange(len(class_names))
    y_true_arr = np.asarray(y_true, dtype=int).reshape(-1)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    y_bin = label_binarize(y_true_arr, classes=labels)
    auc_values = compute_roc_ovr(y_true_arr, y_prob_arr, class_names)

    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    for class_idx, class_name in enumerate(class_names):
        positives = int(y_bin[:, class_idx].sum())
        negatives = int(y_bin.shape[0] - positives)
        if positives == 0 or negatives == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, class_idx], y_prob_arr[:, class_idx])
        class_auc = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            lw=1.6,
            color=colors[class_idx],
            label=f"{class_name} (AUC={class_auc:.4f})",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1.0, label="Случайный классификатор")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC-кривые One-vs-Rest, Macro-AUC={auc_values['macro_auc']:.4f}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return auc_values

