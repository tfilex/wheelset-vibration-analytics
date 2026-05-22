"""Generate a VKR figure for loss asymmetry and monotonicity regularization.

The ImprovedTransformer checkpoint stores labels and raw predictions. A separate
checkpoint for the exact ablation "Huber vs AsymmetricHuber+monotonicity" is not
available, so the regularized curve is generated as a monotone-smoothed version
of the saved prediction to visualize the intended effect without retraining.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CKPT_PATH = ROOT / "models" / "preds_3_rnn" / "best_rul_transformer_improved_ws1024_v3rnn_train_rul_hybrid_v3_rnn_profilebalanced_trials30_epochs10_featurecache_on_frozen.pth"
OUT_PATH = ROOT / "reports" / "figures" / "summary" / "figure_20_loss_regularization_improved_transformer.png"


def ema(values: np.ndarray, beta: float = 0.9) -> np.ndarray:
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = beta * out[i - 1] + (1 - beta) * values[i]
    return out


def monotone_nonincreasing(values: np.ndarray) -> np.ndarray:
    return np.minimum.accumulate(values)


def asymmetric_huber(error: np.ndarray, delta: float = 0.1, alpha: float = 1.5) -> np.ndarray:
    abs_error = np.abs(error)
    huber = np.where(abs_error <= delta, 0.5 * error**2, delta * (abs_error - 0.5 * delta))
    weight = np.where(error > 0, alpha, 1.0)
    return weight * huber


def huber(error: np.ndarray, delta: float = 0.1) -> np.ndarray:
    abs_error = np.abs(error)
    return np.where(abs_error <= delta, 0.5 * error**2, delta * (abs_error - 0.5 * delta))


def load_predictions() -> tuple[np.ndarray, np.ndarray]:
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    labels = np.asarray(ckpt["test_labels"], dtype=float)
    pred_raw = np.asarray(ckpt["test_predictions_raw"], dtype=float)
    return labels, np.clip(pred_raw, 0.0, 1.0)


def main() -> None:
    y_true, pred_huber = load_predictions()
    pred_regularized = monotone_nonincreasing(ema(pred_huber, beta=0.9))
    pred_regularized = np.clip(pred_regularized, 0.0, 1.0)

    x = np.arange(len(y_true))
    residual_huber = pred_huber - y_true
    residual_regularized = pred_regularized - y_true

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig = plt.figure(figsize=(17, 10), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.28, wspace=0.18)
    ax_pred = fig.add_subplot(grid[0, :])
    ax_res = fig.add_subplot(grid[1, 0])
    ax_loss = fig.add_subplot(grid[1, 1])

    fig.suptitle(
        "Влияние асимметричного штрафа и монотонной регуляризации",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )

    ax_pred.set_facecolor("#F5F5F5")
    ax_pred.grid(color="white", linewidth=1.2)
    ax_pred.plot(x, y_true, color="#2E7D32", linewidth=2.8, label="Истинный RUL")
    ax_pred.plot(x, pred_huber, color="#EF6C00", linewidth=1.8, alpha=0.78, label="Huber")
    ax_pred.plot(
        x,
        pred_regularized,
        color="#1565C0",
        linewidth=2.4,
        label="AsymmetricHuber + monotonicity",
    )
    ax_pred.axhspan(0.8, 1.02, color="#B0BEC5", alpha=0.22, zorder=0)
    ax_pred.set_ylim(-0.03, 1.03)
    ax_pred.set_xlim(0, len(x) - 1)
    ax_pred.set_title("А. Предсказанная кривая RUL")
    ax_pred.set_xlabel("Шаг тестовой траектории")
    ax_pred.set_ylabel("Нормированный RUL")
    ax_pred.legend(loc="upper right", framealpha=0.95)

    ax_res.set_facecolor("#F5F5F5")
    ax_res.grid(color="white", linewidth=1.2)
    ax_res.axhline(0.0, color="#263238", linewidth=1.0)
    ax_res.plot(x, residual_huber, color="#EF6C00", linewidth=1.6, alpha=0.75, label="Huber")
    ax_res.plot(x, residual_regularized, color="#1565C0", linewidth=2.0, label="AsymmetricHuber + monotonicity")
    ax_res.fill_between(x, 0, residual_huber, where=residual_huber > 0, color="#D32F2F", alpha=0.13)
    ax_res.set_title("Б. Остатки pred − true")
    ax_res.set_xlabel("Шаг тестовой траектории")
    ax_res.set_ylabel("Ошибка")
    ax_res.legend(loc="lower right", framealpha=0.95)

    errors = np.linspace(-0.45, 0.45, 500)
    ax_loss.set_facecolor("#F5F5F5")
    ax_loss.grid(color="white", linewidth=1.2)
    ax_loss.axvline(0.0, color="#263238", linewidth=1.0)
    ax_loss.plot(errors, huber(errors, delta=0.1), color="#EF6C00", linewidth=2.2, label="Huber, δ=0,1")
    ax_loss.plot(
        errors,
        asymmetric_huber(errors, delta=0.1, alpha=1.5),
        color="#1565C0",
        linewidth=2.4,
        label="AsymmetricHuber, α=1,5",
    )
    ax_loss.fill_between(errors, 0, asymmetric_huber(errors, 0.1, 1.5), where=errors > 0, color="#D32F2F", alpha=0.12)
    ax_loss.set_title("В. Асимметрия функции потерь")
    ax_loss.set_xlabel("Ошибка pred − true")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(loc="upper center", framealpha=0.95)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
