import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

def test_plot():
    n_ex = 8
    n_cols = n_ex
    fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6))

    im = None
    for i in range(n_ex):
        ax_top = axes[0, i]
        ax_bot = axes[1, i]
        ax_top.imshow(np.random.rand(129, 9), aspect="auto", origin="lower", cmap="viridis")
        im = ax_bot.imshow(np.random.rand(129, 9), aspect="auto", origin="lower", cmap="jet")

    fig.suptitle("SHAP-анализ: вклад частотных полос", fontsize=13, fontweight="bold")

    fig.tight_layout()
    fig.subplots_adjust(top=0.90, right=0.92)

    # Вертикальный colorbar справа (manual placement to avoid overlap)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical", label="SHAP value")
    cbar.ax.tick_params(labelsize=8)

    os.makedirs("scratch", exist_ok=True)
    fig.savefig("scratch/shap_test.png", dpi=300, bbox_inches="tight")
    print("Saved to scratch/shap_test.png")

if __name__ == "__main__":
    test_plot()
