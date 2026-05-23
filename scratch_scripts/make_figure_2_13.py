"""
Генерирует figure_2_13_v5_odd_optuna_param_importance.png
в стиле figure_2_10 (два субграфика в рамках, жирный заголовок).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import numpy as np
import os

CONFORMER_PATH = (
    "/home/ish/rudn/VKR/reports/figures/summary/"
    "train_rul_hybrid_v5_odd/balanced/ws2048/"
    "conformer_train_rul_hybrid_v5_odd_profilebalanced"
    "_trials30_epochs25_featurecache_on_ws2048_importance.png"
)
PATCHTST_PATH = (
    "/home/ish/rudn/VKR/reports/figures/summary/"
    "train_rul_hybrid_v5_odd/balanced/ws2048/"
    "patchtst_train_rul_hybrid_v5_odd_profilebalanced"
    "_trials30_epochs25_featurecache_on_ws2048_importance.png"
)
OUT_PATH = (
    "/home/ish/rudn/VKR/reports/figures/summary/"
    "figure_2_13_v5_odd_optuna_param_importance.png"
)

img_conformer = np.array(Image.open(CONFORMER_PATH).convert("RGB"))
img_patchtst  = np.array(Image.open(PATCHTST_PATH).convert("RGB"))

fig = plt.figure(figsize=(18, 13), facecolor="white")

# Main title (как в figure_2_10)
fig.suptitle(
    "Optuna v5_odd: parameter importance (Conformer и PatchTST)",
    fontsize=22,
    fontweight="bold",
    y=0.98,
)

# --- Panel A: Conformer ---
ax1 = fig.add_axes([0.03, 0.08, 0.45, 0.84])  # [left, bottom, width, height]
ax1.imshow(img_conformer)
ax1.axis("off")

# Рамка (как в figure_2_10)
for spine in ["top", "bottom", "left", "right"]:
    ax1.spines[spine].set_visible(True)
    ax1.spines[spine].set_linewidth(1.2)
    ax1.spines[spine].set_color("#aaaaaa")

# Подпись панели в стиле figure_2_10
ax1.text(
    0.01, 1.04, "A. Conformer — parameter importance",
    transform=ax1.transAxes,
    fontsize=14, fontweight="bold", va="bottom", ha="left",
)
ax1.text(
    0.01, 1.00, "Conformer, v5_odd balanced, ws=2048, 30 trials; HPO по nhead, lr, hidden_size и др.",
    transform=ax1.transAxes,
    fontsize=10, va="bottom", ha="left", color="#555555",
)

# --- Panel B: PatchTST ---
ax2 = fig.add_axes([0.52, 0.08, 0.45, 0.84])
ax2.imshow(img_patchtst)
ax2.axis("off")

for spine in ["top", "bottom", "left", "right"]:
    ax2.spines[spine].set_visible(True)
    ax2.spines[spine].set_linewidth(1.2)
    ax2.spines[spine].set_color("#aaaaaa")

ax2.text(
    0.01, 1.04, "B. PatchTST — parameter importance",
    transform=ax2.transAxes,
    fontsize=14, fontweight="bold", va="bottom", ha="left",
)
ax2.text(
    0.01, 1.00, "PatchTST, v5_odd balanced, ws=2048, 30 trials; HPO по patch_size, nhead, lr и др.",
    transform=ax2.transAxes,
    fontsize=10, va="bottom", ha="left", color="#555555",
)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Сохранено: {OUT_PATH}")
