"""
Генерирует три рисунка:
  figure_2_15 — summary bar plot test MSE по всем конфигурациям
  figure_2_16 — Pareto front: test MSE vs inference ms/sample
  figure_2_17 — эволюция best test MSE по поколениям
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ─── ДАННЫЕ ────────────────────────────────────────────────────────────────────
# Все конфигурации (family, label, mode, test_mse, inference_ms)
# Собраны из balanced CSV-файлов (trials30)

# (family, label, mode, test_mse, inference_ms_per_sample)
# Только реальные данные из balanced CSV-файлов (trials30)
CONFIGS = [
    # v3_rnn — trials30, epochs10 (лучший прогон с transformer_improved)
    ("v3_rnn", "transf_impr_frozen",  "frozen",   0.009228, 0.174957),
    ("v3_rnn", "transf_impr_finetune","finetune", 0.011698, 4.089880),
    # v3_rnn — trials30, epochs30 (bilstm)
    ("v3_rnn", "bilstm",              "finetune", 0.010332, 8.423200),
    # v3_rnn — trials10, epochs10 (lstm_attn)
    ("v3_rnn", "lstm_attn_frozen",    "frozen",   0.024411, 0.072417),
    ("v3_rnn", "lstm_attn_finetune",  "finetune", 0.031424, 4.065372),
    # v4_tcn — trials30, epochs25
    ("v4_tcn", "tcn_bi_frozen",       "frozen",   0.013069, 0.051670),
    ("v4_tcn", "tcn_bi_finetune",     "finetune", 0.013021, 9.176849),
    # v5_odd — trials30, epochs25 (conformer)
    ("v5_odd", "conformer_frozen",    "frozen",   0.013271, 0.111422),
    ("v5_odd", "conformer_finetune",  "finetune", 0.012969, 10.437634),
]

# Поколения для Рисунка 2.17
GENERATIONS = [
    ("pred_0\n(CatBoost,\nLSTM base)", 0.0390),
    ("preds_2\n(v2 frozen)", 0.0248),
    ("preds_3\n(v3 balanced)", 0.0180),
    ("preds_3_rnn\n(v3_rnn best)", 0.009228),
    ("preds_4_tcn\n(v4_tcn best)", 0.013021),
    ("preds_5_odd\n(v5_odd best)", 0.012969),
]

FAMILY_COLORS = {
    "v3_rnn": "#2196F3",   # blue
    "v4_tcn": "#FF9800",   # orange
    "v5_odd": "#4CAF50",   # green
}
MODE_HATCH = {
    "frozen":   "",
    "finetune": "//",
}

OUT_DIR = "/home/ish/rudn/VKR/reports/figures/summary"
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Рисунок 2.15 — Summary bar plot
# ═══════════════════════════════════════════════════════════════════════════════
configs_sorted = sorted(CONFIGS, key=lambda x: x[3])  # сортировка по test_mse
labels   = [c[1] for c in configs_sorted]
mse_vals = [c[3] for c in configs_sorted]
families = [c[0] for c in configs_sorted]
modes    = [c[2] for c in configs_sorted]

fig15, ax = plt.subplots(figsize=(16, 7), facecolor="white")
fig15.patch.set_facecolor("white")

ax.set_facecolor("#f5f5f5")
ax.grid(axis="x", color="white", linewidth=1.4, zorder=0)

bars = ax.barh(
    range(len(labels)),
    mse_vals,
    color=[FAMILY_COLORS[f] for f in families],
    hatch=[MODE_HATCH[m] for m in modes],
    edgecolor="white",
    linewidth=0.8,
    height=0.7,
    zorder=3,
)

# Метки значений
for i, v in enumerate(mse_vals):
    ax.text(v + 0.0002, i, f"{v:.4f}", va="center", fontsize=9.5, color="#333")

ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=10.5)
ax.set_xlabel("Test MSE", fontsize=12)
ax.invert_yaxis()

# Легенда: семейства
legend_handles = [
    mpatches.Patch(facecolor=FAMILY_COLORS[k], label=k, edgecolor="white")
    for k in ["v3_rnn", "v4_tcn", "v5_odd"]
]
# Легенда: режим
legend_handles += [
    mpatches.Patch(facecolor="#888888", hatch="",   label="frozen",   edgecolor="white"),
    mpatches.Patch(facecolor="#888888", hatch="//", label="finetune", edgecolor="white"),
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=10, framealpha=0.9)

fig15.suptitle(
    "Summary: test MSE по всем конфигурациям (v3_rnn, v4_tcn, v5_odd)",
    fontsize=17, fontweight="bold", y=1.01,
)
ax.set_title(
    "Balanced profile, trials=30; режимы frozen и finetune; сортировка по возрастанию MSE",
    fontsize=11, color="#555", pad=6,
)

plt.tight_layout()
out15 = os.path.join(OUT_DIR, "figure_2_15_summary_bar_test_mse_all_families.png")
fig15.savefig(out15, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig15)
print(f"Saved: {out15}")


# ═══════════════════════════════════════════════════════════════════════════════
# Рисунок 2.16 — Pareto front
# ═══════════════════════════════════════════════════════════════════════════════
def is_pareto_optimal(mse_list, ms_list):
    """Возвращает маску Pareto-оптимальных точек (min MSE, min ms)."""
    n = len(mse_list)
    pareto = [True] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if mse_list[j] <= mse_list[i] and ms_list[j] <= ms_list[i]:
                if mse_list[j] < mse_list[i] or ms_list[j] < ms_list[i]:
                    pareto[i] = False
                    break
    return pareto

mse_all = [c[3] for c in CONFIGS]
ms_all  = [c[4] for c in CONFIGS]
pareto  = is_pareto_optimal(mse_all, ms_all)

fig16, ax = plt.subplots(figsize=(12, 7), facecolor="white")
fig16.patch.set_facecolor("white")
ax.set_facecolor("#f5f5f5")
ax.grid(color="white", linewidth=1.2, zorder=0)

for i, c in enumerate(CONFIGS):
    fam, lbl, mode, mse, ms = c
    color = FAMILY_COLORS[fam]
    marker = "o" if mode == "frozen" else "^"
    size   = 220 if pareto[i] else 100
    zorder = 5 if pareto[i] else 3
    edgecolor = "#111" if pareto[i] else "white"
    lw = 1.8 if pareto[i] else 0.5

    ax.scatter(ms, mse, s=size, c=color, marker=marker,
               edgecolors=edgecolor, linewidths=lw, zorder=zorder)

    # Подпись Pareto-точек
    if pareto[i]:
        ax.annotate(
            lbl,
            xy=(ms, mse),
            xytext=(8, 4),
            textcoords="offset points",
            fontsize=9,
            color="#222",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="#ccc"),
        )

# Нарисуем Pareto-фронт (ломаная по Pareto-точкам)
pareto_pts = sorted(
    [(ms_all[i], mse_all[i]) for i in range(len(CONFIGS)) if pareto[i]],
    key=lambda x: x[0]
)
if pareto_pts:
    px_vals, py_vals = zip(*pareto_pts)
    ax.step(px_vals, py_vals, where="post", color="#e53935",
            linewidth=2, linestyle="--", zorder=4, label="Pareto front")

ax.set_xlabel("Inference time, ms / sample (log scale)", fontsize=12)
ax.set_ylabel("Test MSE", fontsize=12)
ax.set_xscale("log")

# Легенды
fam_handles = [
    mpatches.Patch(facecolor=FAMILY_COLORS[k], label=k, edgecolor="white")
    for k in ["v3_rnn", "v4_tcn", "v5_odd"]
]
mode_handles = [
    plt.scatter([], [], marker="o", c="#888", label="frozen",   edgecolors="white"),
    plt.scatter([], [], marker="^", c="#888", label="finetune", edgecolors="white"),
]
pareto_handle = mpatches.Patch(facecolor="#e53935", alpha=0.8, label="Pareto front")
pareto_dot    = plt.scatter([], [], s=200, c="#888", edgecolors="#111",
                             linewidths=1.8, label="Pareto-оптимальная точка")
ax.legend(
    handles=fam_handles + mode_handles + [pareto_handle, pareto_dot],
    fontsize=10, loc="upper right", framealpha=0.9,
)

fig16.suptitle(
    "Pareto front: точность vs скорость инференса (v3_rnn, v4_tcn, v5_odd)",
    fontsize=17, fontweight="bold", y=1.01,
)
ax.set_title(
    "Ось X: inference ms/sample (log scale); Ось Y: test MSE; ●=frozen, ▲=finetune; "
    "выделены Pareto-оптимальные конфигурации",
    fontsize=10, color="#555", pad=6,
)

plt.tight_layout()
out16 = os.path.join(OUT_DIR, "figure_2_16_pareto_front_mse_vs_inference.png")
fig16.savefig(out16, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig16)
print(f"Saved: {out16}")


# ═══════════════════════════════════════════════════════════════════════════════
# Рисунок 2.17 — Эволюция best test MSE по поколениям
# ═══════════════════════════════════════════════════════════════════════════════
gen_labels = [g[0] for g in GENERATIONS]
gen_mse    = [g[1] for g in GENERATIONS]

fig17, ax = plt.subplots(figsize=(13, 6), facecolor="white")
fig17.patch.set_facecolor("white")
ax.set_facecolor("#f5f5f5")
ax.grid(axis="y", color="white", linewidth=1.4, zorder=0)

x = np.arange(len(gen_labels))
colors_gen = ["#B0BEC5", "#78909C", "#546E7A", "#2196F3", "#FF9800", "#4CAF50"]

bars17 = ax.bar(x, gen_mse, color=colors_gen, edgecolor="white",
                linewidth=0.8, width=0.6, zorder=3)

# Метки значений
for xi, v in zip(x, gen_mse):
    ax.text(xi, v + 0.0005, f"{v:.4f}", ha="center", fontsize=11,
            fontweight="bold", color="#222")

# Стрелка улучшения
for i in range(len(gen_mse) - 1):
    improvement = (gen_mse[i] - gen_mse[i+1]) / gen_mse[i] * 100
    mid = (x[i] + x[i+1]) / 2
    ax.annotate(
        f"−{improvement:.0f}%",
        xy=(mid, (gen_mse[i] + gen_mse[i+1]) / 2),
        fontsize=8.5,
        color="#555",
        ha="center",
    )

ax.set_xticks(x)
ax.set_xticklabels(gen_labels, fontsize=10.5)
ax.set_ylabel("Best Test MSE", fontsize=12)
ax.set_ylim(0, max(gen_mse) * 1.18)

fig17.suptitle(
    "Эволюция качества RUL-предсказания по этапам разработки",
    fontsize=17, fontweight="bold", y=1.01,
)
ax.set_title(
    "Best test MSE каждого поколения моделей; меньше — лучше",
    fontsize=11, color="#555", pad=6,
)

plt.tight_layout()
out17 = os.path.join(OUT_DIR, "figure_2_17_evolution_best_mse_by_generation.png")
fig17.savefig(out17, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig17)
print(f"Saved: {out17}")

print("Все три рисунка готовы!")
