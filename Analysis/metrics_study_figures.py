"""Figures for Report/METRICS_STUDY.md, rendered from Report/metrics_study/results.json.

Same visual rules as the other study figures: light surface, the reference palette's categorical slots in fixed
order, legends, muted reference lines, and every number tabulated in the report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "Analysis"))
from layer_study_figures import AXIS, GRID, INK, INK_2, MUTED, SURFACE, plt  # noqa: E402
from swap_study_figures import SLOTS  # noqa: E402

RESULTS = REPO / "Report" / "metrics_study" / "results.json"
FIG_DIR = REPO / "Report" / "figures"
TITLES = {
    "differences": "Standardising each dimension is the only lead; no method is a clear win",
    "profiles": "Standardising each dimension beats the cosine at 14 of 17 layers",
}
LABELS = {
    "euclidean": "Euclidean distance", "manhattan": "Manhattan distance", "pearson": "Pearson correlation",
    "spearman": "Spearman rank correlation", "dot": "Dot product",
    "standardise": "Standardise each dimension", "abtt10": "Remove top 10 directions", "whiten": "Soft whitening",
    "csls10": "CSLS hubness correction", "gloss_z": "Gloss z-score", "layer_z": "Layer z-score",
    "ensemble": "Ensemble of 6 representations", "standardise+gloss_z": "Standardise + gloss z-score",
    "cosine": "Cosine (baseline)",
}
FAMILIES = [("Other metrics", ["euclidean", "manhattan", "pearson", "spearman", "dot"], SLOTS[1]),
            ("Re-shaping the space", ["standardise", "abtt10", "whiten"], SLOTS[2]),
            ("Post-processing", ["csls10", "gloss_z", "layer_z", "ensemble", "standardise+gloss_z"], SLOTS[3])]


def _style(ax):
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0)


def fig_differences(r) -> Path:
    rows, y, yi = [], [], 0.0
    for _, methods, color in FAMILIES:
        for m in methods:
            rows.append((m, color))
            y.append(yi)
            yi -= 1.0
        yi -= 0.6  # gap between families
    y = np.array(y)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 5.4), dpi=200, sharey=True)
    fig.subplots_adjust(left=0.29, right=0.97, top=0.73, bottom=0.10, wspace=0.08)
    fig.text(0.02, 0.97, TITLES["differences"], fontsize=11, color=INK, weight="bold", va="top")
    fig.text(0.02, 0.915, "Paired difference in accuracy@1 from the cosine on sense-balanced data, with 95% "
             "sentence-clustered bootstrap CIs.\nLeft: every method at the layer study's layers, both balanced "
             "samples pooled. Right: each method's own\nlayers chosen on dev_balanced, scored on test_balanced.",
             fontsize=8.0, color=INK_2, va="top", linespacing=1.4)
    panels = [(axes[0], [r["fixed"][m]["paired_vs_cosine_acc1"] for m, _ in rows],
               f"Fixed layers 5–7, pooled (n = 400)"),
              (axes[1], [r["selection"][m]["paired_vs_cosine_acc1"] for m, _ in rows],
               f"Own selection, test_balanced (n = 200)")]
    for ax, paired, title in panels:
        ax.axvline(0.0, color=MUTED, linewidth=1.0, zorder=1)
        for yy, p, (m, color) in zip(y, paired, rows):
            ax.plot([p["lo"], p["hi"]], [yy, yy], color=color, linewidth=1.6, solid_capstyle="round", zorder=2)
            ax.plot([p["diff"]], [yy], marker="o", markersize=5.5, color=color, markeredgecolor=SURFACE,
                    markeredgewidth=1.0, zorder=3)
        ax.annotate("cosine", xy=(0.0, y[0] + 0.75), xytext=(3, 0), textcoords="offset points", va="center",
                    fontsize=7.4, color=MUTED)
        ax.set_title(title, fontsize=8.6, color=INK_2, loc="left")
        ax.set_xlim(-0.13, 0.11)
        ax.set_xticks([-0.10, -0.05, 0.0, 0.05, 0.10])
        ax.set_xticklabels(["−0.10", "−0.05", "0", "+0.05", "+0.10"])
        ax.set_ylim(y[-1] - 0.7, y[0] + 1.2)
        ax.set_xlabel("Difference from the cosine")
        _style(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([LABELS[m] for m, _ in rows], fontsize=7.8, color=INK_2)
    handles = [plt.Line2D([0], [0], color=c, marker="o", linewidth=1.6, markersize=5) for _, _, c in FAMILIES]
    fig.legend(handles, [f for f, _, _ in FAMILIES], loc="upper left", bbox_to_anchor=(0.285, 0.805), ncol=3,
               frameon=False, fontsize=7.8, labelcolor=INK_2, handlelength=1.6, columnspacing=1.6)
    out = FIG_DIR / "metrics_fig1_differences.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_profiles(r) -> Path:
    panels = [("Other metrics", ["spearman", "dot", "euclidean"]),
              ("Re-shaping the space", ["standardise", "abtt10", "whiten"]),
              ("Post-processing", ["csls10", "gloss_z", "standardise+gloss_z"])]
    P = r["profiles"]
    chance = r["fixed"]["random"]["pooled"]["acc1"]["mean"]
    x = np.arange(len(P["cosine"]))
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 4.3), dpi=200, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.64, bottom=0.12, wspace=0.10)
    fig.text(0.02, 0.97, TITLES["profiles"], fontsize=11, color=INK, weight="bold", va="top")
    fig.text(0.02, 0.915, "Accuracy@1 by layer on both sense-balanced samples pooled (400 instances), target read as "
             "echo_last × gloss mean.\nEach panel repeats the cosine in blue. Single-layer accuracies vary by about "
             "±0.05 from noise alone.", fontsize=8.0, color=INK_2, va="top", linespacing=1.4)
    for ax, (title, methods) in zip(axes, panels):
        ax.axhline(chance, color=MUTED, linewidth=0.9, zorder=1)
        for m, color in zip(["cosine"] + methods, SLOTS):
            ax.plot(x, P[m], color=color, linewidth=1.5, marker="o", markersize=2.6, markeredgecolor=SURFACE,
                    markeredgewidth=0.6, label=LABELS[m].replace(" (baseline)", ""), zorder=3 if m == "cosine" else 2)
        ax.set_xticks([0, 4, 8, 12, 16])
        ax.set_xlim(-0.5, 16.5)
        ax.set_ylim(0.2, 0.55)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.spines["left"].set_color(AXIS)
        ax.tick_params(length=0, labelsize=7.4)
        ax.set_xlabel("Layer", fontsize=8.0)
        leg = ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.01), frameon=False, fontsize=6.8,
                        labelcolor=INK_2, handlelength=1.4, ncol=1, title=title, title_fontsize=8.4,
                        borderaxespad=0.0)
        leg._legend_box.align = "left"
        leg.get_title().set_color(INK_2)
    axes[0].set_ylabel("Accuracy@1", fontsize=8.0)
    axes[0].annotate("chance", xy=(16.5, chance), xytext=(0, 3), textcoords="offset points", ha="right",
                     fontsize=7.0, color=MUTED)
    out = FIG_DIR / "metrics_fig2_profiles.png"
    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    res = json.loads(RESULTS.read_text())
    for f in (fig_differences, fig_profiles):
        print("wrote", f(res))
