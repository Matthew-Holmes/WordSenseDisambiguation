"""Figure for Report/SWAP_DRAWS_STUDY.md, rendered from Report/swap_draws_study/results.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "Analysis"))
from layer_study_figures import AXIS, GRID, INK, INK_2, MUTED, SURFACE, plt  # noqa: E402
from swap_study_figures import SLOTS  # noqa: E402

RESULTS = REPO / "Report" / "swap_draws_study" / "results.json"
FIG_DIR = REPO / "Report" / "figures"
TITLE = "More swaps bring the contrasts level with the baseline, not above it"
NAMES = {"diff": "Method 1: average, subtract, cosine", "normdiff": "Method 2: normalise first",
         "scorediff": "Method 3: subtract cosines"}


def _style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["left"].set_color(AXIS)
    ax.tick_params(length=0, labelsize=7.6)


def _series(ax, Ks, stats, m, color, off, label):
    mean = [stats[str(K)][m]["mean"] for K in Ks]
    for K, s in zip(Ks, (stats[str(K)][m] for K in Ks)):
        ax.plot([K + off, K + off], [s["min"], s["max"]], color=color, linewidth=1.1, alpha=0.6, zorder=2)
    ax.plot(np.array(Ks) + off, mean, color=color, linewidth=1.6, marker="o", markersize=5, markeredgecolor=SURFACE,
            markeredgewidth=1.0, label=label, zorder=3)


def fig_curve(r) -> Path:
    Ks = sorted(int(k) for k in r["subset_stats"])
    stats = r["subset_stats"]
    base = r["baseline_fixed"]["pooled"]["acc1"]
    chance = r["random_pooled"]["acc1"]["mean"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.6, 4.3), dpi=200, gridspec_kw={"width_ratios": [1.6, 1]})
    fig.subplots_adjust(left=0.09, right=0.98, top=0.68, bottom=0.13, wspace=0.28)
    fig.text(0.02, 0.965, TITLE, fontsize=11, color=INK, weight="bold", va="top")
    fig.text(0.02, 0.905, "Accuracy@1 on both sense-balanced samples pooled (400 instances), at the layer study's "
             "configuration (layers 5–7).\nPoints: the mean over every choice of K of the 5 swapped sentences; bars: "
             "lowest to highest. Shaded: the baseline's 95% CI.", fontsize=8.0, color=INK_2, va="top",
             linespacing=1.4)
    ax.axhspan(base["lo"], base["hi"], color=SLOTS[0], alpha=0.10, linewidth=0, zorder=0)
    ax.axhline(base["mean"], color=SLOTS[0], linewidth=1.4, zorder=1, label=f"Baseline cosine ({base['mean']:.3f})")
    for j, (m, color) in enumerate(zip(NAMES, SLOTS[1:])):
        _series(ax, Ks, stats, m, color, (j - 1) * 0.13, NAMES[m])
    ax.set_title("The contrasts", fontsize=8.6, color=INK_2, loc="left")
    ax.set_ylim(0.40, 0.52)
    ax.set_ylabel("Accuracy@1", fontsize=8.0)
    _series(ax2, Ks, stats, "swap", MUTED, 0.0, None)
    ax2.axhline(chance, color=MUTED, linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)
    ax2.annotate("chance", xy=(Ks[-1] + 0.4, chance), xytext=(0, 3), textcoords="offset points", ha="right",
                 fontsize=7.0, color=MUTED)
    ax2.set_title("The averaged swapped slot alone", fontsize=8.6, color=INK_2, loc="left")
    ax2.set_ylim(0.24, 0.40)
    for a in (ax, ax2):
        a.set_xticks(Ks)
        a.set_xlim(Ks[0] - 0.4, Ks[-1] + 0.4)
        a.set_xlabel("K, swapped sentences averaged", fontsize=8.0)
        _style(a)
    fig.legend(loc="upper left", bbox_to_anchor=(0.085, 0.80), ncol=2, frameon=False, fontsize=7.4,
               labelcolor=INK_2, handlelength=1.6, columnspacing=1.6)
    out = FIG_DIR / "swapdraws_fig1_curve.png"
    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("wrote", fig_curve(json.loads(RESULTS.read_text())))
