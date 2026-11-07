"""Figures for Report/GLOSS_MEAN_STUDY.md, rendered from Report/gloss_mean_study/results.json.

Same visual rules as Analysis/layer_study_figures.py and Analysis/swap_study_figures.py (whose helpers this
reuses): light chart surface, the reference palette's categorical slots in fixed order, a legend plus direct end
labels, muted reference lines, and every figure's numbers tabulated in the report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "Analysis"))
import layer_study_figures as lsf  # noqa: E402
from layer_study_figures import AXIS, GRID, INK, INK_2, MUTED, SURFACE, plt  # noqa: E402
from swap_study_figures import SLOTS, band_of  # noqa: E402

RESULTS = REPO / "Report" / "gloss_mean_study" / "results.json"
FIG_DIR = REPO / "Report" / "figures"
# Headline titles state the finding; they are set after reading results.json (see Report/GLOSS_MEAN_STUDY.md).
TITLES = {
    "dev_profiles": "Removing the shared gloss mean costs accuracy at the best layers",
    "test_validation": "Held out: removing the mean trails the baseline on both test samples",
    "alpha": "The more of the mean is removed, the lower the accuracy on natural text",
}


def fig_dev_profiles(r) -> Path:
    base_cfg = r["selection"]["baseline"]["config"]
    rep = base_cfg.split("/")[0]
    prof = lambda score: r["profiles"][f"dev_natural|{rep}|{score}"]  # noqa: E731
    series = [("Baseline", "Baseline", prof("baseline")),
              ("Remove the mean", "Remove mean", prof("glossmean")),
              ("Normalise, then remove the mean", "Normalise, remove", prof("unitmean")),
              ("Dot product (diagnostic)", "Dot product", prof("dot"))]
    chance = r["validation"]["dev_natural"]["systems"]["random"]["acc1"]["mean"]
    n = r["samples"]["dev_natural"]["n_instances"]
    fig, ax = lsf.frame(
        TITLES["dev_profiles"],
        f"Accuracy@1 by layer on dev documents ({n} instances); target read as {rep.replace('+', ' × gloss ')}.\n"
        "Removing the mean subtracts the candidates' average gloss vector; normalising first tracks the baseline.",
        "Accuracy@1 (dev)", (0.0, 0.85))
    lsf.lines(ax, series, refs=[(chance, "chance")], colors=SLOTS[:len(series)])
    out = FIG_DIR / "glossmean_fig1_dev_profiles.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_test_validation(r) -> Path:
    sel = r["selection"]
    base_cfg = sel["baseline"]["config"]
    alpha = r["best_alpha"]
    rows = [  # (results key, label, colour)
        ("baseline", f"Baseline ({band_of(base_cfg)})", SLOTS[0]),
        ("glossmean|own", f"Remove the mean ({band_of(sel['glossmean']['config'])})", SLOTS[1]),
        ("unitmean|own", f"Normalise, then remove ({band_of(sel['unitmean']['config'])})", SLOTS[1]),
        ("alpha|base_cfg", f"Partial, α = {alpha:g} from dev ({band_of(base_cfg)})".replace("-", "−"), SLOTS[1]),
        ("dot|own", f"Dot product, diagnostic ({band_of(sel['dot']['config'])})", MUTED),
        ("MFS from dev documents", "Most frequent sense (dev docs)", MUTED),
        ("WordNet first sense (optimistic)", "WordNet first sense (optimistic)", MUTED),
        ("random", "Random", MUTED),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.1), dpi=200, sharey=True)
    fig.subplots_adjust(left=0.36, right=0.97, top=0.78, bottom=0.13, wspace=0.08)
    fig.text(0.02, 0.96, TITLES["test_validation"], fontsize=11, color=INK, weight="bold", va="top")
    fig.text(0.02, 0.895, "Accuracy@1 with 95% sentence-clustered bootstrap CIs. Layers and representation were chosen "
             "on dev for each score;\nblue = the layer study's score, orange = gloss-mean removal, grey = diagnostic "
             "and reference baselines.", fontsize=8.0, color=INK_2, va="top", linespacing=1.4)
    y = np.arange(len(rows))[::-1]
    for ax, name, title in zip(axes, ("test_natural", "test_balanced"), ("Natural sense distribution", "Sense-balanced")):
        sysd = r["validation"][name]["systems"]
        for yi, (k, _, color) in zip(y, rows):
            a = sysd[k]["acc1"]
            ax.plot([a["lo"], a["hi"]], [yi, yi], color=color, linewidth=1.6, solid_capstyle="round", zorder=2)
            ax.plot([a["mean"]], [yi], marker="o", markersize=6, color=color, markeredgecolor=SURFACE,
                    markeredgewidth=1.0, zorder=3)
            ax.annotate(f"{a['mean']:.3f}", xy=(a["hi"], yi), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=7.2, color=INK_2)
        ax.set_title(f"{title} (n = {r['validation'][name]['n_instances']})", fontsize=8.6, color=INK_2, loc="left")
        ax.set_xlim(0, 1.0)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(length=0)
        ax.set_xlabel("Accuracy@1 (test)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([lab for _, lab, _ in rows], fontsize=7.8, color=INK_2)
    out = FIG_DIR / "glossmean_fig2_test_validation.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_alpha(r) -> Path:
    alphas = r["alphas"]
    curves = r["alpha_curves"]
    best = r["best_alpha"]
    series = [("Dev documents (used to choose α)", "Dev", curves["dev_natural"]),
              ("Test documents, natural", "Test natural", curves["test_natural"]),
              ("Test documents, sense-balanced", "Test balanced", curves["test_balanced"])]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    fig.subplots_adjust(left=0.09, right=0.80, top=0.72, bottom=0.12)
    fig.text(0.09, 0.965, TITLES["alpha"], fontsize=11, color=INK, weight="bold", va="top")
    base_cfg = r["selection"]["baseline"]["config"]
    fig.text(0.09, 0.91, f"Accuracy@1 of cos(h, g − α·m), m = the candidates' mean gloss vector, at the baseline's "
             f"configuration\n({base_cfg.split('/')[0].replace('+', ' × gloss ')}, {band_of(base_cfg)}). "
             f"α = 0 is the baseline, α = 1 removes the whole mean, α < 0 adds it.",
             fontsize=8.3, color=INK_2, va="top", linespacing=1.4)
    lo = min(min(c) for _, _, c in series)
    hi = max(max(c) for _, _, c in series)
    ax.set_ylim(max(0.0, lo - 0.08), min(1.0, hi + 0.08))
    ax.set_xlim(alphas[0] - 0.05, alphas[-1] + 0.05)
    ax.set_xlabel("α (fraction of the candidates' mean removed)")
    ax.set_ylabel("Accuracy@1")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    for x, text in ((0.0, "baseline"), (1.0, "remove the mean")):
        ax.axvline(x, color=MUTED, linewidth=0.9, zorder=1)
        ax.annotate(text, xy=(x, ax.get_ylim()[1]), xytext=(3, -3), textcoords="offset points", va="top",
                    fontsize=7.6, color=MUTED)
    labels = []
    for (label, short, ys), color in zip(series, SLOTS):
        ax.plot(alphas, ys, color=color, linewidth=1.6, marker="o", markersize=3.2, markeredgecolor=SURFACE,
                markeredgewidth=0.8, label=label, zorder=3)
        labels.append([ys[-1], short, color])
    best_y = curves["dev_natural"][alphas.index(best)]
    ax.plot([best], [best_y], marker="o", markersize=11, markerfacecolor="none", markeredgecolor=INK,
            markeredgewidth=1.0, zorder=5)
    ax.annotate(f"α = {best:g}, chosen on dev".replace("-", "−"), xy=(best, best_y),
                xytext=(alphas[0] + 0.02, best_y - 0.035), textcoords="data", fontsize=7.6, color=INK_2,
                va="center", ha="left",
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8, shrinkA=2, shrinkB=6))
    labels.sort(key=lambda e: e[0])
    gap = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.065
    for i in range(1, len(labels)):
        labels[i][0] = max(labels[i][0], labels[i - 1][0] + gap)
    for yv, text, color in labels:
        ax.plot([alphas[-1] + 0.1], [yv], marker="s", markersize=4, color=color, clip_on=False, zorder=4)
        ax.annotate(text, xy=(alphas[-1] + 0.1, yv), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=7.8, color=INK_2, annotation_clip=False)
    ax.figure.legend(loc="upper left", bbox_to_anchor=(0.085, 0.81), ncol=3, frameon=False, fontsize=7.8,
                     labelcolor=INK_2, handlelength=1.6, columnspacing=1.4)
    out = FIG_DIR / "glossmean_fig3_alpha.png"
    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    res = json.loads(RESULTS.read_text())
    for f in (fig_dev_profiles, fig_test_validation, fig_alpha):
        print("wrote", f(res))
