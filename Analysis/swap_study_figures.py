"""Figures for Report/SWAP_STUDY.md, rendered from Report/swap_study/results.json.

Same visual rules as Analysis/layer_study_figures.py (whose helpers this reuses): light chart surface, the reference
palette's categorical slots in fixed order, a legend plus direct end labels, muted reference lines, and every
figure's numbers tabulated in the report.
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

RESULTS = REPO / "Report" / "swap_study" / "results.json"
FIG_DIR = REPO / "Report" / "figures"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4 (reference palette, light)
# Headline titles state the finding; they are set after reading results.json (see Report/SWAP_STUDY.md).
TITLES = {
    "dev_profiles": "Subtracting the swapped slot costs accuracy at the best layers",
    "test_validation": "Held out: the contrasts trail the baseline on natural text and tie when balanced",
    "beta": "Partial subtraction changes little; method 3's full subtraction hurts",
    "conditions": "No swap variant beats the baseline on either test sample",
}
CONDITION_LABELS = {"pos": "pos, draw 0 (primary)", "pos+form": "pos+form, draw 0", "pos draw 1": "pos, draw 1",
                    "pos x2": "pos, draws 0 and 1 averaged"}


def band_of(config: str) -> str:
    """'echo_last+mean/L5-7' -> 'layers 5–7'; '.../L6' -> 'layer 6'."""
    band = config.split("/")[1][1:]
    return f"layers {band.replace('-', '–')}" if "-" in band else f"layer {band}"


def fig_dev_profiles(r) -> Path:
    base_cfg = r["selection"]["baseline"]["config"]
    rep = base_cfg.split("/")[0]
    prof = lambda score: r["profiles"][f"{r['primary_condition']}|dev_natural|{rep}|{score}"]  # noqa: E731
    same = np.allclose(prof("normdiff"), prof("scorediff"))
    series = [("Baseline", "Baseline", prof("baseline")),
              ("Method 1", "Method 1", prof("diff")),
              ("Methods 2 and 3" if same else "Method 2", "Methods 2, 3" if same else "Method 2", prof("normdiff")),
              ("Swapped slot alone", "Swap alone", prof("swap"))]
    if not same:
        series.insert(3, ("Method 3", "Method 3", prof("scorediff")))
    chance = r["validation"]["dev_natural"]["systems"]["random"]["acc1"]["mean"]
    n = r["samples"]["dev_natural"]["n_instances"]
    fig, ax = lsf.frame(
        TITLES["dev_profiles"],
        f"Accuracy@1 by layer on dev documents ({n} instances); target read as {rep.replace('+', ' × gloss ')}.\n"
        "s = the slot holding a random same-POS word."
        + (" Methods 2 and 3 coincide at every layer" if same else "")
        + ("; method 1 nearly does." if same and np.max(np.abs(np.subtract(prof("diff"), prof("scorediff")))) < 0.03
           else "."),
        "Accuracy@1 (dev)", (0.0, 0.85))
    lsf.lines(ax, series, refs=[(chance, "chance")], colors=SLOTS[:len(series)])
    out = FIG_DIR / "swap_fig1_dev_profiles.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_test_validation(r) -> Path:
    cond = r["primary_condition"]
    sel = r["selection"][cond]
    base_cfg = r["selection"]["baseline"]["config"]
    rows = [  # (results key, label, colour)
        ("baseline", f"Baseline ({band_of(base_cfg)})", SLOTS[0]),
        (f"{cond}|diff|own", f"Method 1: subtract, then cosine ({band_of(sel['diff']['config'])})", SLOTS[1]),
        (f"{cond}|normdiff|own", f"Method 2: normalise, subtract ({band_of(sel['normdiff']['config'])})", SLOTS[1]),
        (f"{cond}|scorediff|own", f"Method 3: subtract cosines ({band_of(sel['scorediff']['config'])})", SLOTS[1]),
        (f"{cond}|swap|own", f"Swapped slot alone ({band_of(sel['swap']['config'])})", MUTED),
        ("MFS from dev documents", "Most frequent sense (dev docs)", MUTED),
        ("WordNet first sense (optimistic)", "WordNet first sense (optimistic)", MUTED),
        ("random", "Random", MUTED),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.1), dpi=200, sharey=True)
    fig.subplots_adjust(left=0.36, right=0.97, top=0.78, bottom=0.13, wspace=0.08)
    fig.text(0.02, 0.96, TITLES["test_validation"], fontsize=11, color=INK, weight="bold", va="top")
    fig.text(0.02, 0.895, "Accuracy@1 with 95% sentence-clustered bootstrap CIs. Each method's layers and representation "
             "were chosen on dev;\nblue = the layer study's score, orange = the three swapped-word contrasts, grey = "
             "diagnostics and baselines.", fontsize=8.0, color=INK_2, va="top", linespacing=1.4)
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
    out = FIG_DIR / "swap_fig2_test_validation.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_beta(r) -> Path:
    cond = r["primary_condition"]
    betas = r["betas"]
    curves = r["beta_curves"][cond]
    best = r["best_beta"][cond]
    series = [("Dev documents (used to choose β)", "Dev", curves["dev_natural"]),
              ("Test documents, natural", "Test natural", curves["test_natural"]),
              ("Test documents, sense-balanced", "Test balanced", curves["test_balanced"])]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    fig.subplots_adjust(left=0.09, right=0.80, top=0.72, bottom=0.12)
    fig.text(0.09, 0.965, TITLES["beta"], fontsize=11, color=INK, weight="bold", va="top")
    base_cfg = r["selection"]["baseline"]["config"]
    fig.text(0.09, 0.91, f"Accuracy@1 of cos(h, g) − β·cos(s, g) at the baseline's configuration "
             f"({base_cfg.split('/')[0].replace('+', ' × gloss ')}, {band_of(base_cfg)}).\nβ = 0 is the baseline, "
             f"β = 1 is method 3, β < 0 adds the swapped slot. Ringed: β = {best:g}, chosen on dev."
             .replace("= -", "= −"),
             fontsize=8.3, color=INK_2, va="top", linespacing=1.4)
    lo = min(min(c) for _, _, c in series)
    hi = max(max(c) for _, _, c in series)
    ax.set_ylim(max(0.0, lo - 0.08), min(1.0, hi + 0.08))
    ax.set_xlim(betas[0] - 0.05, betas[-1] + 0.05)
    ax.set_xlabel("β (weight on the swapped slot's cosine)")
    ax.set_ylabel("Accuracy@1")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    for x, text in ((0.0, "baseline"), (1.0, "method 3")):
        ax.axvline(x, color=MUTED, linewidth=0.9, zorder=1)
        ax.annotate(text, xy=(x, ax.get_ylim()[1]), xytext=(3, -3), textcoords="offset points", va="top",
                    fontsize=7.6, color=MUTED)
    labels = []
    for (label, short, ys), color in zip(series, SLOTS):
        ax.plot(betas, ys, color=color, linewidth=1.6, marker="o", markersize=3.2, markeredgecolor=SURFACE,
                markeredgewidth=0.8, label=label, zorder=3)
        labels.append([ys[-1], short, color])
    ax.plot([best], [curves["dev_natural"][betas.index(best)]], marker="o", markersize=11, markerfacecolor="none",
            markeredgecolor=INK, markeredgewidth=1.0, zorder=5)
    labels.sort(key=lambda e: e[0])
    gap = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.065
    for i in range(1, len(labels)):
        labels[i][0] = max(labels[i][0], labels[i - 1][0] + gap)
    for yv, text, color in labels:
        ax.plot([betas[-1] + 0.12], [yv], marker="s", markersize=4, color=color, clip_on=False, zorder=4)
        ax.annotate(text, xy=(betas[-1] + 0.12, yv), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=7.8, color=INK_2, annotation_clip=False)
    ax.figure.legend(loc="upper left", bbox_to_anchor=(0.085, 0.81), ncol=3, frameon=False, fontsize=7.8,
                     labelcolor=INK_2, handlelength=1.6, columnspacing=1.4)
    out = FIG_DIR / "swap_fig4_beta.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_conditions(r) -> Path:
    """Paired test differences from the baseline for every swap condition and method (own dev selection)."""
    conds = list(r["conditions"])
    methods = [("diff", "Method 1", SLOTS[1], "o", 0.25), ("normdiff", "Method 2", SLOTS[2], "s", 0.0),
               ("scorediff", "Method 3", SLOTS[3], "D", -0.25)]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.3), dpi=200, sharey=True)
    fig.subplots_adjust(left=0.27, right=0.97, top=0.69, bottom=0.13, wspace=0.08)
    fig.text(0.02, 0.965, TITLES["conditions"], fontsize=11, color=INK, weight="bold", va="top")
    fig.text(0.02, 0.905, "Paired difference in test accuracy@1 from the baseline, with 95% sentence-clustered bootstrap "
             "CIs. For every set\nof swaps, each method's layers and representation were chosen on dev. Left of zero = "
             "worse than the baseline.", fontsize=8.0, color=INK_2, va="top", linespacing=1.4)
    y = np.arange(len(conds))[::-1].astype(float)
    for ax, name, title in zip(axes, ("test_natural", "test_balanced"), ("Natural sense distribution", "Sense-balanced")):
        paired = r["validation"][name]["paired_vs_baseline_acc1"]
        ax.axvline(0.0, color=MUTED, linewidth=1.0, zorder=1)
        for key, label, color, marker, off in methods:
            for yi, c in zip(y, conds):
                p = paired[f"{c}|{key}|own"]
                ax.plot([p["lo"], p["hi"]], [yi + off, yi + off], color=color, linewidth=1.6,
                        solid_capstyle="round", zorder=2)
                ax.plot([p["diff"]], [yi + off], marker=marker, markersize=5.5 if marker != "D" else 4.8, color=color,
                        markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3,
                        label=label if (yi == y[0] and name == "test_natural") else None)
        ax.annotate("baseline", xy=(0.0, len(conds) - 0.45), xytext=(3, 0), textcoords="offset points", va="center",
                    fontsize=7.4, color=MUTED)
        ax.set_title(f"{title} (n = {r['validation'][name]['n_instances']})", fontsize=8.6, color=INK_2, loc="left")
        ax.set_xlim(-0.17, 0.09)
        ax.set_xticks([-0.15, -0.10, -0.05, 0.0, 0.05])
        ax.set_xticklabels(["−0.15", "−0.10", "−0.05", "0", "+0.05"])
        ax.set_ylim(-0.6, len(conds) - 0.3)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(length=0)
        ax.set_xlabel("Difference from the baseline (test)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([CONDITION_LABELS.get(c, c) for c in conds], fontsize=7.8, color=INK_2)
    fig.legend(loc="upper left", bbox_to_anchor=(0.265, 0.81), ncol=3, frameon=False, fontsize=7.8,
               labelcolor=INK_2, handlelength=1.4, columnspacing=1.6)
    out = FIG_DIR / "swap_fig3_conditions.png"
    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    res = json.loads(RESULTS.read_text())
    for f in (fig_dev_profiles, fig_test_validation, fig_conditions, fig_beta):
        print("wrote", f(res))
