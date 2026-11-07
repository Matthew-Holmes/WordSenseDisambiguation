"""Figures for Report/LAYER_STUDY.md, rendered from Report/layer_study/results.json.

Visual rules follow Report/verification/05_figures.py: light chart surface (legible in light and dark
viewers), the reference palette's first three categorical slots in fixed order, 2px lines, solid
hairline grid, recessive axes, legend plus direct end labels, reference lines in muted ink and
labelled directly. Each figure's numbers are also tabulated in the report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "Report" / "layer_study" / "results.json"
FIG_DIR = REPO / "Report" / "figures"

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK_2,
    "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})
LAYER_LABEL = "Layer (0 = embeddings, k = output of block k)"
REP_NAMES = {"plain_last": "left context only", "echo_last": "whole sentence (echo), last sub-token",
             "echo_mean": "whole sentence (echo), mean of sub-tokens"}
SHORT = {"plain_last": "Left context", "echo_last": "Echo, last", "echo_mean": "Echo, mean"}
# Headline titles state the finding; they are set after reading results.json (see Report/LAYER_STUDY.md).
TITLES = {
    "dev_profiles": "Blocks 5-8 match a word to its definition best",
    "dev_vs_test": "The middle-layer plateau carries over to held-out documents",
    "test_validation": "Held out: above chance everywhere, above frequency only when balanced",
}


def frame(title, subtitle, ylabel, ylim, xmax=16, right=0.80):
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    fig.subplots_adjust(left=0.09, right=right, top=0.72, bottom=0.12)
    fig.text(0.09, 0.965, title, fontsize=11, color=INK, weight="bold", va="top")
    fig.text(0.09, 0.91, subtitle, fontsize=8.3, color=INK_2, va="top", linespacing=1.4)
    ax.set_ylim(*ylim)
    ax.set_xlim(-0.4, xmax + 0.4)
    ax.set_xticks(range(xmax + 1))
    ax.set_xlabel(LAYER_LABEL)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    return fig, ax


def lines(ax, series, refs=(), x_end=16, colors=None):
    """Series with markers, a one-row legend above the plot, and direct end labels.

    `refs` are (y, text) reference lines (e.g. chance), drawn in muted ink. All end labels, series and
    reference alike, are laid out together so they never overlap. `colors` defaults to the categorical
    slots in order.
    """
    labels = []
    for y, text in refs:
        ax.axhline(y, color=MUTED, linewidth=0.9, zorder=2)
        labels.append([y, text, None])
    for (label, short, ys), color in zip(series, colors or SERIES):
        ax.plot(range(len(ys)), ys, color=color, linewidth=1.6, marker="o", markersize=4.2,
                markeredgecolor=SURFACE, markeredgewidth=1.0, label=label, zorder=3)
        labels.append([ys[-1], short, color])
    labels.sort(key=lambda e: e[0])
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.065
    for i in range(1, len(labels)):
        labels[i][0] = max(labels[i][0], labels[i - 1][0] + gap)
    for y, text, color in labels:
        if color is not None:
            ax.plot([x_end + 0.75], [y], marker="s", markersize=4, color=color, clip_on=False, zorder=4)
        ax.annotate(text, xy=(x_end + 0.75, y), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=7.8, color=INK_2 if color else MUTED, annotation_clip=False)
    ax.figure.legend(loc="upper left", bbox_to_anchor=(0.085, 0.81), ncol=len(series), frameon=False,
                     fontsize=7.8, labelcolor=INK_2, handlelength=1.6, columnspacing=1.4)


def fig_dev_profiles(r) -> Path:
    best = r["dev_selection"]["winner"]
    g = best.split("/")[0].split("+")[1]
    series = [(REP_NAMES[s], SHORT[s], r["profiles"][f"dev_natural|{s}+{g}"]["acc1"]) for s in REP_NAMES]
    val = r["validation"]["dev_natural"]["systems"]
    chance = val["random"]["acc1"]["mean"]
    fig, ax = frame(
        TITLES["dev_profiles"],
        f"Accuracy@1 by layer on dev documents ({r['samples']['dev_natural']['n_instances']} instances, "
        f"{r['samples']['dev_natural']['mean_candidates']:.1f} candidates on average),\n"
        f"for three ways of reading the target word; glosses represented by their {'mean token' if g == 'mean' else 'last token'}.",
        "Accuracy@1 (dev)", (0.0, 0.85))
    lines(ax, series, refs=[(chance, "chance")])
    out = FIG_DIR / "layer_fig1_dev_profiles.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_dev_vs_test(r) -> Path:
    best = r["dev_selection"]["winner"]
    rep, band = best.split("/")
    layers = r["dev_selection"]["winner_layers"]
    series = [("Dev documents (used to choose)", "Dev", r["profiles"][f"dev_natural|{rep}"]["acc1"]),
              ("Test documents (held out)", "Test", r["profiles"][f"test_natural|{rep}"]["acc1"])]
    chance = r["validation"]["test_natural"]["systems"]["random"]["acc1"]["mean"]
    fig, ax = frame(
        TITLES["dev_vs_test"],
        f"Accuracy@1 by layer for the selected representation ({rep.replace('+', ' × gloss ')}), dev vs test "
        f"natural samples.\nShaded: the band chosen on dev ({band}). Spearman correlation of the two profiles: "
        f"{r['profile_agreement']['spearman_dev_vs_test_natural_acc1']:.2f}.",
        "Accuracy@1", (0.0, 0.85))
    ax.axvspan(min(layers) - 0.4, max(layers) + 0.4, color=GRID, alpha=0.7, zorder=1, linewidth=0)
    lines(ax, series, refs=[(chance, "chance")])
    out = FIG_DIR / "layer_fig2_dev_vs_test.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_test_validation(r) -> Path:
    band = r["dev_selection"]["winner"].split("/")[1]
    rows = [  # (results key, label, colour): the two systems of interest in colour, baselines in muted ink
        (f"selected: {r['dev_selection']['winner']}", f"Selected layers ({band})", SERIES[0]),
        ("selected + dev sense-frequency prior", "Selected layers + frequency prior", SERIES[1]),
        ("embeddings only (layer 0)", "Embeddings only (layer 0)", MUTED),
        ("shuffled-sentence control", "Shuffled-sentence control", MUTED),
        ("MFS from dev documents", "Most frequent sense (dev docs)", MUTED),
        ("WordNet first sense (optimistic)", "WordNet first sense (optimistic)", MUTED),
        ("random", "Random", MUTED),
    ]
    order = [k for k, _, _ in rows]
    labels = [lab for _, lab, _ in rows]
    colors = {k: c for k, _, c in rows}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), dpi=200, sharey=True)
    fig.subplots_adjust(left=0.30, right=0.97, top=0.78, bottom=0.14, wspace=0.08)
    fig.text(0.02, 0.96, TITLES["test_validation"], fontsize=11, color=INK, weight="bold", va="top")
    fig.text(0.02, 0.895, f"Accuracy@1 with 95% sentence-clustered bootstrap CIs. Selected on dev: "
             f"{r['dev_selection']['winner'].replace('+', ' × gloss ')}.", fontsize=8.3, color=INK_2, va="top")
    for ax, name, title in zip(axes, ("test_natural", "test_balanced"), ("Natural sense distribution", "Sense-balanced")):
        sysd = r["validation"][name]["systems"]
        y = np.arange(len(order))[::-1]
        for yi, k in zip(y, order):
            a = sysd[k]["acc1"]
            color = colors[k]
            ax.plot([a["lo"], a["hi"]], [yi, yi], color=color, linewidth=1.6, solid_capstyle="round", zorder=2)
            ax.plot([a["mean"]], [yi], marker="o", markersize=6, color=color, markeredgecolor=SURFACE,
                    markeredgewidth=1.0, zorder=3)
            ax.annotate(f"{a['mean']:.3f}", xy=(a["hi"], yi), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=7.2, color=INK_2)
        ax.set_title(f"{title} (n = {r['validation'][name]['n_instances']})", fontsize=8.6, color=INK_2, loc="left")
        ax.set_xlim(0, 1.0)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])  # no 1.0 tick, so the two panels' labels never touch
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
        ax.set_xlabel("Accuracy@1 (test)")
    axes[0].set_yticks(np.arange(len(order))[::-1])
    axes[0].set_yticklabels(labels, fontsize=8, color=INK_2)
    out = FIG_DIR / "layer_fig3_test_validation.png"
    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    res = json.loads(RESULTS.read_text())
    for f in (fig_dev_profiles, fig_dev_vs_test, fig_test_validation):
        print("wrote", f(res))
    sys.exit(0)
