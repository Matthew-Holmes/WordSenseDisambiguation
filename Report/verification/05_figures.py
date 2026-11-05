"""Render the report figures from the JSON results (results/*.json -> ../figures/*.png).

Palette: the dataviz reference palette's first three categorical slots, in fixed order
(blue, orange, aqua), which are documented as passing the all-pairs CVD and normal-vision
checks in light mode. Figures are rendered on the light chart surface so they read the same in
light and dark viewers; every figure has a numeric table twin in REPORT.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import REPO, RESULTS  # noqa: E402

FIG_DIR = REPO / "Report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # slots 1-3, fixed order

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titlecolor": INK,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


def load(name: str) -> dict:
    with open(RESULTS / name, encoding="utf-8") as f:
        return json.load(f)


def base_axes(title: str, subtitle: str, ylabel: str, ylim: tuple[float, float]):
    fig, ax = plt.subplots(figsize=(7.2, 4.1), dpi=200)
    fig.subplots_adjust(left=0.09, right=0.76, top=0.78, bottom=0.13)
    fig.text(0.09, 0.96, title, fontsize=11, color=INK, weight="bold", va="top")
    fig.text(0.09, 0.90, subtitle, fontsize=8.3, color=INK_2, va="top", linespacing=1.4)
    ax.set_ylim(*ylim)
    ax.set_xlim(-0.4, 15.4)
    ax.set_xticks(range(16))
    ax.set_xlabel("Layer (output of transformer block; 0 = first)")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(length=0)
    return fig, ax


def draw_series(ax, series: list[tuple[str, str, list[float]]], legend_anchor=(0.02, 0.02)):
    """Lines with markers, a legend, and a short direct label at each line end (<= 3 series).

    series: (legend label, short end label, values)
    """
    ends = []
    for (label, short, ys), color in zip(series, SERIES):
        xs = list(range(len(ys)))
        ax.plot(xs, ys, color=color, linewidth=1.6, marker="o", markersize=4.2,
                markeredgecolor=SURFACE, markeredgewidth=1.0, label=label, zorder=3)
        ends.append([ys[-1], short, color])
    # nudge end labels apart so they never collide
    ends.sort(key=lambda e: e[0])
    lo, hi = ax.get_ylim()
    min_gap = (hi - lo) * 0.07
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < min_gap:
            ends[i][0] = ends[i - 1][0] + min_gap
    for y, short, color in ends:
        ax.plot([15.75], [y], marker="s", markersize=4, color=color, clip_on=False, zorder=4)
        ax.annotate(short, xy=(15.75, y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=7.8, color=INK_2, annotation_clip=False)
    leg = ax.legend(loc="lower left", bbox_to_anchor=legend_anchor, frameon=False, fontsize=7.8,
                    labelcolor=INK_2, handlelength=1.6)
    return leg


def chance_line(ax, y: float, text: str):
    """Solid muted reference line, labelled in the end-label column so it never sits on a series."""
    ax.axhline(y, color=MUTED, linewidth=0.9, linestyle="-", zorder=2)
    ax.annotate(text, xy=(15.75, y), xytext=(6, 0), textcoords="offset points", va="center",
                fontsize=7.8, color=MUTED, annotation_clip=False)


def fig_fidelity() -> Path:
    res = load("02_model_fidelity.json")
    r = res["partB_forward_fidelity"]
    # "All fixes" coincides with "causal + un-permuted" to 3 decimals, so it is described, not drawn.
    series = [
        ("As used by the pipeline (all bugs)", "As used", r["J0_as_used"]["centred_cos_focus_token"]),
        ("Causal mask only (q/k still permuted)", "Causal only", r["Jc_causal_only"]["centred_cos_focus_token"]),
        ("Causal mask + q/k un-permuted", "Causal + q/k fix", r["J2_unpermuted_plus_causal"]["centred_cos_focus_token"]),
    ]
    fig, ax = base_axes(
        "The stored activations are not Llama-3.2-1B's activations",
        "Centred cosine similarity to HuggingFace transformers at the focus token, same input tokens\n"
        f"({res['partB_test_set']['n_sentences']} SemCor sentences). Adding the remaining fixes "
        "(RoPE scaling, eps) changes nothing visible.",
        "Cosine similarity to reference",
        (0.0, 1.05),
    )
    draw_series(ax, series, legend_anchor=(0.18, 0.33))
    out = FIG_DIR / "fig1_model_fidelity.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_pilot() -> Path:
    pilot = load("04_pilot_fixed_pipeline.json")
    stored = load("03_reanalysis.json")["uniform2_chunk0"]["per_layer"]["raw_dot"]["mean_norm_rank_by_layer"]
    # S_echo x G_echo was pre-registered as the headline before the pilot ran; it did not beat chance.
    # It is drawn next to the mean-pooled-gloss variant that did, so the figure shows both outcomes.
    # The kNN and left-context variants are reported in the table in REPORT.md.
    series = [
        ("Original pipeline (stored activations)", "Original",
         stored),
        ("Corrected: echo sentence vs mean-pooled gloss", "Mean-pooled gloss",
         pilot["gloss_match_S_echo_x_G_mean"]["mean_norm_rank_by_layer"]),
        ("Corrected: echo sentence vs gloss echo (pre-registered)", "Gloss echo",
         pilot["gloss_match_S_echo_x_G_echo"]["mean_norm_rank_by_layer"]),
    ]
    fig, ax = base_axes(
        "With a correct forward pass, gloss matching beats chance",
        "Mean normalised rank of the gold definition by layer (1 = always first, 0.5 = chance), on the same\n"
        f"{pilot['n_instances']} Uniform2 instances. How the gloss is represented matters.",
        "Mean normalised rank of gold",
        (0.3, 0.85),
    )
    chance_line(ax, 0.5, "chance")
    draw_series(ax, series, legend_anchor=(0.02, 0.0))
    out = FIG_DIR / "fig2_pilot.png"
    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for f in (fig_fidelity, fig_pilot):
        try:
            print("wrote", f())
        except FileNotFoundError as e:
            print("skipped", f.__name__, "-", e)
