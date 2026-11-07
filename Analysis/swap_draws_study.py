"""Does averaging more swapped sentences make the swap contrast work? (the swap-draws study)

A follow-up to the swap study (Analysis/swap_study.py). Each target's sentence is re-run with the target replaced
by K random words of the same part of speech (draws 0-4 of Processing/swaps.py, match "pos"). The K swapped-slot
vectors are averaged, and the average is subtracted from the original sentence's vector (method 1); methods 2 and
3 use the K draws in their own way. Judged on sense-balanced data only.

Scores (Processing/contrastive.py), per layer, averaged over a band:
    baseline   cos(h~, g~)                          the layer study's score
    diff       cos(h~ - mean_j s~_j, g~)            method 1: average the swaps, subtract, then cosine
    normdiff   cos(h^ - mean_j s^_j, g~)            method 2: normalise, average, subtract, then cosine
    scorediff  cos(h~, g~) - mean_j cos(s~_j, g~)   method 3: cosine with each, then subtract the average
    swap       mean_j cos(s~_j, g~)                 diagnostic: the swapped slot alone

Protocol, fixed before any result was seen.
  1. Primary: the layer study's frozen configuration (echo_last x mean gloss, layers 5-7). Nothing is tuned on
     balanced data, so dev_balanced and test_balanced are pooled (400 instances). For K = 1 to 5, using draws
     0 to K-1, each method is compared with the baseline by a paired, sentence-clustered bootstrap. Primary
     question: does method 1 with K = 5 beat the baseline? The trend over K is read from the same table.
  Secondary, reported whatever they show:
  2. Which words were drawn: for each K, every subset of K of the 5 draws (5, 10, 10, 5 and 1 subsets), with
     the mean, lowest and highest pooled accuracy over subsets.
  3. Own selection, K = 5: each method's dev search on dev_balanced (6 representations x 62 bands), frozen and
     scored on test_balanced, paired against the baseline's own dev_balanced winner.
  4. Geometry: for each K, the centred cosine between h and the averaged slot, and the ratio of their lengths, by
     layer, at echo_last.
The natural-text samples are not used: only draws 0 and 1 exist for them.

Run after extracting the extra draws:
    .venv/bin/python -u -m Processing.script extract --samples dev_balanced_200 test_balanced_200 --swaps pos:2-4
    .venv/bin/python Analysis/swap_draws_study.py
Outputs: Report/swap_draws_study/results.json; figures via Analysis/swap_draws_figures.py.
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "Analysis"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import layer_study as ls  # noqa: E402
from Processing import contrastive as ct  # noqa: E402
from Processing import evaluation as ev  # noqa: E402
from Processing.extract import GLOSS_VARIANTS, SENTENCE_VARIANTS  # noqa: E402
from Processing.script import load_sample, open_stores, open_swap_store  # noqa: E402
from Processing.semcor_data import load_dataset  # noqa: E402
from Processing.swaps import swap_key  # noqa: E402

OUT_DIR = REPO / "Report" / "swap_draws_study"
N_LAYERS, N_BOOT = ls.N_LAYERS, ls.N_BOOT
FIXED = "echo_last+mean/L5-7"  # the layer study's frozen configuration
BALANCED = ("dev_balanced", "test_balanced")
DRAWS = (0, 1, 2, 3, 4)
CONTRASTS = ("diff", "normdiff", "scorediff")
METHODS = CONTRASTS + ("swap",)


def concat_sets(a: ev.EvalSet, b: ev.EvalSet) -> ev.EvalSet:
    return ev.EvalSet(instance_ids=a.instance_ids + b.instance_ids, clusters=a.clusters + b.clusters,
                      candidates=a.candidates + b.candidates, gold=a.gold + b.gold, pos=a.pos + b.pos)


def main() -> None:
    t0 = time.time()
    ds = load_dataset()
    sent_store, gloss_store = open_stores()
    swap_store = open_swap_store()
    names = ("dev_natural",) + BALANCED
    sets = {name: ls.eval_set(ds, load_sample(ls.SAMPLES[name])["instance_ids"]) for name in names}
    dev_ids = list(dict.fromkeys(sets["dev_natural"].instance_ids + sets["dev_balanced"].instance_ids))
    dev_all = ls.eval_set(ds, dev_ids)
    keys = sorted({k for es in sets.values() for c in es.candidates for k in c})
    banks = {g: ev.GlossBank(keys, gloss_store.get(keys, g)) for g in GLOSS_VARIANTS}
    centerings = {}
    for s in SENTENCE_VARIANTS:  # the layer study's centring
        dev_vecs = sent_store.get(dev_ids, s)
        for g in GLOSS_VARIANTS:
            centerings[(s, g)] = ev.fit_centering(dev_vecs, banks[g], dev_all)

    orig, swaps = {}, {}  # (set, s) -> (N, L, D) and (N, 5, L, D)
    for name in BALANCED:
        es = sets[name]
        wanted = [swap_key(iid, pos, j, "pos") for iid, pos in zip(es.instance_ids, es.pos) for j in DRAWS]
        missing = [k for k in wanted if k not in swap_store]
        if missing:
            raise SystemExit(f"{len(missing)} swapped sentences missing for {name}, e.g. {missing[0]}; extract them first")
        for s in SENTENCE_VARIANTS:
            orig[(name, s)] = sent_store.get(es.instance_ids, s)
            flat = swap_store.get(wanted, s)
            swaps[(name, s)] = flat.reshape(len(es), len(DRAWS), *flat.shape[1:])
    print(f"loaded ({time.time() - t0:.0f}s)")

    bands = ev.contiguous_bands(N_LAYERS)
    pooled = concat_sets(sets["dev_balanced"], sets["test_balanced"])
    rep, band = FIXED.split("/")
    fs, fg = rep.split("+")

    def score(name, s, g, subset):
        return ct.contrastive_layer_scores(orig[(name, s)], swaps[(name, s)][:, list(subset)], banks[g], sets[name],
                                           centerings[(s, g)])

    def pooled_metrics(per_set):  # per_set: {name: [(n_i, L)]} -> pooled metrics at the fixed band
        parts = [ev.instance_metrics(ev.config_scores(per_set[name], bands[band]), sets[name]) for name in BALANCED]
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    # ------------------------------------------------------------------ 1-2. every subset of draws, fixed configuration
    subsets = [c for k in range(1, len(DRAWS) + 1) for c in itertools.combinations(DRAWS, k)]
    by_subset = {}  # subset -> {method: pooled metrics}
    base = None
    for subset in subsets:
        per = {name: score(name, fs, fg, subset) for name in BALANCED}
        if base is None:
            base = pooled_metrics({name: per[name]["baseline"] for name in BALANCED})
        by_subset[subset] = {m: pooled_metrics({name: per[name][m] for name in BALANCED}) for m in METHODS}
    print(f"scored {len(subsets)} subsets of draws ({time.time() - t0:.0f}s)")

    nested, subset_stats = {}, {}
    for K in range(1, len(DRAWS) + 1):
        sub = tuple(DRAWS[:K])
        nested[K] = {}
        for m in METHODS:
            met = by_subset[sub][m]
            nested[K][m] = {"pooled": ls.summary(met, pooled),
                            "paired_vs_baseline_acc1": ev.paired_bootstrap(met["acc1"], base["acc1"], pooled.clusters,
                                                                           n_boot=N_BOOT),
                            "by_set": {n: float(np.mean(met["acc1"][:len(sets["dev_balanced"])] if n == "dev_balanced"
                                                        else met["acc1"][len(sets["dev_balanced"]):])) for n in BALANCED}}
        accs = {m: [float(np.mean(by_subset[c][m]["acc1"])) for c in subsets if len(c) == K] for m in METHODS}
        subset_stats[K] = {m: {"n_subsets": len(a), "mean": float(np.mean(a)), "min": float(np.min(a)),
                               "max": float(np.max(a))} for m, a in accs.items()}
        line = "  ".join(f"{m} {np.mean(by_subset[sub][m]['acc1']):.3f} "
                         f"({nested[K][m]['paired_vs_baseline_acc1']['diff']:+.3f}, "
                         f"p={nested[K][m]['paired_vs_baseline_acc1']['p_two_sided']:.3f})" for m in CONTRASTS)
        print(f"K={K}: baseline {np.mean(base['acc1']):.3f}  {line}  swap {np.mean(by_subset[sub]['swap']['acc1']):.3f}")

    # ------------------------------------------------------------------ 3. own selection with K = 5
    full = tuple(DRAWS)
    per_rep = {(name, s, g): score(name, s, g, full) for name in BALANCED for s in SENTENCE_VARIANTS for g in GLOSS_VARIANTS}

    def select(m):
        grid = {}
        for s in SENTENCE_VARIANTS:
            for g in GLOSS_VARIANTS:
                res = ev.evaluate_configs(per_rep[("dev_balanced", s, g)][m], sets["dev_balanced"], bands)
                grid.update({f"{s}+{g}/{b}": r for b, r in res.items()})
        cfg = ev.select_best(grid, "acc1")
        r_, b_ = cfg.split("/")
        s_, g_ = r_.split("+")
        test = ev.instance_metrics(ev.config_scores(per_rep[("test_balanced", s_, g_)][m], bands[b_]),
                                   sets["test_balanced"])
        return cfg, float(np.mean(grid[cfg]["acc1"])), test

    test_es = sets["test_balanced"]
    b_cfg, b_dev, b_test = select("baseline")
    selection = {"baseline": {"config": b_cfg, "dev_acc1": b_dev, "test": ls.summary(b_test, test_es)}}
    for m in CONTRASTS:
        cfg, dev_acc, test = select(m)
        selection[m] = {"config": cfg, "dev_acc1": dev_acc, "test": ls.summary(test, test_es),
                        "paired_vs_baseline_acc1": ev.paired_bootstrap(test["acc1"], b_test["acc1"], test_es.clusters,
                                                                       n_boot=N_BOOT)}
        print(f"  K=5 own selection {m:9s} {cfg:22s} dev {dev_acc:.3f} test {selection[m]['test']['acc1']['mean']:.3f} "
              f"(baseline {b_cfg} test {selection['baseline']['test']['acc1']['mean']:.3f})")

    # ------------------------------------------------------------------ 4. geometry of the averaged slot
    geometry = {}
    mu = centerings[("echo_last", "mean")].mu_sentence.astype(np.float32)
    h = np.concatenate([orig[(n, "echo_last")] for n in BALANCED]).astype(np.float32) - mu  # (N, L, D)
    sw = np.concatenate([swaps[(n, "echo_last")] for n in BALANCED]).astype(np.float32)  # (N, 5, L, D)
    for K in range(1, len(DRAWS) + 1):
        sbar = sw[:, :K].mean(axis=1) - mu
        geometry[K] = {"cos_h_slot": (ct.unit(h) * ct.unit(sbar)).sum(-1).mean(0).tolist(),
                       "norm_ratio_slot_h": (np.linalg.norm(sbar, axis=-1) / np.linalg.norm(h, axis=-1)).mean(0).tolist()}

    results = {
        "protocol": __doc__,
        "fixed_config": FIXED,
        "draws": list(DRAWS),
        "samples": {n: {"n_instances": len(sets[n]), "n_sentences": len(set(sets[n].clusters))} for n in BALANCED},
        "baseline_fixed": {"pooled": ls.summary(base, pooled),
                           "by_set": {"dev_balanced": float(np.mean(base["acc1"][:len(sets["dev_balanced"])])),
                                      "test_balanced": float(np.mean(base["acc1"][len(sets["dev_balanced"]):]))}},
        "random_pooled": ls.summary(ev.random_baseline_exact(pooled), pooled),
        "nested": {str(K): v for K, v in nested.items()},
        "subset_stats": {str(K): v for K, v in subset_stats.items()},
        "selection_k5": selection,
        "geometry": {str(K): v for K, v in geometry.items()},
        "runtime_seconds": time.time() - t0,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=1, default=float))
    print(f"saved {OUT_DIR / 'results.json'} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
