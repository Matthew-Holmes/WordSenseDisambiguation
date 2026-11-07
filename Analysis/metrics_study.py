"""Do other similarity metrics, or cheap post-processing, pick out word senses better than the cosine?

The metrics study. The goal is better sense discrimination, judged only on sense-balanced data. Nothing here uses
sense frequency, and nothing is tuned on natural-text accuracy.

Methods (Processing/similarity.py), all starting from the layer study's centred vectors:
  Metrics       cosine (the baseline), euclidean, manhattan, pearson, spearman, dot
  Re-shaping    standardise, abtt (k = 10), whiten (top 128 directions), each followed by the cosine
  Post-proc.    csls (k = 10), gloss_z, layer_z, ensemble, standardise + gloss_z
The ensemble averages layer_z-scored cosines over all 6 target x gloss representations. Every statistic the
methods need (standard deviations, principal directions, the reference pool of target vectors) is fitted on
dev_natural without labels: its target vectors for the sentence side, its candidates' gloss vectors for the gloss
side. No balanced instance enters any fit.

Protocol, fixed before any result was seen.
  1. Primary: every method at one fixed configuration, the layer study's frozen winner (echo_last x mean gloss,
     layers 5-7; the ensemble uses all six representations at layers 5-7). Nothing is tuned on balanced data, so
     dev_balanced and test_balanced are pooled (400 instances). Each method is compared with the cosine by a
     paired, sentence-clustered bootstrap, and the 13 p-values are also Holm-adjusted.
  Secondary, reported whatever they show:
  2. Own selection: each method's dev search on dev_balanced (6 representations x 62 bands = 372 configurations;
     62 bands for the ensemble), frozen and scored on test_balanced, paired against the cosine's own
     dev_balanced winner.
  3. Sensitivity at the fixed configuration, pooled: abtt with k = 1, 3, 30; csls with k = 5, 50; gloss_center
     (gloss_z without the division by sd).
  4. Diagnostics: per-layer accuracy profiles on the pooled balanced data, at echo_last x mean gloss.

Run from the repository root, in WSL:
    .venv/bin/python Analysis/metrics_study.py
Outputs: Report/metrics_study/results.json; figures via Analysis/metrics_study_figures.py.
"""

from __future__ import annotations

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
from Processing import evaluation as ev  # noqa: E402
from Processing import similarity as sim  # noqa: E402
from Processing.extract import GLOSS_VARIANTS, SENTENCE_VARIANTS  # noqa: E402
from Processing.script import load_sample, open_stores  # noqa: E402
from Processing.semcor_data import load_dataset  # noqa: E402

OUT_DIR = REPO / "Report" / "metrics_study"
N_LAYERS, N_BOOT = ls.N_LAYERS, ls.N_BOOT
FIXED = "echo_last+mean/L5-7"  # the layer study's frozen configuration
BALANCED = ("dev_balanced", "test_balanced")
REPS = [(s, g) for s in SENTENCE_VARIANTS for g in GLOSS_VARIANTS]
PRIMARY = ["euclidean", "manhattan", "pearson", "spearman", "dot", "standardise", "abtt10", "whiten", "csls10",
           "gloss_z", "layer_z", "ensemble", "standardise+gloss_z"]
SENSITIVITY = ["abtt1", "abtt3", "abtt30", "csls5", "csls50", "gloss_center"]
ABTT_KS = (1, 3, 10, 30)
CSLS_KS = (5, 10, 50)


def chunked(fn, x: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Apply fn to x in blocks of rows (keeps temporary arrays small)."""
    return np.concatenate([fn(x[i:i + chunk]) for i in range(0, len(x), chunk)]) if len(x) else x


def demean_dims(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=-1, keepdims=True)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values."""
    order = sorted(pvalues, key=pvalues.get)
    m, running, out = len(order), 0.0, {}
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[k]))
        out[k] = running
    return out


def concat_sets(a: ev.EvalSet, b: ev.EvalSet) -> ev.EvalSet:
    return ev.EvalSet(instance_ids=a.instance_ids + b.instance_ids, clusters=a.clusters + b.clusters,
                      candidates=a.candidates + b.candidates, gold=a.gold + b.gold, pos=a.pos + b.pos)


def main() -> None:
    t0 = time.time()
    ds = load_dataset()
    sent_store, gloss_store = open_stores()
    names = ("dev_natural",) + BALANCED
    sets = {name: ls.eval_set(ds, load_sample(ls.SAMPLES[name])["instance_ids"]) for name in names}
    dev_ids = list(dict.fromkeys(sets["dev_natural"].instance_ids + sets["dev_balanced"].instance_ids))
    dev_all = ls.eval_set(ds, dev_ids)
    keys = sorted({k for es in sets.values() for c in es.candidates for k in c})
    banks = {g: ev.GlossBank(keys, gloss_store.get(keys, g)) for g in GLOSS_VARIANTS}
    idx = {name: [banks["mean"].indices(c) for c in es.candidates] for name, es in sets.items()}
    ref_idx = banks["mean"].indices(sorted({k for c in sets["dev_natural"].candidates for k in c}))

    mu_s, mu_g = {}, {}  # the layer study's centring (dev_natural + dev_balanced), shared by every method
    for s in SENTENCE_VARIANTS:
        dev_vecs = sent_store.get(dev_ids, s)
        for g in GLOSS_VARIANTS:
            c = ev.fit_centering(dev_vecs, banks[g], dev_all)
            mu_s[s], mu_g[g] = c.mu_sentence.astype(np.float32), c.mu_gloss.astype(np.float32)
    sent = {s: {name: sent_store.get(es.instance_ids, s).astype(np.float32) - mu_s[s] for name, es in sets.items()}
            for s in SENTENCE_VARIANTS}
    gl = {g: banks[g].vectors.astype(np.float32) - mu_g[g] for g in GLOSS_VARIANTS}
    sfit = {s: sim.fit_space(sent[s]["dev_natural"]) for s in SENTENCE_VARIANTS}
    gfit = {g: sim.fit_space(gl[g][ref_idx]) for g in GLOSS_VARIANTS}
    print(f"loaded and fitted ({time.time() - t0:.0f}s): {len(keys)} glosses, reference {len(ref_idx)} glosses "
          f"and {len(sets['dev_natural'])} targets")

    # ------------------------------------------------------------------ per-layer scores for every method
    transforms = {
        "none": (lambda x, s: x, lambda x, g: x),
        "pearson": (lambda x, s: demean_dims(x), lambda x, g: demean_dims(x)),
        "spearman": (lambda x, s: sim.centred_ranks(x), lambda x, g: sim.centred_ranks(x)),
        "standardise": (lambda x, s: sim.standardise(x, sfit[s]), lambda x, g: sim.standardise(x, gfit[g])),
        "whiten": (lambda x, s: sim.whiten(x, sfit[s]), lambda x, g: sim.whiten(x, gfit[g])),
    }
    for k in ABTT_KS:
        transforms[f"abtt{k}"] = (lambda x, s, k=k: sim.abtt(x, sfit[s], k), lambda x, g, k=k: sim.abtt(x, gfit[g], k))

    per_layer = {}  # (method, set, s, g) -> [(n_i, L), ...]

    def per_instance(name, s_vecs, g_vecs, fn):
        return [fn(s_vecs[name][i], g_vecs[idx[name][i]]) for i in range(len(sets[name]))]

    def cosine(h, g):
        return sim.metric_scores(h, g, "cosine")

    for tname, (t_s, t_g) in transforms.items():
        for g in GLOSS_VARIANTS:
            gt = chunked(lambda x: t_g(x, g), gl[g])
            for s in SENTENCE_VARIANTS:
                st = {name: t_s(sent[s][name], s) for name in names}
                cos = {name: per_instance(name, st, gt, cosine) for name in BALANCED}
                method = {"none": "cosine"}.get(tname, tname)
                for name in BALANCED:
                    per_layer[(method, name, s, g)] = cos[name]
                if tname == "none":
                    for metric in ("euclidean", "manhattan", "dot"):
                        for name in BALANCED:
                            per_layer[(metric, name, s, g)] = per_instance(
                                name, st, gt, lambda h, gg, metric=metric: sim.metric_scores(h, gg, metric))
                    for name in BALANCED:
                        per_layer[("layer_z", name, s, g)] = [sim.layer_z(c) for c in cos[name]]
                if tname in ("none", "standardise"):
                    stats = sim.gloss_stats(st["dev_natural"], gt, ks=CSLS_KS)
                    for name in BALANCED:
                        rows = [(c, ix) for c, ix in zip(cos[name], idx[name])]
                        if tname == "none":
                            for k in CSLS_KS:
                                per_layer[(f"csls{k}", name, s, g)] = [sim.csls(c, stats.knn[k][ix]) for c, ix in rows]
                            per_layer[("gloss_z", name, s, g)] = [sim.gloss_z(c, stats.mean[ix], stats.sd[ix]) for c, ix in rows]
                            per_layer[("gloss_center", name, s, g)] = [c - stats.mean[ix] for c, ix in rows]
                        else:
                            per_layer[("standardise+gloss_z", name, s, g)] = [
                                sim.gloss_z(c, stats.mean[ix], stats.sd[ix]) for c, ix in rows]
            del gt
        print(f"  scored transform {tname!r} ({time.time() - t0:.0f}s)")
    for name in BALANCED:  # the ensemble: z-scored cosines averaged over the six representations
        per_layer[("ensemble", name, "all", "all")] = [
            np.mean([sim.layer_z(per_layer[("cosine", name, s, g)][i]) for s, g in REPS], axis=0)
            for i in range(len(sets[name]))]

    # sanity checks: the cosine is the layer study's scorer, and the transformed metrics equal the direct ones
    cen = ev.Centering(mu_s["echo_last"], mu_g["mean"])
    old = ev.layer_scores(sent_store.get(sets["test_balanced"].instance_ids, "echo_last"), banks["mean"],
                          sets["test_balanced"], cen)
    checks = {"max_abs_cosine_vs_layer_scores": max(float(np.abs(a - b).max()) for a, b in
                                                    zip(per_layer[("cosine", "test_balanced", "echo_last", "mean")], old))}
    for metric in ("pearson", "spearman"):
        checks[f"max_abs_{metric}_vs_direct"] = max(float(np.abs(
            per_layer[(metric, "test_balanced", "echo_last", "mean")][i]
            - sim.metric_scores(sent["echo_last"]["test_balanced"][i], gl["mean"][idx["test_balanced"][i]], metric)).max())
            for i in range(10))
    print("checks:", checks)

    # ------------------------------------------------------------------ evaluation helpers
    bands = ev.contiguous_bands(N_LAYERS)
    pooled = concat_sets(sets["dev_balanced"], sets["test_balanced"])

    def fixed_cfg(m):
        return "all+all/L5-7" if m == "ensemble" else FIXED

    def reps_of(m):
        return [("all", "all")] if m == "ensemble" else REPS

    def scores_at(m, name, cfg):
        rep, band = cfg.split("/")
        s, g = rep.split("+")
        layers = bands[band] if band in bands else [int(band[1:])]
        return ev.config_scores(per_layer[(m, name, s, g)], layers)

    def metrics_at(m, name, cfg):
        return ev.instance_metrics(scores_at(m, name, cfg), sets[name])

    def pooled_metrics(m, cfg):
        parts = [metrics_at(m, name, cfg) for name in BALANCED]
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    def paired(values, base, es):
        return ev.paired_bootstrap(values, base, es.clusters, n_boot=N_BOOT)

    # ------------------------------------------------------------------ 1. primary: fixed configuration, pooled
    base = pooled_metrics("cosine", FIXED)
    fixed = {"cosine": {"config": FIXED, "pooled": ls.summary(base, pooled),
                        "by_set": {n: float(np.mean(metrics_at("cosine", n, FIXED)["acc1"])) for n in BALANCED}}}
    for m in PRIMARY + SENSITIVITY:
        met = pooled_metrics(m, fixed_cfg(m))
        fixed[m] = {"config": fixed_cfg(m), "pooled": ls.summary(met, pooled),
                    "by_set": {n: float(np.mean(metrics_at(m, n, fixed_cfg(m))["acc1"])) for n in BALANCED},
                    "paired_vs_cosine_acc1": paired(met["acc1"], base["acc1"], pooled),
                    "changes_pick": float(np.mean([int(np.argmax(a)) != int(np.argmax(b)) for name in BALANCED
                                                   for a, b in zip(scores_at(m, name, fixed_cfg(m)),
                                                                   scores_at("cosine", name, FIXED))]))}
    adjusted = holm({m: fixed[m]["paired_vs_cosine_acc1"]["p_two_sided"] for m in PRIMARY})
    for m in PRIMARY:
        fixed[m]["holm_p"] = adjusted[m]
    fixed["random"] = {"pooled": ls.summary(ev.random_baseline_exact(pooled), pooled)}
    print(f"cosine at {FIXED}: pooled {np.mean(base['acc1']):.3f} "
          + " ".join(f"{n} {fixed['cosine']['by_set'][n]:.3f}" for n in BALANCED))
    for m in PRIMARY + SENSITIVITY:
        p = fixed[m]["paired_vs_cosine_acc1"]
        print(f"  {m:20s} pooled {fixed[m]['pooled']['acc1']['mean']:.3f}  diff {p['diff']:+.3f} "
              f"[{p['lo']:+.3f},{p['hi']:+.3f}] p={p['p_two_sided']:.3f}"
              + (f" holm={fixed[m]['holm_p']:.3f}" if m in adjusted else ""))

    # ------------------------------------------------------------------ 2. own selection on dev_balanced
    test_es = sets["test_balanced"]
    selection = {}
    for m in ["cosine"] + PRIMARY:
        grid = {}
        for s, g in reps_of(m):
            res = ev.evaluate_configs(per_layer[(m, "dev_balanced", s, g)], sets["dev_balanced"], bands)
            grid.update({f"{s}+{g}/{band}": r for band, r in res.items()})
        cfg = ev.select_best(grid, "acc1")
        ranked = sorted(grid, key=lambda k: -np.mean(grid[k]["acc1"]))
        selection[m] = {"config": cfg, "dev_acc1": float(np.mean(grid[cfg]["acc1"])),
                        "top5": [{"config": k, "dev_acc1": float(np.mean(grid[k]["acc1"]))} for k in ranked[:5]],
                        "_test": metrics_at(m, "test_balanced", cfg)}
    base_sel = selection["cosine"]["_test"]["acc1"]
    for m in selection:
        met = selection[m].pop("_test")
        selection[m]["test"] = ls.summary(met, test_es)
        if m != "cosine":
            selection[m]["paired_vs_cosine_acc1"] = paired(met["acc1"], base_sel, test_es)
    print(f"own selection: cosine {selection['cosine']['config']} dev {selection['cosine']['dev_acc1']:.3f} "
          f"test {selection['cosine']['test']['acc1']['mean']:.3f}")
    for m in PRIMARY:
        p = selection[m]["paired_vs_cosine_acc1"]
        print(f"  {m:20s} {selection[m]['config']:22s} dev {selection[m]['dev_acc1']:.3f} "
              f"test {selection[m]['test']['acc1']['mean']:.3f} diff {p['diff']:+.3f} p={p['p_two_sided']:.3f}")

    # ------------------------------------------------------------------ 4. per-layer profiles, pooled
    profiles = {}
    for m in ["cosine"] + PRIMARY + SENSITIVITY:
        rep = "all+all" if m == "ensemble" else FIXED.split("/")[0]
        profiles[m] = [float(np.mean(pooled_metrics(m, f"{rep}/L{i}")["acc1"])) for i in range(N_LAYERS)]

    results = {
        "protocol": __doc__,
        "fixed_config": FIXED,
        "samples": {n: {"n_instances": len(sets[n]), "n_sentences": len(set(sets[n].clusters)),
                        "mean_candidates": float(np.mean(sets[n].n_candidates))} for n in names},
        "reference": {"targets": len(sets["dev_natural"]), "glosses": int(len(ref_idx))},
        "checks": checks,
        "primary_methods": PRIMARY, "sensitivity_methods": SENSITIVITY,
        "fixed": fixed,
        "selection": selection,
        "profiles": profiles,
        "runtime_seconds": time.time() - t0,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=1, default=float))
    print(f"saved {OUT_DIR / 'results.json'} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
