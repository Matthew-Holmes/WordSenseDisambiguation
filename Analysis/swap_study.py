"""Does contrasting with a swapped-word control sharpen gloss matching? (the swap study)

Hypothesis: the cosine between a target's hidden state and a gloss vector mixes semantic similarity with
grammatical similarity. Re-running each sentence with the target replaced by a random word of the same part of
speech keeps the grammar but changes the meaning, so contrasting the two runs should remove the grammatical part.

Scores (Processing/contrastive.py), computed per layer and averaged over a band of layers:
    baseline   cos(h~, g~)                         the layer study's score
    diff       cos(h~ - s~, g~)                    method 1: subtract, then cosine
    normdiff   cos(h^ - s^, g~)                    method 2: normalise, subtract, then cosine
    scorediff  cos(h~, g~) - cos(s~, g~)           method 3: cosine with each, then subtract
    swap       cos(s~, g~)                         diagnostic: the slot holding a random word, alone
(h = target in the original sentence, s = the same slot in the swapped sentence, g = gloss; ~ = centred with the
layer study's dev means, ^ = centred and unit length.)

Protocol, fixed before any result was seen. Every choice is made on dev documents; test documents are scored once.
  1. Vectors: the layer study's sentence and gloss vectors, plus one swapped sentence per target from
     Processing/swaps.py (match="pos", draw 0; the design as specified). The centring is the layer study's.
  2. For each score, search dev_natural: 3 sentence representations x 2 gloss representations x 62 contiguous
     bands of 1-4 layers = 372 configurations, ranked by accuracy@1 with evaluation.select_best. The baseline
     search is the layer study's own and must reproduce its winner.
  3. Validate on test_natural and test_balanced. Each score keeps its frozen configuration. Results carry
     cluster-bootstrap CIs and paired comparisons against the frozen baseline. Primary question: does any of
     methods 1-3 beat the baseline on test?
  Secondary analyses, reported whatever they show:
  4. Same configuration: each method at the baseline's frozen configuration, which removes selection noise.
  5. Inflection-matched swaps (match="pos+form", draw 0) in place of POS-only ones; steps 2-4 repeated.
  6. A second POS-matched draw (draw 1) on its own, as a replication with different random words, and draws 0
     and 1 averaged; steps 2-4 repeated for each.
  7. The family baseline - beta * swap (beta = 0: baseline, beta = 1: method 3) at the baseline's configuration,
     with beta tuned on dev_natural (grid -1.0 to 1.5 in steps of 0.1; ties go to the beta nearest 0).
  8. Diagnostics: per-layer accuracy of the swap score alone; per-layer cosine between h and s and their length
     ratio; how often methods 2 and 3 pick different senses.
  Exploratory, added after the primary result was seen: how often the baseline's pick is also the swapped slot's
  pick, split by whether the baseline is right, against the rate expected if the two were independent.

Run after the swapped sentences are extracted:
    .venv/bin/python -m Processing.script extract --samples dev_natural_80 dev_balanced_200 test_natural_80 \\
        test_balanced_200 --swaps pos:0 pos+form:0 pos:1
    .venv/bin/python Analysis/swap_study.py
Outputs: Report/swap_study/results.json; figures via Analysis/swap_study_figures.py.
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
from Processing import contrastive as ct  # noqa: E402
from Processing import evaluation as ev  # noqa: E402
from Processing.extract import GLOSS_VARIANTS, SENTENCE_VARIANTS  # noqa: E402
from Processing.script import load_sample, open_stores, open_swap_store  # noqa: E402
from Processing.semcor_data import load_dataset  # noqa: E402
from Processing.swaps import needed_pos, swap_key  # noqa: E402

OUT_DIR = REPO / "Report" / "swap_study"
N_LAYERS, N_BOOT = ls.N_LAYERS, ls.N_BOOT
CONDITIONS = {  # name -> (match mode, draws averaged)
    "pos": ("pos", (0,)),  # primary: one random word of the same part of speech
    "pos+form": ("pos+form", (0,)),  # secondary: also the same inflection class
    "pos draw 1": ("pos", (1,)),  # secondary: a different random word, same rule (replication)
    "pos x2": ("pos", (0, 1)),  # secondary: two POS-matched words, averaged
}
PRIMARY = "pos"
CONTRASTS = ("diff", "normdiff", "scorediff")
SCORES = ("baseline",) + CONTRASTS + ("swap",)
BETAS = tuple(float(b) + 0.0 for b in np.round(np.arange(-1.0, 1.51, 0.1), 2))  # + 0.0 turns -0.0 into 0.0


def load_swaps(store, es: ev.EvalSet, variant: str, match: str, draws) -> np.ndarray:
    """(N, K, L, D) float16 swapped-slot vectors, rows aligned with ``es``, one column per draw."""
    keys = [swap_key(iid, pos, j, match) for iid, pos in zip(es.instance_ids, es.pos) for j in draws]
    flat = store.get(keys, variant)
    return flat.reshape(len(es), len(draws), *flat.shape[1:])


def top_pick(scores: list[np.ndarray]) -> np.ndarray:
    return np.array([int(np.argmax(s)) for s in scores])


def main() -> None:
    t0 = time.time()
    ds = load_dataset()
    sent_store, gloss_store = open_stores()
    swap_store = open_swap_store()
    sets = {name: ls.eval_set(ds, load_sample(sample)["instance_ids"]) for name, sample in ls.SAMPLES.items()}
    # v2 candidates share the target's POS, so each target has exactly one swap set per draw
    assert all(needed_pos(ds.instances[i]["candidates"]) == [ds.instances[i]["pos"]]
               for es in sets.values() for i in es.instance_ids)

    keys = sorted({k for es in sets.values() for c in es.candidates for k in c})
    banks = {g: ev.GlossBank(keys, gloss_store.get(keys, g)) for g in GLOSS_VARIANTS}
    vecs = {name: {s: sent_store.get(es.instance_ids, s) for s in SENTENCE_VARIANTS} for name, es in sets.items()}
    dev_ids = list(dict.fromkeys(sets["dev_natural"].instance_ids + sets["dev_balanced"].instance_ids))
    dev_all = ls.eval_set(ds, dev_ids)
    centerings = {}
    for s in SENTENCE_VARIANTS:
        dev_vecs = sent_store.get(dev_ids, s)
        for g in GLOSS_VARIANTS:
            centerings[(s, g)] = ev.fit_centering(dev_vecs, banks[g], dev_all)
    print(f"loaded originals and glosses ({time.time() - t0:.0f}s)")

    # ------------------------------------------------------------------ per-layer scores
    per_layer = {}  # (condition, set, s, g) -> {score: [(n_i, L), ...]}
    swaps = {}  # (condition, set, s) -> (N, K, L, D)
    for cond, (match, draws) in CONDITIONS.items():
        for name, es in sets.items():
            for s in SENTENCE_VARIANTS:
                swaps[(cond, name, s)] = load_swaps(swap_store, es, s, match, draws)
                for g in GLOSS_VARIANTS:
                    per_layer[(cond, name, s, g)] = ct.contrastive_layer_scores(
                        vecs[name][s], swaps[(cond, name, s)], banks[g], es, centerings[(s, g)])
        print(f"scored condition {cond!r} ({time.time() - t0:.0f}s)")

    bands = ev.contiguous_bands(N_LAYERS)
    singles = {f"L{i}": [i] for i in range(N_LAYERS)}

    def scores_at(cond, name, score, config):
        rep, band = config.split("/")
        s, g = rep.split("+")
        return ev.config_scores(per_layer[(cond, name, s, g)][score], bands[band])

    def metrics_at(cond, name, score, config):
        return ev.instance_metrics(scores_at(cond, name, score, config), sets[name])

    def dev_grid(cond, score):
        grid = {}
        for s in SENTENCE_VARIANTS:
            for g in GLOSS_VARIANTS:
                res = ev.evaluate_configs(per_layer[(cond, "dev_natural", s, g)][score], sets["dev_natural"], bands)
                grid.update({f"{s}+{g}/{band}": m for band, m in res.items()})
        return grid

    # ------------------------------------------------------------------ dev selection
    base_grid = dev_grid(PRIMARY, "baseline")
    base_cfg = ev.select_best(base_grid, "acc1")
    layer_study = json.loads((REPO / "Report" / "layer_study" / "results.json").read_text())
    reproduces = base_cfg == layer_study["dev_selection"]["winner"]
    print(f"baseline dev winner {base_cfg} (layer study: {layer_study['dev_selection']['winner']}; "
          f"reproduced: {reproduces})")
    selection = {"baseline": {"config": base_cfg, "dev_acc1": ls.summary(base_grid[base_cfg], sets["dev_natural"])["acc1"]}}
    selected = {}
    for cond in CONDITIONS:
        selection[cond] = {}
        for score in CONTRASTS + ("swap",):
            grid = dev_grid(cond, score)
            cfg = ev.select_best(grid, "acc1")
            selected[(cond, score)] = cfg
            ranked = sorted(grid, key=lambda k: -np.mean(grid[k]["acc1"]))
            selection[cond][score] = {
                "config": cfg, "dev_acc1": ls.summary(grid[cfg], sets["dev_natural"])["acc1"],
                "dev_acc1_at_baseline_config": float(np.mean(grid[base_cfg]["acc1"])),
                "top5": [{"config": k, "dev_acc1": float(np.mean(grid[k]["acc1"]))} for k in ranked[:5]],
            }
            print(f"  {cond:9s} {score:9s} dev winner {cfg:22s} acc1 {np.mean(grid[cfg]['acc1']):.3f}  "
                  f"(at baseline config {np.mean(grid[base_cfg]['acc1']):.3f})")

    # ------------------------------------------------------------------ beta family (tuned on dev)
    beta_curves, best_beta = {}, {}
    rep, band = base_cfg.split("/")
    bs, bg = rep.split("+")
    for cond in CONDITIONS:
        beta_curves[cond] = {}
        for name in sets:
            pl = per_layer[(cond, name, bs, bg)]
            beta_curves[cond][name] = [
                float(np.mean(ev.instance_metrics(ev.config_scores(ct.with_beta(pl, b), bands[band]), sets[name])["acc1"]))
                for b in BETAS]
        dev_curve = beta_curves[cond]["dev_natural"]
        best_beta[cond] = max(BETAS, key=lambda b: (dev_curve[BETAS.index(b)], -abs(b)))
        print(f"  {cond:9s} beta tuned on dev: {best_beta[cond]} (dev acc1 {dev_curve[BETAS.index(best_beta[cond])]:.3f})")

    def beta_metrics(cond, name, b):
        pl = per_layer[(cond, name, bs, bg)]
        return ev.instance_metrics(ev.config_scores(ct.with_beta(pl, b), bands[band]), sets[name])

    # ------------------------------------------------------------------ validation
    validation = {}
    for name in ("test_natural", "test_balanced", "dev_natural", "dev_balanced"):
        es = sets[name]
        systems = {"baseline": metrics_at(PRIMARY, name, "baseline", base_cfg)}
        for cond in CONDITIONS:
            for score in CONTRASTS:
                systems[f"{cond}|{score}|own"] = metrics_at(cond, name, score, selected[(cond, score)])
                systems[f"{cond}|{score}|base_cfg"] = metrics_at(cond, name, score, base_cfg)
            systems[f"{cond}|swap|own"] = metrics_at(cond, name, "swap", selected[(cond, "swap")])
            systems[f"{cond}|swap|base_cfg"] = metrics_at(cond, name, "swap", base_cfg)
            systems[f"{cond}|beta|base_cfg"] = beta_metrics(cond, name, best_beta[cond])
        systems["MFS from dev documents"] = ls.dev_mfs_baseline(ds, es)
        systems["WordNet first sense (optimistic)"] = ev.first_sense_baseline(es)
        systems["random"] = ev.random_baseline_exact(es)
        base = systems["baseline"]["acc1"]
        validation[name] = {
            "n_instances": len(es), "n_sentences": len(set(es.clusters)),
            "systems": {k: ls.summary(v, es) for k, v in systems.items()},
            "paired_vs_baseline_acc1": {k: ev.paired_bootstrap(v["acc1"], base, es.clusters, n_boot=N_BOOT)
                                        for k, v in systems.items() if k != "baseline"},
            "acc1_by_pos": {k: ev.by_group(systems[k]["acc1"], es.pos)
                            for k in ["baseline"] + [f"{PRIMARY}|{m}|own" for m in CONTRASTS]},
        }
        if name.startswith("test"):
            line = "  ".join(f"{m} {validation[name]['systems'][f'{PRIMARY}|{m}|own']['acc1']['mean']:.3f}"
                             for m in CONTRASTS)
            print(f"{name}: baseline {validation[name]['systems']['baseline']['acc1']['mean']:.3f}  {line}")

    # ------------------------------------------------------------------ profiles and diagnostics
    profiles = {}
    for cond in CONDITIONS:
        for name in sets:
            for s in SENTENCE_VARIANTS:
                for g in GLOSS_VARIANTS:
                    if cond != PRIMARY and f"{s}+{g}" != rep:
                        continue
                    for score in SCORES:
                        res = ev.evaluate_configs(per_layer[(cond, name, s, g)][score], sets[name], singles)
                        profiles[f"{cond}|{name}|{s}+{g}|{score}"] = [float(np.mean(res[f"L{i}"]["acc1"]))
                                                                      for i in range(N_LAYERS)]

    geometry = {}  # how different is the swapped slot from the original one?
    for name, es in sets.items():
        for s in SENTENCE_VARIANTS:
            h = vecs[name][s].astype(np.float32)  # (N, L, D)
            sw = swaps[(PRIMARY, name, s)][:, 0].astype(np.float32)
            mu = centerings[(s, GLOSS_VARIANTS[0])].mu_sentence
            hc, sc = h - mu, sw - mu
            geometry[f"{name}|{s}"] = {
                "cos_raw": (ct.unit(h) * ct.unit(sw)).sum(-1).mean(0).tolist(),
                "cos_centred": (ct.unit(hc) * ct.unit(sc)).sum(-1).mean(0).tolist(),
                "norm_ratio_centred": (np.linalg.norm(sc, axis=-1) / np.linalg.norm(hc, axis=-1)).mean(0).tolist(),
            }

    disagreement = {}  # methods 2 vs 3: share of instances whose top pick differs
    for cond in CONDITIONS:
        for name in sets:
            for label, cfg2, cfg3 in (("base_cfg", base_cfg, base_cfg),
                                      ("own", selected[(cond, "normdiff")], selected[(cond, "scorediff")])):
                a = top_pick(scores_at(cond, name, "normdiff", cfg2))
                b = top_pick(scores_at(cond, name, "scorediff", cfg3))
                disagreement[f"{cond}|{name}|{label}"] = float(np.mean(a != b))
    for name in sets:  # single layers: identical by construction (Processing/contrastive.py)
        disagreement[f"{PRIMARY}|{name}|single_layer_max"] = max(
            float(np.mean(top_pick(scores_at(PRIMARY, name, "normdiff", f"{rep}/L{i}"))
                          != top_pick(scores_at(PRIMARY, name, "scorediff", f"{rep}/L{i}")))) for i in range(N_LAYERS))

    # Exploratory, added after the primary result was seen: does the swapped slot drive the baseline's errors?
    # If grammatical similarity caused the wrong picks, the baseline's wrong pick would often be the sense that the
    # swapped slot alone picks.
    def spearman(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])

    alignment = {}
    for name, es in sets.items():
        base_sc = scores_at(PRIMARY, name, "baseline", base_cfg)
        swap_sc = scores_at(PRIMARY, name, "swap", base_cfg)
        bp, wp = top_pick(base_sc), top_pick(swap_sc)
        m3 = top_pick(scores_at(PRIMARY, name, "scorediff", base_cfg))
        right = np.array([p in g for p, g in zip(bp, es.gold)])
        m3_right = np.array([p in g for p, g in zip(m3, es.gold)])
        same = bp == wp
        alignment[name] = {
            "method3_changes_pick": float(np.mean(m3 != bp)),
            "changed_picks_fixed": int((~right & m3_right).sum()),  # baseline wrong, method 3 right
            "changed_picks_broken": int((right & ~m3_right).sum()),  # baseline right, method 3 wrong
            "baseline_wrong": int((~right).sum()),
            "same_pick_when_baseline_wrong": float(same[~right].mean()),
            "same_pick_when_baseline_right": float(same[right].mean()),
            "same_pick_if_independent": float(np.mean(1.0 / es.n_candidates)),
            "mean_spearman_baseline_vs_swap": float(np.mean(
                [spearman(b, w) for b, w in zip(base_sc, swap_sc) if len(b) >= 3])),
        }

    lst = layer_study["validation"]["test_natural"]["systems"][f"selected: {base_cfg}"]["acc1"]["mean"] \
        if reproduces else None
    results = {
        "protocol": __doc__,
        "conditions": {k: {"match": m, "draws": list(d)} for k, (m, d) in CONDITIONS.items()},
        "primary_condition": PRIMARY,
        "samples": {k: {"sample": v, "n_instances": len(sets[k]), "n_sentences": len(set(sets[k].clusters)),
                        "mean_candidates": float(np.mean(sets[k].n_candidates))} for k, v in ls.SAMPLES.items()},
        "reproduces_layer_study": {
            "baseline_winner": base_cfg, "layer_study_winner": layer_study["dev_selection"]["winner"],
            "same_winner": reproduces,
            "test_natural_acc1": validation["test_natural"]["systems"]["baseline"]["acc1"]["mean"],
            "layer_study_test_natural_acc1": lst,
        },
        "selection": selection,
        "betas": list(BETAS), "beta_curves": beta_curves, "best_beta": best_beta,
        "validation": validation,
        "profiles": profiles,
        "geometry": geometry,
        "normdiff_vs_scorediff_disagreement": disagreement,
        "exploratory_error_alignment": alignment,
        "runtime_seconds": time.time() - t0,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=1, default=float))
    print(f"saved {OUT_DIR / 'results.json'} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
