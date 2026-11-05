"""Which layers of Llama-3.2-1B best match a word in context to its WordNet definition?

Protocol (every choice is made on dev documents; test documents are only scored at the end):
  1. Score: centred cosine between the target word's vector and each candidate's gloss vector; the
     highest-scoring candidate is the prediction. Centring means are fitted on dev.
  2. Search on dev (dev_natural): 3 sentence representations (plain_last, echo_last, echo_mean)
     x 2 gloss representations (mean, last) x 62 contiguous bands of 1-4 layers = 372 configurations,
     ranked by accuracy@1. The winner, and the best single layer for the winning representation,
     are frozen.
  3. Validate on test (test_natural, test_balanced): the frozen configurations, unchanged, with
     sentence-clustered bootstrap CIs and paired comparisons against baselines.
  4. Follow-up: add a sense-frequency prior (log P(sense | lemma) from dev-document counts, leave-own-
     document-out on dev) to the frozen score; its weight lambda is tuned on dev_natural only.

Layers: index 0 = token embeddings, index k = output of block k (HuggingFace convention). The March
2025 notebooks numbered blocks 0-15, so their "layer 2" is index 3 here.

Run after `python -m Processing.script extract --samples ...` (see Processing/script.py):
    .venv/bin/python Analysis/layer_study.py
Outputs: Report/layer_study/results.json and Report/figures/layer_*.png
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from Processing import evaluation as ev  # noqa: E402
from Processing.extract import GLOSS_VARIANTS, SENTENCE_VARIANTS  # noqa: E402
from Processing.prompts import gloss_key  # noqa: E402
from Processing.script import load_sample, open_stores  # noqa: E402
from Processing.semcor_data import load_dataset  # noqa: E402

SAMPLES = {"dev_natural": "dev_natural_80", "dev_balanced": "dev_balanced_200",
           "test_natural": "test_natural_80", "test_balanced": "test_balanced_200"}
OUT_DIR = REPO / "Report" / "layer_study"
N_LAYERS = 17
N_BOOT = 2000


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def eval_set(ds, ids: list[str]) -> ev.EvalSet:
    insts = [ds.instances[i] for i in ids]
    return ev.EvalSet(
        instance_ids=list(ids),
        clusters=[x["sentence_id"] for x in insts],
        candidates=[[gloss_key(x["lemma"], s) for s in x["candidates"]] for x in insts],
        gold=[list(x["gold_indices"]) for x in insts],
        pos=[x["pos"] for x in insts],
    )


def dev_mfs_baseline(ds, es: ev.EvalSet) -> dict[str, np.ndarray]:
    """Most frequent sense among *dev-document* gold labels per (lemma, pos); WordNet order breaks ties
    and covers lemmas unseen in dev. Fair for test documents, unlike WordNet's first sense."""
    counts: dict[tuple[str, str], Counter] = {}
    for iid in ds.instances_in_split("dev"):
        x = ds.instances[iid]
        c = counts.setdefault((x["lemma"], x["pos"]), Counter())
        for g in x["gold_synsets"]:
            c[g] += 1
    scores = []
    for iid, cands in zip(es.instance_ids, es.candidates):
        x = ds.instances[iid]
        c = counts.get((x["lemma"], x["pos"]), Counter())
        n = len(cands)
        # frequency first, WordNet order second (strictly decreasing secondary key -> no ties)
        scores.append(np.array([c[s] * n - k for k, s in enumerate(x["candidates"])], dtype=np.float64))
    return ev.instance_metrics(scores, es, seed=0)


def sense_log_priors(ds, es: ev.EvalSet, leave_own_doc_out: bool) -> list[np.ndarray]:
    """log P(sense | lemma, pos) per candidate from dev-document gold counts, add-one smoothed.

    For dev instances pass leave_own_doc_out=True: the instance's own document is removed from the
    counts, so tuning on dev never sees its own label. Lemmas unseen in dev get a uniform prior, which
    leaves the decision to the model's similarity scores.
    """
    totals: dict[tuple[str, str], Counter] = {}
    by_doc: dict[tuple[str, str, str], Counter] = {}
    for iid in ds.instances_in_split("dev"):
        x = ds.instances[iid]
        for g in x["gold_synsets"]:
            totals.setdefault((x["lemma"], x["pos"]), Counter())[g] += 1
            by_doc.setdefault((x["lemma"], x["pos"], x["doc_id"]), Counter())[g] += 1
    out = []
    for iid in es.instance_ids:
        x = ds.instances[iid]
        c = Counter(totals.get((x["lemma"], x["pos"]), Counter()))
        if leave_own_doc_out:
            c.subtract(by_doc.get((x["lemma"], x["pos"], x["doc_id"]), Counter()))
        counts = np.array([max(c[s], 0) for s in x["candidates"]], dtype=np.float64)
        out.append(np.log((counts + 1.0) / (counts.sum() + len(counts))))
    return out


LAMBDAS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 3.0)


def summary(metrics: dict[str, np.ndarray], es: ev.EvalSet) -> dict:
    return {k: ev.bootstrap_ci(v, es.clusters, n_boot=N_BOOT) for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    ds = load_dataset()
    sent_store, gloss_store = open_stores()
    sets = {name: eval_set(ds, load_sample(sample)["instance_ids"]) for name, sample in SAMPLES.items()}
    for name, es in sets.items():
        print(f"{name}: {len(es)} instances, {len(set(es.clusters))} sentences, "
              f"mean candidates {np.mean(es.n_candidates):.1f}")

    keys = sorted({k for es in sets.values() for c in es.candidates for k in c})
    banks = {g: ev.GlossBank(keys, gloss_store.get(keys, g)) for g in GLOSS_VARIANTS}
    vecs = {name: {s: sent_store.get(es.instance_ids, s) for s in SENTENCE_VARIANTS} for name, es in sets.items()}
    print(f"loaded vectors ({time.time() - t0:.0f}s)")

    # centring fitted on all dev instances (natural + balanced, de-duplicated)
    dev_ids = list(dict.fromkeys(sets["dev_natural"].instance_ids + sets["dev_balanced"].instance_ids))
    dev_all = eval_set(ds, dev_ids)
    centerings = {}
    for s in SENTENCE_VARIANTS:
        dev_vecs = sent_store.get(dev_ids, s)
        for g in GLOSS_VARIANTS:
            centerings[(s, g)] = ev.fit_centering(dev_vecs, banks[g], dev_all)

    bands = ev.contiguous_bands(N_LAYERS)
    singles = {f"L{i}": [i] for i in range(N_LAYERS)}
    per_layer = {}  # (set, s, g) -> list of (n_i, L)
    for name, es in sets.items():
        for s in SENTENCE_VARIANTS:
            for g in GLOSS_VARIANTS:
                per_layer[(name, s, g)] = ev.layer_scores(vecs[name][s], banks[g], es, centerings[(s, g)])
    print(f"scored ({time.time() - t0:.0f}s)")

    def evaluate(name, s, g, configs):
        return ev.evaluate_configs(per_layer[(name, s, g)], sets[name], configs)

    # --------------------------------------------------------------- dev search
    dev_grid = {}
    for s in SENTENCE_VARIANTS:
        for g in GLOSS_VARIANTS:
            for band, metrics in evaluate("dev_natural", s, g, bands).items():
                dev_grid[f"{s}+{g}/{band}"] = metrics
    best = ev.select_best(dev_grid, "acc1")
    best_rep, best_band = best.split("/")
    best_s, best_g = best_rep.split("+")
    single_grid = {f"{best_rep}/L{i}": dev_grid[f"{best_rep}/L{i}"] for i in range(N_LAYERS)}
    best_single = ev.select_best(single_grid, "acc1")
    best_layers = bands[best_band]
    best_single_layer = int(best_single.split("/L")[1])
    ranked = sorted(dev_grid, key=lambda k: -np.mean(dev_grid[k]["acc1"]))
    print(f"dev winner: {best} (acc1 {np.mean(dev_grid[best]['acc1']):.3f}); best single layer: {best_single}")

    # per-layer profiles for every representation on every set (descriptive)
    profiles = {}
    for name, es in sets.items():
        for s in SENTENCE_VARIANTS:
            for g in GLOSS_VARIANTS:
                res = evaluate(name, s, g, singles)
                profiles[f"{name}|{s}+{g}"] = {
                    m: [float(np.mean(res[f"L{i}"][m])) for i in range(N_LAYERS)] for m in ev.METRICS
                }
    # dev-balanced view: does a sense-balanced dev set pick the same layers?
    bal_grid = {}
    for band, metrics in evaluate("dev_balanced", best_s, best_g, bands).items():
        bal_grid[f"{best_rep}/{band}"] = metrics
    best_balanced = ev.select_best(bal_grid, "acc1")

    # --------------------------------------------------------------- frequency prior (tuned on dev only)
    # combined score = layer-band cosine + lambda * log P(sense | lemma) from dev-document counts
    def combined(name, lam):
        priors = sense_log_priors(ds, sets[name], leave_own_doc_out=name.startswith("dev"))
        cos = ev.config_scores(per_layer[(name, best_s, best_g)], best_layers)
        return [c + lam * p for c, p in zip(cos, priors)]

    lam_dev = {lam: float(np.mean(ev.instance_metrics(combined("dev_natural", lam), sets["dev_natural"])["acc1"]))
               for lam in LAMBDAS}
    best_lam = max(LAMBDAS, key=lambda lam: (lam_dev[lam], -lam))
    print(f"frequency prior: best lambda on dev {best_lam} (dev acc1 {lam_dev[best_lam]:.3f})")

    # --------------------------------------------------------------- test validation
    validation = {}
    for name in ("test_natural", "test_balanced", "dev_natural", "dev_balanced"):
        es = sets[name]
        systems = {
            f"selected: {best}": ev.instance_metrics(ev.config_scores(per_layer[(name, best_s, best_g)], best_layers), es),
            "selected + dev sense-frequency prior": ev.instance_metrics(combined(name, best_lam), es),
            "dev sense-frequency prior alone": ev.instance_metrics(
                [np.asarray(p) for p in sense_log_priors(ds, es, leave_own_doc_out=name.startswith("dev"))], es),
            f"best single layer: {best_single}": ev.instance_metrics(
                ev.config_scores(per_layer[(name, best_s, best_g)], [best_single_layer]), es),
            "embeddings only (layer 0)": ev.instance_metrics(ev.config_scores(per_layer[(name, best_s, best_g)], [0]), es),
            "shuffled-sentence control": ev.shuffled_sentence_metrics(
                vecs[name][best_s], banks[best_g], es, centerings[(best_s, best_g)], best_layers),
            "MFS from dev documents": dev_mfs_baseline(ds, es),
            "WordNet first sense (optimistic)": ev.first_sense_baseline(es),
            "random": ev.random_baseline_exact(es),
        }
        sel = systems[f"selected: {best}"]
        validation[name] = {
            "n_instances": len(es),
            "n_sentences": len(set(es.clusters)),
            "systems": {k: summary(v, es) for k, v in systems.items()},
            "paired_vs_selected_acc1": {
                k: ev.paired_bootstrap(sel["acc1"], v["acc1"], es.clusters, n_boot=N_BOOT)
                for k, v in systems.items() if not k.startswith("selected")
            },
            "paired_combined_acc1": {
                k: ev.paired_bootstrap(systems["selected + dev sense-frequency prior"]["acc1"], systems[k]["acc1"],
                                       es.clusters, n_boot=N_BOOT)
                for k in ("MFS from dev documents", "dev sense-frequency prior alone", f"selected: {best}")
            },
            "selected_acc1_by_pos": ev.by_group(sel["acc1"], es.pos),
            "first_sense_acc1_by_pos": ev.by_group(systems["WordNet first sense (optimistic)"]["acc1"], es.pos),
        }
        if name.startswith("test"):
            print(f"{name}: selected acc1 {validation[name]['systems'][f'selected: {best}']['acc1']}")

    # how well does the dev layer profile predict the test profile?
    def spearman(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])

    dev_prof = profiles[f"dev_natural|{best_rep}"]["acc1"]
    test_prof = profiles[f"test_natural|{best_rep}"]["acc1"]
    test_grid_best = max(
        (k for k in bands), key=lambda b: np.mean(evaluate("test_natural", best_s, best_g, {b: bands[b]})[b]["acc1"])
    )

    results = {
        "protocol": __doc__,
        "samples": {k: {"sample": v, "n_instances": len(sets[k]), "n_sentences": len(set(sets[k].clusters)),
                        "mean_candidates": float(np.mean(sets[k].n_candidates))} for k, v in SAMPLES.items()},
        "n_configs_searched": len(dev_grid),
        "frequency_prior": {"lambda_grid_dev_acc1": {str(k): v for k, v in lam_dev.items()}, "best_lambda": best_lam},
        "dev_selection": {
            "winner": best, "winner_layers": best_layers,
            "winner_dev_acc1": summary(dev_grid[best], sets["dev_natural"])["acc1"],
            "best_single_layer": best_single,
            "best_single_dev_acc1": summary(dev_grid[best_single], sets["dev_natural"])["acc1"],
            "top10": [{"config": k, "dev_acc1": float(np.mean(dev_grid[k]["acc1"]))} for k in ranked[:10]],
            "best_per_representation": {
                rep: max((k for k in dev_grid if k.startswith(rep + "/")), key=lambda k: np.mean(dev_grid[k]["acc1"]))
                for rep in {k.split("/")[0] for k in dev_grid}
            },
            "dev_balanced_winner_same_representation": best_balanced,
        },
        "profiles": profiles,
        "profile_agreement": {
            "spearman_dev_vs_test_natural_acc1": spearman(dev_prof, test_prof),
            "test_natural_argmax_single_layer": int(np.argmax(test_prof)),
            "test_natural_best_band_hindsight": test_grid_best,
        },
        "validation": validation,
        "runtime_seconds": time.time() - t0,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=1, default=float))
    print(f"saved {OUT_DIR / 'results.json'} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
