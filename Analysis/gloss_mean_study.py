"""Does removing what a target's candidate glosses share sharpen gloss matching? (the gloss-mean study)

Hypothesis: all of a lemma's definition prompts begin "lemma:", so their gloss vectors share a component (the
lemma, and the shape of a definition). The cosine between a target's vector and a gloss vector is then partly a
cosine with that shared component, which says nothing about which sense is meant. Subtracting the mean of the
candidates' gloss vectors before the cosine keeps only what tells the senses apart. This is the gloss-side
counterpart of the swap study (Analysis/swap_study.py).

Scores (Processing/gloss_contrast.py), computed per layer and averaged over a band of layers:
    baseline   cos(h~, g~_k)                                the layer study's score
    glossmean  cos(h~, g~_k - m),  m = mean_j g~_j          primary: remove the candidates' mean gloss vector
    unitmean   cos(h~, g^_k - mean_j g^_j)                  secondary: normalise each gloss first, then remove
    dot        <h^, g~_k> / mean_j ||g~_j||                  diagnostic: what glossmean reduces to for 2 candidates
(h = the target's vector, g_k = candidate k's gloss vector; ~ = centred with the layer study's dev means,
^ = centred and unit length.) The score-level version, cos(h~, g~_k) - mean_j cos(h~, g~_j), is not run: it ranks
exactly like the baseline.

Protocol, fixed before any result was seen. Every choice is made on dev documents; test documents are scored once.
  1. Vectors and centring: the layer study's own. No new extraction.
  2. For each score, search dev_natural: 3 sentence representations x 2 gloss representations x 62 contiguous
     bands of 1-4 layers = 372 configurations, ranked by accuracy@1 with evaluation.select_best. The baseline's
     search must reproduce the layer study's winner.
  3. Validate on test_natural and test_balanced. Each score keeps its frozen configuration. Results carry
     cluster-bootstrap CIs and paired comparisons against the frozen baseline. Primary question: does glossmean
     beat the baseline on test?
  Secondary analyses, reported whatever they show:
  4. Each score at the baseline's frozen configuration, which removes selection noise.
  5. Partial removal, cos(h~, g~_k - alpha m), at the baseline's configuration, with alpha tuned on dev_natural
     (grid -0.5 to 1.5 in steps of 0.1; ties go to the alpha nearest 0).
  6. Diagnostics: how much of each gloss vector is the candidates' mean (cosine and length ratio by layer, and
     the target's cosine with that mean); per-layer accuracy profiles; accuracy split by the number of candidates
     (2, 3-5, 6 or more), since with two candidates glossmean is the dot product and unitmean is the baseline;
     how often each score changes the baseline's pick, and whether the changes fix or break it.
  Exploratory, added after the primary result was seen: does a mean-pooled gloss vector's length mostly reflect
  how many words the gloss has, and do the gold gloss and each score's pick tend to be wordier than their rivals?

Run from the repository root, in WSL:
    .venv/bin/python Analysis/gloss_mean_study.py
Outputs: Report/gloss_mean_study/results.json; figures via Analysis/gloss_mean_study_figures.py.
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
from Processing import gloss_contrast as gc  # noqa: E402
from Processing.contrastive import unit  # noqa: E402
from Processing.extract import GLOSS_VARIANTS, SENTENCE_VARIANTS  # noqa: E402
from Processing.script import load_sample, open_stores  # noqa: E402
from Processing.semcor_data import load_dataset  # noqa: E402

OUT_DIR = REPO / "Report" / "gloss_mean_study"
N_LAYERS, N_BOOT = ls.N_LAYERS, ls.N_BOOT
PRIMARY = "glossmean"
CONTRASTS = ("glossmean", "unitmean", "dot")
ALPHAS = tuple(float(a) + 0.0 for a in np.round(np.arange(-0.5, 1.51, 0.1), 2))  # + 0.0 turns -0.0 into 0.0


def n_bin(n: int) -> str:
    return "2" if n == 2 else "3-5" if n <= 5 else "6+"


def top_pick(scores: list[np.ndarray]) -> np.ndarray:
    return np.array([int(np.argmax(s)) for s in scores])


def main() -> None:
    t0 = time.time()
    ds = load_dataset()
    sent_store, gloss_store = open_stores()
    sets = {name: ls.eval_set(ds, load_sample(sample)["instance_ids"]) for name, sample in ls.SAMPLES.items()}
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
    print(f"loaded vectors ({time.time() - t0:.0f}s)")

    # ------------------------------------------------------------------ per-layer scores
    per_layer = {}  # (set, s, g) -> {score: [(n_i, L), ...]}
    for name, es in sets.items():
        for s in SENTENCE_VARIANTS:
            for g in GLOSS_VARIANTS:
                per_layer[(name, s, g)] = gc.gloss_contrast_layer_scores(vecs[name][s], banks[g], es, centerings[(s, g)])
    # the baseline here must be the layer study's own scorer
    check = ev.layer_scores(vecs["dev_natural"]["echo_last"], banks["mean"], sets["dev_natural"],
                            centerings[("echo_last", "mean")])
    baseline_check = max(float(np.abs(a - b).max())
                         for a, b in zip(per_layer[("dev_natural", "echo_last", "mean")]["baseline"], check))
    print(f"scored ({time.time() - t0:.0f}s); max |baseline - evaluation.layer_scores| = {baseline_check:.2e}")

    bands = ev.contiguous_bands(N_LAYERS)
    singles = {f"L{i}": [i] for i in range(N_LAYERS)}

    def scores_at(name, score, config):
        rep, band = config.split("/")
        s, g = rep.split("+")
        return ev.config_scores(per_layer[(name, s, g)][score], bands[band])

    def metrics_at(name, score, config):
        return ev.instance_metrics(scores_at(name, score, config), sets[name])

    def dev_grid(score):
        grid = {}
        for s in SENTENCE_VARIANTS:
            for g in GLOSS_VARIANTS:
                res = ev.evaluate_configs(per_layer[("dev_natural", s, g)][score], sets["dev_natural"], bands)
                grid.update({f"{s}+{g}/{band}": m for band, m in res.items()})
        return grid

    # ------------------------------------------------------------------ dev selection
    base_grid = dev_grid("baseline")
    base_cfg = ev.select_best(base_grid, "acc1")
    layer_study = json.loads((REPO / "Report" / "layer_study" / "results.json").read_text())
    reproduces = base_cfg == layer_study["dev_selection"]["winner"]
    print(f"baseline dev winner {base_cfg} (layer study: {layer_study['dev_selection']['winner']}; "
          f"reproduced: {reproduces})")
    selection = {"baseline": {"config": base_cfg, "dev_acc1": ls.summary(base_grid[base_cfg], sets["dev_natural"])["acc1"]}}
    selected = {}
    for score in CONTRASTS:
        grid = dev_grid(score)
        cfg = ev.select_best(grid, "acc1")
        selected[score] = cfg
        ranked = sorted(grid, key=lambda k: -np.mean(grid[k]["acc1"]))
        selection[score] = {
            "config": cfg, "dev_acc1": ls.summary(grid[cfg], sets["dev_natural"])["acc1"],
            "dev_acc1_at_baseline_config": float(np.mean(grid[base_cfg]["acc1"])),
            "top5": [{"config": k, "dev_acc1": float(np.mean(grid[k]["acc1"]))} for k in ranked[:5]],
        }
        print(f"  {score:9s} dev winner {cfg:22s} acc1 {np.mean(grid[cfg]['acc1']):.3f}  "
              f"(at baseline config {np.mean(grid[base_cfg]['acc1']):.3f})")

    # ------------------------------------------------------------------ partial removal (alpha tuned on dev)
    rep, band = base_cfg.split("/")
    bs, bg = rep.split("+")
    alpha_scores = {name: gc.gloss_contrast_layer_scores(vecs[name][bs], banks[bg], es, centerings[(bs, bg)], ALPHAS)
                    for name, es in sets.items()}

    def alpha_metrics(name, a):
        return ev.instance_metrics(ev.config_scores(alpha_scores[name][gc.alpha_key(a)], bands[band]), sets[name])

    alpha_curves = {name: [float(np.mean(alpha_metrics(name, a)["acc1"])) for a in ALPHAS] for name in sets}
    dev_curve = alpha_curves["dev_natural"]
    best_alpha = max(ALPHAS, key=lambda a: (dev_curve[ALPHAS.index(a)], -abs(a)))
    print(f"  alpha tuned on dev: {best_alpha} (dev acc1 {dev_curve[ALPHAS.index(best_alpha)]:.3f})")

    # ------------------------------------------------------------------ validation
    validation = {}
    for name in ("test_natural", "test_balanced", "dev_natural", "dev_balanced"):
        es = sets[name]
        systems = {"baseline": metrics_at(name, "baseline", base_cfg)}
        for score in CONTRASTS:
            systems[f"{score}|own"] = metrics_at(name, score, selected[score])
            systems[f"{score}|base_cfg"] = metrics_at(name, score, base_cfg)
        systems["alpha|base_cfg"] = alpha_metrics(name, best_alpha)
        systems["MFS from dev documents"] = ls.dev_mfs_baseline(ds, es)
        systems["WordNet first sense (optimistic)"] = ev.first_sense_baseline(es)
        systems["random"] = ev.random_baseline_exact(es)
        base = systems["baseline"]["acc1"]
        bins = [n_bin(n) for n in es.n_candidates]
        validation[name] = {
            "n_instances": len(es), "n_sentences": len(set(es.clusters)),
            "systems": {k: ls.summary(v, es) for k, v in systems.items()},
            "paired_vs_baseline_acc1": {k: ev.paired_bootstrap(v["acc1"], base, es.clusters, n_boot=N_BOOT)
                                        for k, v in systems.items() if k != "baseline"},
            "acc1_by_pos": {k: ev.by_group(systems[k]["acc1"], es.pos)
                            for k in ["baseline"] + [f"{m}|own" for m in CONTRASTS]},
            "acc1_by_n_candidates": {k: ev.by_group(systems[k]["acc1"], bins)
                                     for k in ["baseline"] + [f"{m}|{w}" for m in CONTRASTS for w in ("own", "base_cfg")]},
        }
        if name.startswith("test"):
            line = "  ".join(f"{m} {validation[name]['systems'][f'{m}|own']['acc1']['mean']:.3f}" for m in CONTRASTS)
            print(f"{name}: baseline {validation[name]['systems']['baseline']['acc1']['mean']:.3f}  {line}")

    # ------------------------------------------------------------------ profiles and diagnostics
    profiles = {}  # f"{set}|{s}+{g}|{score}" -> accuracy@1 per layer
    for (name, s, g), pl in per_layer.items():
        for score in gc.SCORES:
            res = ev.evaluate_configs(pl[score], sets[name], singles)
            profiles[f"{name}|{s}+{g}|{score}"] = [float(np.mean(res[f"L{i}"]["acc1"])) for i in range(N_LAYERS)]

    geometry = {}  # how much of each gloss vector is the candidates' shared mean?
    for name, es in sets.items():
        for g in GLOSS_VARIANTS:
            mu_g = centerings[(bs, g)].mu_gloss.astype(np.float32)
            mu_s = centerings[(bs, g)].mu_sentence.astype(np.float32)
            cos_raw, cos_c, ratio, cos_hm = [], [], [], []
            for i, cands in enumerate(es.candidates):
                raw = banks[g].get(cands).astype(np.float32)  # (n, L, D)
                cen = raw - mu_g
                m_raw, m_c = raw.mean(axis=0), cen.mean(axis=0)
                cos_raw.append((unit(raw) * unit(m_raw)).sum(-1).mean(0))  # mean over candidates: (L,)
                cos_c.append((unit(cen) * unit(m_c)).sum(-1).mean(0))
                ratio.append((np.linalg.norm(cen - m_c, axis=-1) / np.linalg.norm(cen, axis=-1)).mean(0))
                h = vecs[name][bs][i].astype(np.float32) - mu_s
                cos_hm.append((unit(h) * unit(m_c)).sum(-1))
            geometry[f"{name}|{g}"] = {
                "cos_gloss_mean_raw": np.mean(cos_raw, axis=0).tolist(),
                "cos_gloss_mean_centred": np.mean(cos_c, axis=0).tolist(),
                "residual_norm_ratio_centred": np.mean(ratio, axis=0).tolist(),
                "cos_target_mean_centred": np.mean(cos_hm, axis=0).tolist(),
            }

    pick_changes = {}  # at the baseline's configuration: how often does each score change the baseline's pick?
    for name, es in sets.items():
        bp = top_pick(scores_at(name, "baseline", base_cfg))
        right = np.array([p in g for p, g in zip(bp, es.gold)])
        pick_changes[name] = {"baseline_wrong": int((~right).sum())}
        for score in CONTRASTS:
            cp = top_pick(scores_at(name, score, base_cfg))
            c_right = np.array([p in g for p, g in zip(cp, es.gold)])
            pick_changes[name][score] = {
                "changes_pick": float(np.mean(cp != bp)),
                "fixed": int((~right & c_right).sum()),  # baseline wrong, contrast right
                "broken": int((right & ~c_right).sum()),  # baseline right, contrast wrong
            }

    # Exploratory, added after the primary result was seen: gloss length against gloss-vector length, at the
    # baseline's representation and layers. Ranks run from 0 (fewest words / shortest vector) to 1, with ties averaged.
    def avg_rank01(x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty(len(x))
        ranks[order] = np.arange(len(x))
        _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        ranks = (np.bincount(inv, weights=ranks) / counts)[inv]
        return ranks / max(len(x) - 1, 1)

    n_words = {k: len(ds.synsets[k.split("|", 1)[1]]["gloss"].split()) for k in keys}
    band_layers = bands[band]
    mu_band = centerings[(bs, bg)].mu_gloss[band_layers].astype(np.float32)

    def vec_length(cands) -> np.ndarray:  # centred gloss-vector length, averaged over the baseline's layers
        return np.linalg.norm(banks[bg].get(cands).astype(np.float32)[:, band_layers] - mu_band, axis=-1).mean(1)

    uniq = sorted({k for c in dev_all.candidates for k in c})
    wc, vl = np.array([n_words[k] for k in uniq]), vec_length(uniq)
    word_bins = {"1-4": (1, 4), "5-8": (5, 8), "9-14": (9, 14), "15+": (15, 10**6)}
    gloss_length = {
        "dev_glosses": len(uniq),
        "spearman_words_vs_vector_length": float(np.corrcoef(avg_rank01(wc), avg_rank01(vl))[0, 1]),
        "vector_length_by_words": {b: {"n": int(((wc >= lo) & (wc <= hi)).sum()),
                                       "mean_length": float(vl[(wc >= lo) & (wc <= hi)].mean())}
                                   for b, (lo, hi) in word_bins.items()},
        "sets": {},
    }
    for name, es in sets.items():
        picks = {score: top_pick(scores_at(name, score, base_cfg)) for score in gc.SCORES}
        rows = {"gold_words": [], "gold_vector_length": [], "first_sense_words": []}
        rows.update({f"pick_words|{score}": [] for score in gc.SCORES})
        for i, (cands, gold) in enumerate(zip(es.candidates, es.gold)):
            rw, rl = avg_rank01([n_words[k] for k in cands]), avg_rank01(vec_length(cands))
            rows["gold_words"].append(rw[list(gold)].mean())
            rows["gold_vector_length"].append(rl[list(gold)].mean())
            rows["first_sense_words"].append(rw[0])
            for score in gc.SCORES:
                rows[f"pick_words|{score}"].append(rw[picks[score][i]])
        gloss_length["sets"][name] = {k: float(np.mean(v)) for k, v in rows.items()}

    results = {
        "protocol": __doc__,
        "primary_score": PRIMARY,
        "samples": {k: {"sample": v, "n_instances": len(sets[k]), "n_sentences": len(set(sets[k].clusters)),
                        "mean_candidates": float(np.mean(sets[k].n_candidates)),
                        "n_candidates_bins": {b: int(sum(n_bin(n) == b for n in sets[k].n_candidates))
                                              for b in ("2", "3-5", "6+")}}
                    for k, v in ls.SAMPLES.items()},
        "baseline_matches_layer_scores": baseline_check,
        "reproduces_layer_study": {
            "baseline_winner": base_cfg, "layer_study_winner": layer_study["dev_selection"]["winner"],
            "same_winner": reproduces,
            "test_natural_acc1": validation["test_natural"]["systems"]["baseline"]["acc1"]["mean"],
        },
        "selection": selection,
        "alphas": list(ALPHAS), "alpha_curves": alpha_curves, "best_alpha": best_alpha,
        "validation": validation,
        "profiles": profiles,
        "geometry": geometry,
        "pick_changes": pick_changes,
        "exploratory_gloss_length": gloss_length,
        "runtime_seconds": time.time() - t0,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=1, default=float))
    print(f"saved {OUT_DIR / 'results.json'} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
