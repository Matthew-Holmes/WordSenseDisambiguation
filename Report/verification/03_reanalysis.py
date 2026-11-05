"""Re-analyse the activations that the Analysis/ notebooks used.

1. Reproduce the headline numbers from the notebooks (validates this code).
2. Re-check the out-of-order result in analysisUniform_chunk000.ipynb cell 64
   (0.855), which was executed before the function it calls was redefined.
3. Add what the notebooks lacked:
   * bootstrap confidence intervals, accuracy@1 vs chance;
   * baselines that ignore the sentence (list order, definition norm,
     typicality) and a shuffled-sentence control;
   * a permutation null for the *maximum* over the 16x16 layer-pair grid
     (the notebooks picked the best cells of that grid);
   * the score split by whether the sentence vector came from the right token.
   * the effect of argsort tie-breaking (lower list index wins ties) on the
     rank-averaging "mixture" scores.

Results: results/03_reanalysis.json
"""

from __future__ import annotations

import os
import pickle
import zipfile

import numpy as np
import pandas as pd

from common import ANALYSED_CHUNKS, add_project_to_path, load_tokenizer, save_json, true_focus_token_index

add_project_to_path()
from Processing import script as pipeline  # noqa: E402

TOK_IDX = 4  # centre of the stored 9-token window = the (intended) focus token
RNG = np.random.default_rng(0)
N_BOOT = 2000
N_PERM = 1000


# ---------------------------------------------------------------- loading
def load_store(zip_path) -> dict[str, np.ndarray]:
    """Read every pickled (16, 9, 2048) array straight from the zip; keep the focus position."""
    out = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.endswith(".pkl"):
                continue
            with z.open(name) as f:
                arr = pickle.load(f)
            out[os.path.basename(name)[:-4]] = np.asarray(arr[:, TOK_IDX, :], dtype=np.float32)
    return out


def candidates_of(definitions: str) -> list[str]:
    return definitions.split(" | ") if " | " in definitions else definitions.split("|")


def build_instances(df: pd.DataFrame, store: dict, tok) -> list[dict]:
    """Parse the processed CSV sequentially: a 'sentence' row followed by its definition rows."""
    instances = []
    cur = None
    for _, r in df.iterrows():
        if r.def_or_sentence == "sentence":
            cur = {
                "word": r.word,
                "sentence": r.sentence,
                "word_loc": int(r.word_loc),
                "gold_def": r.definition,
                "cands": candidates_of(r.definitions),
                "S": store[r.model_output_key],
                "defs": {},
            }
            instances.append(cur)
        else:
            cur["defs"][int(r.def_index)] = store[r.model_output_key]
    out = []
    for inst in instances:
        n = len(inst["cands"])
        inst["complete"] = sorted(inst["defs"]) == list(range(n))
        if inst["gold_def"] in inst["cands"]:
            inst["gold"] = inst["cands"].index(inst["gold_def"])
        else:
            inst["gold"] = None
        if inst["complete"]:
            inst["D"] = np.stack([inst["defs"][i] for i in range(n)])  # (n, 16, 2048)
        # focus-token alignment, recomputed with the pipeline's own code
        row = pd.Series(
            {"word": inst["word"], "sentence": inst["sentence"], "word_loc": inst["word_loc"],
             "definitions": "|".join(inst["cands"])}
        )
        prepped = pipeline.prep_row(row, tok)
        gt, _ = true_focus_token_index(inst["sentence"].split("|"), inst["word_loc"], prepped["sentence_toks"], tok)
        inst["aligned"] = gt is not None and gt == int(prepped["word_tok_index"])
        out.append(inst)
    return out


# ---------------------------------------------------------------- scoring
def norm_rank(scores: np.ndarray, gold: int) -> np.ndarray:
    """scores (n, L) -> normalised rank of gold per column, 1 = best (notebook convention)."""
    n = scores.shape[0]
    ranks = np.argsort(np.argsort(-scores, axis=0), axis=0)
    return 1 - ranks[gold] / (n - 1)


def raw_dot(S, D):
    return np.einsum("nle,le->nl", D, S)


def cosine(S, D):
    return raw_dot(S, D) / (np.linalg.norm(D, axis=-1) * np.linalg.norm(S, axis=-1)[None] + 1e-12)


def centred_abs(S, D):
    X = np.concatenate([S[None], D])
    X = X - X.mean(0, keepdims=True)
    return np.abs(np.einsum("nle,le->nl", X[1:], X[0]))


def from_start(S, D):
    X = np.concatenate([S[None], D])
    X = X[:, 1:, :] - X[:, :1, :]
    return np.einsum("nle,le->nl", X[1:], X[0])


def from_start_centred_abs(S, D):
    X = np.concatenate([S[None], D])
    X = X[:, 1:, :] - X[:, :1, :]
    X = X - X.mean(0, keepdims=True)
    return np.abs(np.einsum("nle,le->nl", X[1:], X[0]))


SCORERS = {
    "raw_dot": raw_dot,
    "cosine": cosine,
    "centred_abs": centred_abs,
    "from_start": from_start,
    "from_start_centred_abs": from_start_centred_abs,
}


def boot_ci(x: np.ndarray) -> tuple[float, float]:
    idx = RNG.integers(0, len(x), size=(N_BOOT, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def per_layer_summary(insts, scorer) -> dict:
    scores = np.stack([norm_rank(scorer(i["S"], i["D"]), i["gold"]) for i in insts])  # (N, L)
    acc1 = np.stack([(scorer(i["S"], i["D"]).argmax(0) == i["gold"]) for i in insts]).astype(float)
    chance_acc = float(np.mean([1 / len(i["cands"]) for i in insts]))
    best = int(scores.mean(0).argmax())
    return {
        "mean_norm_rank_by_layer": scores.mean(0).round(4).tolist(),
        "median_norm_rank_by_layer": np.median(scores, 0).round(4).tolist(),
        "acc@1_by_layer": acc1.mean(0).round(4).tolist(),
        "chance_acc@1": round(chance_acc, 4),
        "best_layer_by_mean": best,
        "best_layer_mean": round(float(scores.mean(0)[best]), 4),
        "best_layer_95ci": boot_ci(scores[:, best]),
    }


# ---------------------------------------------------------------- baselines
def baselines(insts, layer: int) -> dict:
    out = {}
    lo = [1 - i["gold"] / (len(i["cands"]) - 1) for i in insts]
    out["list_order (earlier = better)"] = float(np.mean(lo))
    out["definition_norm (ignores sentence)"] = float(
        np.mean([norm_rank(np.linalg.norm(i["D"][:, layer], axis=-1)[:, None], i["gold"])[0] for i in insts])
    )
    out["typicality d.mean(d) (ignores sentence)"] = float(
        np.mean(
            [norm_rank((i["D"][:, layer] @ i["D"][:, layer].mean(0))[:, None], i["gold"])[0] for i in insts]
        )
    )
    # shuffled-sentence control: score definitions against another instance's sentence vector
    vals = []
    for _ in range(20):
        perm = RNG.permutation(len(insts))
        vals.append(
            np.mean(
                [
                    norm_rank((i["D"][:, layer] @ insts[j]["S"][layer])[:, None], i["gold"])[0]
                    for i, j in zip(insts, perm)
                ]
            )
        )
    out["raw_dot_with_shuffled_sentence"] = float(np.mean(vals))
    out["raw_dot_with_true_sentence"] = float(
        np.mean([norm_rank((i["D"][:, layer] @ i["S"][layer])[:, None], i["gold"])[0] for i in insts])
    )
    return out


# ---------------------------------------------------------------- mixtures (notebook replicas)
def nb_mixture(l1_list, l2_list, layer1, layer2, gold, how="median", how_mix="avg"):
    """Verbatim logic of compute_mixture_ranking_scores (2-expert version)."""
    scores = []
    for m1, m2, g in zip(l1_list, l2_list, gold):
        n = m1.shape[0]
        if n <= 1:  # notebook used `== 1`; n == 0 would have crashed it
            continue
        r1 = np.argsort(np.argsort(-m1[:, layer1]))
        r2 = np.argsort(np.argsort(-m2[:, layer2]))
        with np.errstate(divide="ignore"):
            avg = {"avg": (r1 + r2) / 2.0, "min": np.minimum(r1, r2), "max": np.maximum(r1, r2),
                   "har": 1.0 / (((1.0 / r1) + (1.0 / r2)) / 2.0)}[how_mix]
        fr = np.argsort(np.argsort(avg))
        scores.append(1.0 - fr[g] / (n - 1))
    s = np.array(scores)
    return float(np.median(s) if how == "median" else np.mean(s)), float(np.percentile(s, 25))


def nb_mixture_multi(l1_list, l2_list, layers1, layers2, gold, how="median"):
    """Verbatim logic of the later multi-layer compute_mixture_ranking_scores."""
    scores = []
    for m1, m2, g in zip(l1_list, l2_list, gold):
        n = m1.shape[0]
        if n <= 1:
            continue
        combined = None
        for a, b in zip(layers1, layers2):
            r = np.argsort(np.argsort(-m1[:, a])) + np.argsort(np.argsort(-m2[:, b]))
            combined = r if combined is None else combined + r
        fr = np.argsort(np.argsort(combined))
        scores.append(1.0 - fr[g] / (n - 1))
    s = np.array(scores)
    return float(np.median(s) if how == "median" else np.mean(s)), float(np.percentile(s, 25))


def user_grouping(df: pd.DataFrame, store: dict):
    """Replicates the notebooks: groupby('sentence') -> groupby('word'); row 0 = sentence."""
    sep = " | " if df.definitions.str.contains(" | ", regex=False).any() else "|"
    lists = {k: [] for k in SCORERS}
    gold, merged_groups = [], 0
    for _, grp in df.groupby("sentence"):
        for _, sub in grp.groupby("word"):
            merged_groups += int((sub.def_or_sentence == "sentence").sum() > 1)
            S = store[sub.iloc[0].model_output_key]
            D = np.stack([store[k] for k in sub.model_output_key.iloc[1:]]) if len(sub) > 1 else np.zeros((0, 16, 2048))
            for name, f in SCORERS.items():
                X = np.concatenate([S[None], D])  # scorer expects (S, D)
                lists[name].append(f(X[0], X[1:]) if len(D) else np.zeros((0, 16)))
            poss = sub.iloc[0].definitions.split(sep)
            gold.append(poss.index(sub.iloc[0].definition))
    return lists, gold, merged_groups


def mixture_grid_null(l1_list, l2_list, gold, n_layers1=16, n_layers2=16, n_perm=N_PERM, tie_break="index"):
    """Mean 'avg' mixture score for every layer pair, plus the null distribution of the grid max.

    For each sample and every possible gold index g we precompute the mixture score, so a
    permutation (random gold per sample) is a cheap gather. tie_break='index' reproduces
    numpy's argsort behaviour on the small arrays used here (lower index wins ties);
    tie_break='random' averages over random tie orders.
    """
    per_sample_all_g = []
    observed = np.zeros((n_layers1, n_layers2))
    keep = 0
    for m1, m2, g in zip(l1_list, l2_list, gold):
        n = m1.shape[0]
        if n <= 1:
            continue
        keep += 1
        r1 = np.argsort(np.argsort(-m1[:, :n_layers1], axis=0), axis=0)  # (n, L1)
        r2 = np.argsort(np.argsort(-m2[:, :n_layers2], axis=0), axis=0)  # (n, L2)
        avg = r1[:, :, None] + r2[:, None, :]  # (n, L1, L2)  (x2 of the notebook's /2: same order)
        less = (avg[None, :] < avg[:, None]).sum(1)  # [g] -> number strictly better than g
        ties = (avg[None, :] == avg[:, None])  # [g, j]
        if tie_break == "index":
            lower_idx = np.tril(np.ones((n, n), dtype=bool), k=-1)  # j < g
            tie_wins = (ties & lower_idx[:, :, None, None]).sum(1)
            rank = less + tie_wins
        else:
            rank = less + (ties.sum(1) - 1) / 2.0  # expected rank under random tie order
        sc = 1.0 - rank / (n - 1)  # (n_g, L1, L2)
        per_sample_all_g.append(sc)
        observed += sc[g]
    observed /= keep
    null_max = np.empty(n_perm)
    for p in range(n_perm):
        acc = np.zeros((n_layers1, n_layers2))
        for sc in per_sample_all_g:
            acc += sc[RNG.integers(0, sc.shape[0])]
        null_max[p] = (acc / keep).max()
    return {
        "observed_grid_max": float(observed.max()),
        "observed_grid_argmax": [int(x) for x in np.unravel_index(observed.argmax(), observed.shape)],
        "observed_cell_(2,12)": float(observed[2, 12]) if observed.shape[1] > 12 else None,
        "null_grid_max_mean": float(null_max.mean()),
        "null_grid_max_95th_pct": float(np.percentile(null_max, 95)),
        "p_value_grid_max": float((null_max >= observed.max()).mean()),
        "n_samples": keep,
    }


# ---------------------------------------------------------------- main
def main() -> None:
    tok = load_tokenizer()
    results = {}
    pooled_uniform = []
    for name, paths in ANALYSED_CHUNKS.items():
        print(f"\n=== {name}")
        df = pd.read_csv(paths["processed"])
        store = load_store(paths["store"])
        missing = sum(k not in store for k in df.model_output_key)
        insts = build_instances(df, store, tok)
        complete = [i for i in insts if i["complete"] and i["gold"] is not None and len(i["cands"]) > 1]
        rep = {
            "processed_rows": len(df),
            "pickles_in_store": len(store),
            "csv_keys_missing_from_store": missing,
            "instances": len(insts),
            "instances_with_all_candidate_definitions": sum(i["complete"] for i in insts),
            "instances_scored": len(complete),
            "instances_with_misaligned_focus_token": sum(not i["aligned"] for i in complete),
        }
        print({k: v for k, v in rep.items()})

        # --- reproduce notebook numbers (their grouping)
        lists, gold, merged = user_grouping(df, store)
        rep["notebook_grouping_merged_(sentence,word)_groups"] = merged
        repro = {
            "mixture_raw2_centred12_median": nb_mixture(lists["raw_dot"], lists["centred_abs"], 2, 12, gold, "median"),
            "mixture_raw2_centred12_mean": nb_mixture(lists["raw_dot"], lists["centred_abs"], 2, 12, gold, "mean"),
        }
        if name == "uniform_chunk0":
            repro["cell64_(7,7),(8,8)_from_start_mean_multi_version"] = nb_mixture_multi(
                lists["from_start"], lists["from_start_centred_abs"], (7, 7), (8, 8), gold, "mean")
            for how_mix in ("avg", "min", "max", "har"):
                repro[f"cell64_2expert_from_start_(7,8)_mean_{how_mix}"] = nb_mixture(
                    lists["from_start"], lists["from_start_centred_abs"], 7, 8, gold, "mean", how_mix)
        rep["reproduction_of_notebook_numbers"] = repro
        print("reproduced:", repro)

        # --- corrected per-instance analysis
        rep["per_layer"] = {k: per_layer_summary(complete, f) for k, f in SCORERS.items()}
        for k in ("raw_dot", "cosine"):
            s = rep["per_layer"][k]
            print(f"{k}: best layer {s['best_layer_by_mean']} mean {s['best_layer_mean']} CI {s['best_layer_95ci']}")
        rep["baselines_at_layer2"] = baselines(complete, 2)  # the notebooks' favourite uncentred layer
        rep["baselines_at_best_raw_layer"] = baselines(complete, rep["per_layer"]["raw_dot"]["best_layer_by_mean"])
        print("baselines:", rep["baselines_at_best_raw_layer"])

        aligned = [i for i in complete if i["aligned"]]
        mis = [i for i in complete if not i["aligned"]]
        rep["aligned_vs_misaligned_raw_dot_mean_by_layer"] = {
            "aligned_n": len(aligned),
            "misaligned_n": len(mis),
            "aligned": per_layer_summary(aligned, raw_dot)["mean_norm_rank_by_layer"] if len(aligned) > 2 else None,
            "misaligned": per_layer_summary(mis, raw_dot)["mean_norm_rank_by_layer"] if len(mis) > 2 else None,
        }

        # --- selection-adjusted null for the heatmap the notebooks searched
        raw_l = [raw_dot(i["S"], i["D"]) for i in complete]
        cen_l = [centred_abs(i["S"], i["D"]) for i in complete]
        g = [i["gold"] for i in complete]
        rep["mixture_grid_index_tiebreak"] = mixture_grid_null(raw_l, cen_l, g, tie_break="index")
        rep["mixture_grid_random_tiebreak"] = mixture_grid_null(raw_l, cen_l, g, tie_break="random", n_perm=200)
        print("grid (index ties):", rep["mixture_grid_index_tiebreak"])
        print("grid (random ties):", {k: rep["mixture_grid_random_tiebreak"][k]
                                      for k in ("observed_grid_max", "observed_cell_(2,12)", "null_grid_max_95th_pct")})
        results[name] = rep
        if name.startswith("uniform"):
            pooled_uniform.extend(complete)
        del store

    # pooled 'held-out' test of the pre-selected (2,12) configuration on all balanced chunks
    raw_l = [raw_dot(i["S"], i["D"]) for i in pooled_uniform]
    cen_l = [centred_abs(i["S"], i["D"]) for i in pooled_uniform]
    g = [i["gold"] for i in pooled_uniform]
    s = []
    for m1, m2, gg in zip(raw_l, cen_l, g):
        n = m1.shape[0]
        r = np.argsort(np.argsort(-m1[:, 2])) + np.argsort(np.argsort(-m2[:, 12]))
        # expected rank under random tie-breaking
        rank = (r < r[gg]).sum() + ((r == r[gg]).sum() - 1) / 2
        s.append(1 - rank / (n - 1))
    s = np.array(s)
    results["pooled_balanced_chunks_(2,12)_random_tiebreak"] = {
        "n": len(s),
        "mean": float(s.mean()),
        "95ci": boot_ci(s),
    }
    print("\npooled (2,12):", results["pooled_balanced_chunks_(2,12)_random_tiebreak"])
    print("saved", save_json("03_reanalysis.json", results))


if __name__ == "__main__":
    main()
