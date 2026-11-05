"""Tests for Processing/evaluation.py, on toy data only.

Run from the repository root (WSL):  .venv/bin/python -m tests.run_tests evaluation
"""

from __future__ import annotations

import math
import time

import numpy as np

from Processing import evaluation as ev


def _es(cands, gold, clusters=None, pos=None):
    n = len(cands)
    return ev.EvalSet(
        instance_ids=[f"i{k}" for k in range(n)],
        clusters=clusters if clusters is not None else [f"s{k}" for k in range(n)],
        candidates=cands,
        gold=gold,
        pos=pos if pos is not None else ["n"] * n,
    )


def _raises(exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc:
        return
    raise AssertionError(f"{fn.__name__} did not raise {exc.__name__}")


# ---------------------------------------------------------------------------------- containers
def test_evalset_validation_and_helpers():
    good = dict(instance_ids=["a"], clusters=["s"], candidates=[["x", "y"]], gold=[[1]], pos=["n"])
    for field, bad in (("clusters", []), ("candidates", [[]]), ("gold", [[]]), ("gold", [[2]])):
        _raises(ValueError, ev.EvalSet, **{**good, field: bad})
    es = _es([["x", "y"], ["y"]], [[1], [0]])
    assert len(es) == 2 and es.n_candidates.tolist() == [2, 1]
    assert es.gold_mask(0).tolist() == [False, True]
    sub = es.subset([1])
    assert sub.candidates == [["y"]] and sub.instance_ids == ["i1"]


def test_glossbank_lookup_order_and_errors():
    vecs = np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4)
    bank = ev.GlossBank(["a", "b", "c"], vecs)
    np.testing.assert_array_equal(bank.get(["c", "a"]), vecs[[2, 0]])
    assert "b" in bank and "zzz" not in bank and len(bank) == 3
    assert bank.n_layers == 2 and bank.dim == 4
    _raises(KeyError, bank.get, ["a", "zzz"])
    _raises(ValueError, ev.GlossBank, ["a", "a"], vecs[:2])
    _raises(ValueError, ev.GlossBank, ["a"], vecs)


# ------------------------------------------------------------------------- centring and scoring
def test_layer_scores_match_hand_computed_cosine():
    rng = np.random.default_rng(1)
    L, D = 3, 5
    bank = ev.GlossBank(["a", "b", "c"], rng.normal(size=(3, L, D)).astype(np.float32))
    es = _es([["a", "b"], ["b", "c"], ["c"]], [[0], [1], [0]])
    sv = rng.normal(size=(3, L, D)).astype(np.float32)
    cen = ev.fit_centering(sv, bank, es)
    scores = ev.layer_scores(sv, bank, es, cen)
    assert [s.shape for s in scores] == [(2, L), (2, L), (1, L)]
    assert all(s.dtype == np.float32 for s in scores)
    s = sv[1] - cen.mu_sentence
    for j, name in enumerate(["b", "c"]):
        g = bank.get([name])[0] - cen.mu_gloss
        for layer in range(L):
            expected = g[layer] @ s[layer] / (np.linalg.norm(g[layer]) * np.linalg.norm(s[layer]))
            assert abs(scores[1][j, layer] - expected) < 1e-5


def test_centring_is_fit_on_one_split_over_unique_synsets_and_applied_to_another():
    L, D = 2, 3
    vecs = np.zeros((3, L, D), np.float32)
    vecs[0], vecs[1], vecs[2] = 1.0, 3.0, 100.0  # "x", "y", and "z" (only in split B)
    bank = ev.GlossBank(["x", "y", "z"], vecs)
    es_a = _es([["x", "y"], ["x"], ["x"]], [[0], [0], [0]])  # x is a candidate 3 times, y once
    sv_a = np.stack([np.full((L, D), v, np.float32) for v in (1.0, 2.0, 6.0)])
    cen = ev.fit_centering(sv_a, bank, es_a)
    np.testing.assert_allclose(cen.mu_gloss, 2.0)  # mean(x, y) over unique synsets, not (3*1 + 3) / 4
    np.testing.assert_allclose(cen.mu_sentence, 3.0)

    # Applying split A's centring to split B must use A's means. B's own sentence mean would centre
    # its single sentence vector to zero and give all-zero scores.
    es_b = _es([["y", "z"]], [[0]])
    sv_b = np.full((1, L, D), 10.0, np.float32)
    sv_b[0, :, 0] = 4.0
    got = ev.layer_scores(sv_b, bank, es_b, cen)[0]
    s = sv_b[0] - 3.0
    for j, name in enumerate(["y", "z"]):
        g = bank.get([name])[0] - 2.0
        expected = np.sum(g * s, -1) / (np.linalg.norm(g, axis=-1) * np.linalg.norm(s, axis=-1))
        np.testing.assert_allclose(got[j], expected, rtol=1e-5)
    assert np.all(np.abs(got) > 0.1)


def test_no_centring_is_plain_cosine_and_shapes_are_checked():
    bank = ev.GlossBank(["a", "b"], np.array([[[1.0, 0.0]], [[1.0, 1.0]]], np.float32))
    es = _es([["a", "b"]], [[1]])
    sv = np.array([[[0.0, 2.0]]], np.float32)
    got = ev.layer_scores(sv, bank, es, ev.no_centering(1, 2))[0][:, 0]
    np.testing.assert_allclose(got, [0.0, 1 / math.sqrt(2)], atol=1e-6)
    _raises(ValueError, ev.layer_scores, sv[:, :, :1], bank, es, ev.no_centering(1, 2))
    _raises(ValueError, ev.layer_scores, np.concatenate([sv, sv]), bank, es, ev.no_centering(1, 2))
    _raises(ValueError, ev.layer_scores, sv, bank, es, ev.no_centering(2, 2))


def test_float16_inputs_are_scored_in_float32():
    rng = np.random.default_rng(2)
    bank = ev.GlossBank(["a", "b"], rng.normal(size=(2, 4, 8)).astype(np.float16))
    es = _es([["a", "b"]], [[0]])
    sv = rng.normal(size=(1, 4, 8)).astype(np.float16)
    got = ev.layer_scores(sv, bank, es, ev.no_centering(4, 8))[0]
    assert got.dtype == np.float32
    g, s = bank.vectors.astype(np.float64), sv[0].astype(np.float64)
    expected = np.einsum("nld,ld->nl", g, s) / (np.linalg.norm(g, axis=-1) * np.linalg.norm(s, axis=-1))
    np.testing.assert_allclose(got, expected, atol=1e-5)
    cen = ev.fit_centering(sv, bank, es)
    assert cen.mu_sentence.dtype == np.float32 and cen.mu_gloss.dtype == np.float32


def test_config_scores_average_the_selected_layers():
    per_layer = [np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])]
    np.testing.assert_allclose(ev.config_scores(per_layer, [1, 2])[0], [1.5, 4.5])
    np.testing.assert_allclose(ev.config_scores(per_layer, [0])[0], [0.0, 3.0])
    _raises(ValueError, ev.config_scores, per_layer, [])


# --------------------------------------------------------------------------------------- metrics
def test_tie_breaking_is_random_unbiased_and_seeded():
    n, N = 4, 4000
    scores = [np.zeros(n)] * N
    es = _es([["a", "b", "c", "d"]] * N, [[0]] * N)
    m1 = ev.instance_metrics(scores, es, seed=7)
    m2 = ev.instance_metrics(scores, es, seed=7)
    m3 = ev.instance_metrics(scores, es, seed=8)
    np.testing.assert_array_equal(m1["acc1"], m2["acc1"])
    assert not np.array_equal(m1["acc1"], m3["acc1"])
    assert abs(m1["acc1"].mean() - 1 / n) < 0.03  # sd over 4000 draws is ~0.007
    # Unlike argsort (bug B8), a tie does not favour the first-listed candidate.
    last = _es([["a", "b", "c", "d"]] * N, [[3]] * N)
    assert abs(ev.instance_metrics(scores, last, seed=7)["acc1"].mean() - 1 / n) < 0.03
    # mrr and norm_rank resolve ties in expectation: 4-way tie, one gold -> rank 2.5.
    np.testing.assert_allclose(m1["mrr"], 1 / 2.5)
    np.testing.assert_allclose(m1["norm_rank"], 0.5)


def test_ties_and_multiple_gold():
    es = _es([["a", "b", "c"]] * 4, [[0], [0, 1], [0, 2], [2]])
    scores = [
        np.array([1.0, 1.0, 0.0]),  # gold tied for first with one other candidate
        np.array([1.0, 1.0, 0.0]),  # both tied candidates are gold
        np.array([0.1, 0.9, 0.5]),  # best gold (index 2) is second
        np.array([0.5, 0.5, 0.5]),  # three-way tie, one gold
    ]
    m = ev.instance_metrics(scores, es, seed=0)
    assert math.isclose(m["mrr"][0], 1 / 1.5) and math.isclose(m["norm_rank"][0], 0.75)
    assert m["acc1"][1] == 1.0 and m["mrr"][1] == 1.0 and m["norm_rank"][1] == 1.0
    assert m["acc1"][2] == 0.0 and math.isclose(m["mrr"][2], 0.5) and math.isclose(m["norm_rank"][2], 0.5)
    assert math.isclose(m["mrr"][3], 0.5) and math.isclose(m["norm_rank"][3], 0.5)
    # With multiple gold, acc1 counts any gold at the top: over many seeds case 0 averages ~1/2.
    acc = np.mean([ev.instance_metrics(scores[:1], es.subset([0]), seed=s)["acc1"][0] for s in range(400)])
    assert abs(acc - 0.5) < 0.1


def test_single_and_two_candidate_edge_cases():
    es = _es([["a"], ["a", "b"], ["a", "b"]], [[0], [0], [1]])
    m = ev.instance_metrics([np.array([0.3]), np.array([0.2, 0.8]), np.array([0.2, 0.8])], es)
    np.testing.assert_allclose(m["acc1"], [1, 0, 1])
    np.testing.assert_allclose(m["mrr"], [1, 0.5, 1])
    np.testing.assert_allclose(m["norm_rank"], [1, 0, 1])


def test_instance_metrics_input_checks_and_minus_inf():
    es = _es([["a", "b"]], [[0]])
    for bad in ([np.array([np.nan, 1.0])], [np.array([1.0])], []):
        _raises(ValueError, ev.instance_metrics, bad, es)
    m = ev.instance_metrics([np.array([-np.inf, 1.0])], es)  # e.g. a candidate with no evidence
    assert m["acc1"][0] == 0.0 and m["norm_rank"][0] == 0.0


# ------------------------------------------------------------------------------------- bootstrap
def test_bootstrap_ci_is_deterministic_and_contains_the_mean():
    rng = np.random.default_rng(3)
    values = rng.random(300)
    clusters = [f"s{k // 3}" for k in range(300)]
    a = ev.bootstrap_ci(values, clusters, n_boot=500, seed=1)
    b = ev.bootstrap_ci(values, clusters, n_boot=500, seed=1)
    assert a == b
    assert a["lo"] <= a["mean"] <= a["hi"] and math.isclose(a["mean"], values.mean())
    assert a != ev.bootstrap_ci(values, clusters, n_boot=500, seed=2)
    _raises(ValueError, ev.bootstrap_ci, np.array([1.0, np.nan]), ["a", "b"])


def test_bootstrap_resamples_whole_clusters():
    values = np.array([1.0] * 10 + [0.0] * 10)
    clusters = ["A"] * 10 + ["B"] * 10
    dist = ev.bootstrap_distribution(values, clusters, n_boot=2000, seed=0)
    assert set(np.round(dist, 12)) <= {0.0, 0.5, 1.0}
    ci = ev.bootstrap_ci(values, clusters, n_boot=2000, seed=0)
    assert ci["lo"] == 0.0 and ci["hi"] == 1.0  # an instance-level bootstrap would give roughly [0.3, 0.7]
    # Unequal cluster sizes: replicates are pooled (instance-weighted) means.
    dist = ev.bootstrap_distribution(np.array([1.0, 0.0, 0.0, 0.0]), ["A", "B", "B", "B"], n_boot=500, seed=0)
    assert set(np.round(dist, 12)) <= {0.0, 0.25, 1.0}


def test_paired_bootstrap_sign_and_p_value():
    rng = np.random.default_rng(4)
    b = rng.random(200)
    clusters = [f"s{k // 2}" for k in range(200)]
    up = ev.paired_bootstrap(b + 0.1, b, clusters, n_boot=1000, seed=0)
    assert math.isclose(up["diff"], 0.1) and up["lo"] > 0 and up["p_two_sided"] < 0.01
    down = ev.paired_bootstrap(b, b + 0.1, clusters, n_boot=1000, seed=0)
    assert down["diff"] < 0 and down["hi"] < 0 and down["p_two_sided"] < 0.01
    same = ev.paired_bootstrap(b, b, clusters, n_boot=1000, seed=0)
    assert same["diff"] == 0.0 and same["p_two_sided"] == 1.0
    noise = ev.paired_bootstrap(rng.random(200), rng.random(200), clusters, n_boot=1000, seed=0)
    assert noise["lo"] < 0 < noise["hi"] and noise["p_two_sided"] > 0.05
    _raises(ValueError, ev.paired_bootstrap, b, b[:-1], clusters)


def test_summarize_covers_every_metric():
    es = _es([["a", "b"]] * 6, [[0]] * 6, clusters=["s0", "s0", "s1", "s1", "s2", "s2"])
    m = ev.instance_metrics([np.array([1.0, 0.0])] * 6, es)
    out = ev.summarize(m, es.clusters, n_boot=200)
    assert set(out) == set(ev.METRICS)
    assert out["acc1"] == {"mean": 1.0, "lo": 1.0, "hi": 1.0}


# ------------------------------------------------------------------------------------- baselines
def test_random_baseline_matches_closed_form():
    cands = [["a", "b"], ["a", "b", "c", "d", "e"], ["a", "b", "c", "d"], ["a"]]
    gold = [[0], [1, 3], [2], [0]]
    es = _es(cands, gold)
    exact = ev.random_baseline_exact(es)
    np.testing.assert_allclose(exact["acc1"], [1 / 2, 2 / 5, 1 / 4, 1.0])
    np.testing.assert_allclose(exact["norm_rank"], [0.5, 0.75, 0.5, 1.0])
    np.testing.assert_allclose(exact["mrr"], [0.75, 0.4 + 0.15 + 0.2 / 3 + 0.025, 25 / 48, 1.0])
    mc = ev.random_baseline(es, n_draws=4000, seed=0)
    for metric in ev.METRICS:
        np.testing.assert_allclose(mc[metric], exact[metric], atol=0.03)
    assert abs(mc["acc1"].mean() - np.mean([len(g) / len(c) for g, c in zip(gold, cands)])) < 0.02


def test_first_sense_baseline():
    es = _es([["a", "b", "c", "d", "e"], ["a", "b"], ["a", "b", "c"]], [[0], [1], [2, 1]])
    m = ev.first_sense_baseline(es)
    np.testing.assert_allclose(m["acc1"], [1, 0, 0])
    np.testing.assert_allclose(m["mrr"], [1, 0.5, 0.5])
    np.testing.assert_allclose(m["norm_rank"], [1, 0, 0.5])


def test_shuffled_sentence_never_uses_an_instance_own_vector():
    N, D = 12, 13
    eye = np.eye(D, dtype=np.float32)
    # L = 1. Gloss g_i = e_i; one shared distractor = e_12; sentence i = e_i, so each sentence
    # matches its own gold exactly.
    bank = ev.GlossBank([f"g{i}" for i in range(N)] + ["distractor"], eye[:, None, :])
    es = _es([[f"g{i}", "distractor"] for i in range(N)], [[0]] * N)
    sv = eye[:N, None, :]
    cen = ev.no_centering(1, D)
    own = ev.instance_metrics(ev.config_scores(ev.layer_scores(sv, bank, es, cen), [0]), es)
    assert own["acc1"].min() == 1.0
    shuf = ev.shuffled_sentence_metrics(sv, bank, es, cen, layers=[0], n_perm=25, seed=3)
    # Against any other instance's vector both candidates score 0, a two-way tie, so norm_rank is
    # exactly 0.5 for every permutation. One self-pairing would push it towards 1.
    np.testing.assert_allclose(shuf["norm_rank"], 0.5)
    np.testing.assert_allclose(shuf["mrr"], 1 / 1.5)
    assert abs(shuf["acc1"].mean() - 0.5) < 0.1
    _raises(ValueError, ev.shuffled_sentence_metrics, sv[:1], bank, es.subset([0]), cen, [0])


def test_shuffled_sentence_matches_layer_scores_when_partners_are_identical():
    # Every instance has the same sentence vector, so a shuffled partner must give exactly the
    # same scores as the instance's own vector (checks the centring and layer averaging).
    rng = np.random.default_rng(5)
    L, D = 3, 6
    bank = ev.GlossBank(["a", "b", "c"], rng.normal(size=(3, L, D)).astype(np.float32))
    es = _es([["a", "b", "c"]] * 5, [[1]] * 5)
    sv = np.repeat(rng.normal(size=(1, L, D)).astype(np.float32), 5, axis=0)
    cen = ev.Centering(mu_sentence=rng.normal(size=(L, D)).astype(np.float32),
                       mu_gloss=rng.normal(size=(L, D)).astype(np.float32))
    own = ev.instance_metrics(ev.config_scores(ev.layer_scores(sv, bank, es, cen), [0, 2]), es)
    shuf = ev.shuffled_sentence_metrics(sv, bank, es, cen, layers=[0, 2], n_perm=5)
    for metric in ("mrr", "norm_rank"):
        np.testing.assert_allclose(shuf[metric], own[metric], atol=1e-6)


# ----------------------------------------------------------------------- configs and selection
def test_contiguous_bands():
    bands = ev.contiguous_bands(17)
    assert len(bands) == 17 + 16 + 15 + 14
    assert bands["L0"] == [0] and bands["L16"] == [16]
    assert bands["L4-6"] == [4, 5, 6] and bands["L13-16"] == [13, 14, 15, 16]
    assert "L14-17" not in bands
    assert list(ev.contiguous_bands(3, widths=(1, 3))) == ["L0", "L1", "L2", "L0-2"]
    _raises(ValueError, ev.contiguous_bands, 3, (4,))


def test_evaluate_configs_and_select_best_tie_rules():
    # Layer 1 ranks the gold first for every instance; layer 0 never does.
    per_layer = [np.array([[0.0, 1.0], [1.0, 0.0]])] * 3  # (n = 2 candidates, L = 2)
    es = _es([["a", "b"]] * 3, [[0]] * 3)
    res = ev.evaluate_configs(per_layer, es, {"L0": [0], "L1": [1], "L0-1": [0, 1]})
    assert res["L1"]["acc1"].mean() == 1.0 and res["L0"]["acc1"].mean() == 0.0
    assert set(res["L0-1"]) == set(ev.METRICS)
    assert ev.select_best(res) == "L1"
    assert ev.select_best(res, metric="norm_rank") == "L1"
    one, zero = {"acc1": np.array([1.0])}, {"acc1": np.array([0.0])}
    assert ev.select_best({"L4-6": one, "L5": one, "L2": zero}) == "L5"  # narrower band wins a tie
    assert ev.select_best({"L7": one, "L3": one, "L3-4": one}) == "L3"  # then the lower layer
    assert ev.select_best({"echo/L9": one, "echo/L2-3": one}) == "echo/L9"
    assert ev.select_best({"x": one, "y": one}, configs={"x": [5, 6], "y": [2]}) == "y"
    _raises(ValueError, ev.select_best, {})


def test_by_group():
    out = ev.by_group(np.array([1.0, 0.0, 1.0, 1.0]), ["n", "v", "n", "a"])
    assert out == {"a": {"mean": 1.0, "n": 1}, "n": {"mean": 1.0, "n": 2}, "v": {"mean": 0.0, "n": 1}}


def test_realistic_shapes_run_quickly():
    rng = np.random.default_rng(6)
    L, D, N, G = 17, 2048, 60, 400
    bank = ev.GlossBank([f"s{k}" for k in range(G)], rng.normal(size=(G, L, D)).astype(np.float16))
    sizes = rng.integers(2, 31, size=N)
    cands = [[f"s{k}" for k in rng.choice(G, size=n, replace=False)] for n in sizes]
    es = _es(cands, [[0]] * N, clusters=[f"c{k // 4}" for k in range(N)])
    sv = rng.normal(size=(N, L, D)).astype(np.float16)
    start = time.time()
    cen = ev.fit_centering(sv, bank, es)
    per_layer = ev.layer_scores(sv, bank, es, cen)
    res = ev.evaluate_configs(per_layer, es, ev.contiguous_bands(L))
    best = ev.select_best(res)
    ev.shuffled_sentence_metrics(sv, bank, es, cen, ev.contiguous_bands(L)[best], n_perm=5)
    assert [s.shape for s in per_layer] == [(n, L) for n in sizes]
    assert time.time() - start < 30
