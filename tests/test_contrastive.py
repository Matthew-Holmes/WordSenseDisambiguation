"""Contrastive scoring methods (Processing/contrastive.py): hand-checked values and the algebraic identities."""

from __future__ import annotations

import numpy as np

from Processing import contrastive as ct
from Processing import evaluation as ev


def _toy():
    """Dimensions: [sense A, sense B, 'grammar']. The target means A; the random word shares only the grammar."""
    h = np.array([[1.0, 0.0, 3.0]])  # (L=1, D=3)
    s = np.array([[0.0, 0.5, 3.0]])
    g = np.array([[[1.0, 0.0, 0.5]], [[0.0, 1.0, 2.0]]])  # (n=2, L=1, D=3): gloss A, gloss B
    return h, s, g


def _random(n=6, K=1, L=4, D=16, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((L, D)), rng.standard_normal((K, L, D)), rng.standard_normal((n, L, D))


def test_toy_example_matches_the_hand_calculation():
    h, s, g = _toy()
    out = {m: v[:, 0] for m, v in ct.contrastive_scores(h, s, g).items()}
    # baseline: cos(h, gA) = 2.5 / sqrt(10 * 1.25) = 1/sqrt(2); cos(h, gB) = 6 / sqrt(10 * 5)
    np.testing.assert_allclose(out["baseline"], [1 / np.sqrt(2), 6 / np.sqrt(50)], rtol=1e-5)
    # method 1: d = h - s = (1, -0.5, 0); cos(d, gA) = 1 / 1.25, cos(d, gB) = -0.5 / 2.5
    np.testing.assert_allclose(out["diff"], [0.8, -0.2], rtol=1e-5)
    # method 3: cos(s, gA) = 1.5 / sqrt(9.25 * 1.25), cos(s, gB) = 6.5 / sqrt(9.25 * 5)
    cos_s = np.array([1.5 / np.sqrt(9.25 * 1.25), 6.5 / np.sqrt(9.25 * 5)])
    np.testing.assert_allclose(out["swap"], cos_s, rtol=1e-5)
    np.testing.assert_allclose(out["scorediff"], out["baseline"] - cos_s, rtol=1e-5)
    # method 2: d = h/|h| - s/|s|
    d = h[0] / np.sqrt(10) - s[0] / np.sqrt(9.25)
    expected = [d @ g[k, 0] / (np.linalg.norm(d) * np.linalg.norm(g[k, 0])) for k in range(2)]
    np.testing.assert_allclose(out["normdiff"], expected, rtol=1e-5)
    assert np.argmax(out["baseline"]) == 1  # the shared 'grammar' dimension misleads the plain cosine...
    assert all(np.argmax(out[m]) == 0 for m in ("diff", "normdiff", "scorediff"))  # ...the contrasts do not


def test_normdiff_and_scorediff_rank_identically_at_every_layer():
    for seed in range(5):
        h, s, g = _random(seed=seed, K=1 + seed % 3)
        out = ct.contrastive_scores(h, s, g)
        d = ct.unit(h) - ct.unit(s).mean(axis=0)  # (L, D)
        np.testing.assert_allclose(out["scorediff"], out["normdiff"] * np.linalg.norm(d, axis=-1), rtol=1e-4,
                                   atol=1e-6)
        assert (np.argsort(-out["scorediff"], axis=0) == np.argsort(-out["normdiff"], axis=0)).all()


def test_diff_is_the_length_weighted_score_difference():
    h, s, g = _random(K=1)
    out = ct.contrastive_scores(h, s, g)
    cos = lambda a, b: (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))  # noqa: E731
    weighted = np.linalg.norm(h, axis=-1) * cos(h[None], g) - np.linalg.norm(s[0], axis=-1) * cos(s[0][None], g)
    d_norm = np.linalg.norm(h - s[0], axis=-1)
    np.testing.assert_allclose(out["diff"], weighted / d_norm, rtol=1e-4, atol=1e-6)


def test_sentence_centring_cancels_in_diff_only():
    h, s, g = _random(K=1)
    mu = np.random.default_rng(9).standard_normal(h.shape) * 3
    a, b = ct.contrastive_scores(h, s, g), ct.contrastive_scores(h, s, g, mu_s=mu, mu_g=np.zeros_like(mu))
    np.testing.assert_allclose(a["diff"], b["diff"], rtol=1e-4, atol=1e-6)
    for m in ("baseline", "normdiff", "scorediff"):
        assert not np.allclose(a[m], b[m], atol=1e-3), m


def test_baseline_and_scorediff_agree_with_the_layer_study_scorer():
    rng = np.random.default_rng(3)
    L, D = 5, 12
    names = [f"w|s{k}" for k in range(7)]
    bank = ev.GlossBank(names, rng.standard_normal((7, L, D)).astype(np.float16))
    es = ev.EvalSet(["i0", "i1", "i2"], ["c0", "c0", "c1"], [names[:3], names[2:7], names[1:3]], [[0], [1], [0]],
                    ["n", "v", "n"])
    orig = rng.standard_normal((3, L, D)).astype(np.float16)
    swap = rng.standard_normal((3, L, D)).astype(np.float16)
    cen = ev.fit_centering(orig, bank, es)
    out = ct.contrastive_layer_scores(orig, swap, bank, es, cen)
    base, other = ev.layer_scores(orig, bank, es, cen), ev.layer_scores(swap, bank, es, cen)
    for i in range(3):
        np.testing.assert_allclose(out["baseline"][i], base[i], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(out["swap"][i], other[i], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(out["scorediff"][i], base[i] - other[i], rtol=1e-5, atol=1e-6)


def test_identical_vectors_give_zero_scores_not_nan():
    h, _, g = _random()
    out = ct.contrastive_scores(h, h[None], g)
    for m in ("diff", "normdiff", "scorediff"):
        assert np.isfinite(out[m]).all() and np.allclose(out[m], 0, atol=1e-6), m


def test_several_draws_average_the_swapped_side():
    h, s, g = _random(K=3)
    out = ct.contrastive_scores(h, s, g)
    singles = [ct.contrastive_scores(h, s[j], g) for j in range(3)]
    np.testing.assert_allclose(out["swap"], np.mean([x["swap"] for x in singles], axis=0), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(out["diff"], ct.contrastive_scores(h, s.mean(axis=0), g)["diff"], rtol=1e-5, atol=1e-6)


def test_per_pos_swap_sets_score_each_candidate_against_its_own_pos():
    rng = np.random.default_rng(5)
    L, D = 3, 8
    names = ["bank|bank.n.01", "bank|bank.v.01", "bank|bank.n.02"]
    bank = ev.GlossBank(names, rng.standard_normal((3, L, D)))
    es = ev.EvalSet(["i0"], ["c0"], [names], [[0]], ["n"])
    orig = rng.standard_normal((1, L, D))
    s_n, s_v = rng.standard_normal((1, L, D)), rng.standard_normal((2, L, D))
    out = ct.contrastive_layer_scores(orig, [{"n": s_n, "v": s_v}], bank, es, None, cand_pos=[["n", "v", "n"]])
    noun = ct.contrastive_scores(orig[0], s_n, bank.get([names[0], names[2]]))
    verb = ct.contrastive_scores(orig[0], s_v, bank.get([names[1]]))
    for m in ct.METHODS:
        np.testing.assert_allclose(out[m][0][[0, 2]], noun[m], rtol=1e-5)
        np.testing.assert_allclose(out[m][0][[1]], verb[m], rtol=1e-5)
    try:
        ct.contrastive_layer_scores(orig, [{"n": s_n}], bank, es, None, cand_pos=[["n", "v", "n"]])
    except KeyError:
        pass
    else:
        raise AssertionError("expected a KeyError for a POS without swap vectors")


def test_with_beta_interpolates_between_baseline_and_scorediff():
    h, s, g = _random()
    per_layer = {m: [v] for m, v in ct.contrastive_scores(h, s, g).items()}
    np.testing.assert_allclose(ct.with_beta(per_layer, 0.0)[0], per_layer["baseline"][0])
    np.testing.assert_allclose(ct.with_beta(per_layer, 1.0)[0], per_layer["scorediff"][0], rtol=1e-6, atol=1e-7)
