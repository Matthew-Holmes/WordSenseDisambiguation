"""Metrics and post-processing for the metrics study (Processing/similarity.py)."""

from __future__ import annotations

import numpy as np

from Processing import similarity as sim
from Processing.contrastive import unit


def _random(n=5, L=3, D=12, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((L, D)), rng.standard_normal((n, L, D))


def test_metrics_on_a_hand_example():
    h = np.array([[3.0, 0.0, 0.0, 1.0]])  # (L=1, D=4)
    g = np.array([[[1.0, 0.0, 0.0, 1.0]], [[0.0, 2.0, 0.0, 0.0]]])  # (n=2, L=1, D=4)
    cos = sim.metric_scores(h, g, "cosine")[:, 0]
    np.testing.assert_allclose(cos, [4 / (np.sqrt(10) * np.sqrt(2)), 0.0], atol=1e-6)
    d2 = np.array([np.sqrt(4.0), np.sqrt(9 + 4 + 1)])  # |h - g_k|
    np.testing.assert_allclose(sim.metric_scores(h, g, "euclidean")[:, 0], -d2 / d2.mean(), rtol=1e-6)
    d1 = np.array([2.0, 3 + 2 + 1])
    np.testing.assert_allclose(sim.metric_scores(h, g, "manhattan")[:, 0], -d1 / d1.mean(), rtol=1e-6)
    np.testing.assert_allclose(sim.metric_scores(h, g, "dot")[:, 0], np.array([4.0, 0.0]) / np.sqrt(10)
                               / np.mean([np.sqrt(2), 2.0]), rtol=1e-6)


def test_pearson_and_spearman_are_correlations_across_dimensions():
    h, g = _random()
    pear = sim.metric_scores(h, g, "pearson")
    spear = sim.metric_scores(h, g, "spearman")
    rank = lambda v: np.argsort(np.argsort(v))  # noqa: E731
    for k in range(g.shape[0]):
        for layer in range(h.shape[0]):
            assert abs(pear[k, layer] - np.corrcoef(h[layer], g[k, layer])[0, 1]) < 1e-5
            assert abs(spear[k, layer] - np.corrcoef(rank(h[layer]), rank(g[k, layer]))[0, 1]) < 1e-5
    # Spearman ignores how large each entry is: any increasing transform leaves it unchanged
    np.testing.assert_allclose(sim.metric_scores(h ** 3, g ** 3, "spearman"), spear, atol=1e-6)


def test_scaling_keeps_each_layers_ranking():
    h, g = _random(n=7)
    raw = {"euclidean": -np.linalg.norm(g - h[None], axis=-1), "manhattan": -np.abs(g - h[None]).sum(-1),
           "dot": np.einsum("ld,nld->nl", h, g)}
    for metric, r in raw.items():
        np.testing.assert_array_equal(np.argsort(sim.metric_scores(h, g, metric), axis=0), np.argsort(r, axis=0))
    # on unit vectors, the Euclidean distance ranks exactly like the cosine
    np.testing.assert_array_equal(np.argmax(sim.metric_scores(unit(h), unit(g), "euclidean"), axis=0),
                                  np.argmax(sim.metric_scores(h, g, "cosine"), axis=0))


def test_fit_space_standardise_abtt_and_whiten():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((60, 2, 8)) * np.array([5.0, 3.0, 1, 1, 1, 1, 1, 0.5])  # (n, L, D), uneven variances
    X -= X.mean(axis=0)
    fit = sim.fit_space(X, n_components=4)
    for layer in range(2):
        v = fit.components[layer]
        np.testing.assert_allclose(v @ v.T, np.eye(4), atol=1e-5)  # orthonormal directions
        assert np.all(np.diff(fit.variances[layer]) <= 1e-6)  # largest variance first
    np.testing.assert_allclose(fit.std, X.std(axis=0), rtol=1e-5)
    np.testing.assert_allclose(sim.standardise(X, fit), X / X.std(axis=0), rtol=1e-5)
    np.testing.assert_allclose(fit.mean_variance, (X ** 2).sum(axis=(0, 2)) / 60 / 8, rtol=1e-5)

    out = sim.abtt(X, fit, k=2)  # nothing is left along the top two directions
    assert np.abs(np.einsum("nld,lkd->nlk", out, fit.components[:, :2])).max() < 1e-4
    np.testing.assert_allclose(sim.abtt(X, fit, k=0), X, atol=1e-6)

    w = sim.whiten(X, fit)  # direction i is scaled by sqrt(ebar / (e_i + ebar)); the rest is untouched
    before = np.einsum("nld,lkd->nlk", X, fit.components)
    after = np.einsum("nld,lkd->nlk", w, fit.components)
    expected = np.sqrt(fit.mean_variance[:, None] / (fit.variances + fit.mean_variance[:, None]))
    np.testing.assert_allclose(after, before * expected, rtol=1e-4, atol=1e-5)
    rest = lambda a: a - np.einsum("nlk,lkd->nld", np.einsum("nld,lkd->nlk", a, fit.components), fit.components)  # noqa: E731
    np.testing.assert_allclose(rest(w), rest(X), atol=1e-4)


def test_gloss_stats_csls_and_gloss_z():
    rng = np.random.default_rng(2)
    pool, glosses = rng.standard_normal((20, 2, 6)), rng.standard_normal((4, 2, 6))
    st = sim.gloss_stats(pool, glosses, ks=(3,))
    c = np.einsum("gld,pld->glp", unit(glosses), unit(pool))  # (G, L, P)
    np.testing.assert_allclose(st.mean, c.mean(-1), atol=1e-6)
    np.testing.assert_allclose(st.sd, c.std(-1), atol=1e-6)
    np.testing.assert_allclose(st.knn[3], -np.sort(-c, axis=-1)[..., :3].mean(-1), atol=1e-6)
    cos = rng.standard_normal((4, 2))
    np.testing.assert_allclose(sim.csls(cos, st.knn[3]), 2 * cos - st.knn[3])
    np.testing.assert_allclose(sim.gloss_z(cos, st.mean, st.sd), (cos - st.mean) / st.sd, rtol=1e-6)


def test_layer_z_keeps_each_layer_but_can_change_a_band():
    s = np.array([[0.30, 0.10], [0.20, 0.12], [0.00, 0.11]])  # (n=3 candidates, L=2 layers)
    z = sim.layer_z(s)
    np.testing.assert_array_equal(np.argmax(z, axis=0), np.argmax(s, axis=0))
    np.testing.assert_allclose(z.mean(axis=0), 0, atol=1e-6)
    assert np.argmax(s.mean(axis=1)) == 0  # the raw band average follows layer 0, which has the larger spread
    assert np.argmax(z.mean(axis=1)) == 1  # z-scored, both layers count equally and candidate 1 wins
    assert np.all(sim.layer_z(np.ones((3, 2))) == 0)
