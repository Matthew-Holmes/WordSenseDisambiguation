"""Gloss-side contrast (Processing/gloss_contrast.py): hand-checked values and the algebraic identities."""

from __future__ import annotations

import numpy as np

from Processing import evaluation as ev
from Processing import gloss_contrast as gc


def _toy():
    """Dimensions: [sense A, sense B, sense C, lemma]. The target means A but is dominated by the lemma."""
    h = np.array([[1.0, 0.0, 0.0, 4.0]])  # (L=1, D=4)
    g = np.array([[[2.0, 0.0, 0.0, 2.0]],  # gloss A
                  [[0.0, 0.3, 0.0, 2.0]],  # gloss B: almost all lemma
                  [[0.0, 0.0, 1.0, 2.0]]])  # gloss C
    return h, g


def _random(n=5, L=4, D=16, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((L, D)), rng.standard_normal((n, L, D)), rng.standard_normal((2, L, D))


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_toy_example_matches_the_hand_calculation():
    h, g = _toy()
    out = {k: v[:, 0] for k, v in gc.gloss_contrast_scores(h, g).items()}
    r17 = np.sqrt(17.0)
    # baseline: <h, g> = 10, 8, 8; |g|^2 = 8, 4.09, 5
    np.testing.assert_allclose(out["baseline"], [10 / (r17 * np.sqrt(8)), 8 / (r17 * np.sqrt(4.09)), 8 / (r17 * np.sqrt(5))],
                               rtol=1e-5)
    # m = (2/3, 0.1, 1/3, 2): the lemma dimension cancels, and <h, g_k - m> is the A coordinate of g_k - m
    num = np.array([4 / 3, -2 / 3, -2 / 3])
    den = r17 * np.sqrt([16 / 9 + 0.01 + 1 / 9, 4 / 9 + 0.04 + 1 / 9, 4 / 9 + 0.01 + 4 / 9])
    np.testing.assert_allclose(out["glossmean"], num / den, rtol=1e-5)
    # dot: <h^, g_k> / mean_j |g_j|
    np.testing.assert_allclose(out["dot"], np.array([10, 8, 8]) / r17 / np.mean(np.sqrt([8, 4.09, 5])), rtol=1e-5)
    assert np.argmax(out["baseline"]) == 1  # the almost-all-lemma gloss B wins the plain cosine...
    assert np.argmax(out["glossmean"]) == 0  # ...removing the shared mean recovers sense A
    assert np.argmax(out["unitmean"]) == 1  # B is the only candidate above the mean cosine, so unitmean keeps it


def test_score_level_subtraction_never_changes_the_ranking():
    h, g, _ = _random()
    base = gc.gloss_contrast_scores(h, g)["baseline"]  # (n, L)
    shifted = base - base.mean(axis=0, keepdims=True)  # cos(h, g_k) - mean_j cos(h, g_j), per layer
    np.testing.assert_array_equal(np.argsort(-base, axis=0), np.argsort(-shifted, axis=0))
    band = [1, 2, 3]  # also over a band of layers
    assert np.argmax(base[:, band].mean(axis=1)) == np.argmax(shifted[:, band].mean(axis=1))


def test_closed_forms():
    h, g, _ = _random()
    out = gc.gloss_contrast_scores(h, g)
    h_hat = h / np.linalg.norm(h, axis=-1, keepdims=True)
    g_hat = g / np.linalg.norm(g, axis=-1, keepdims=True)
    m, m_hat = g.mean(axis=0), g_hat.mean(axis=0)
    dots = np.einsum("ld,nld->nl", h_hat, g)
    np.testing.assert_allclose(
        out["glossmean"], (dots - np.einsum("ld,ld->l", h_hat, m)) / np.linalg.norm(g - m, axis=-1), rtol=1e-4, atol=1e-6)
    cos = out["baseline"]
    np.testing.assert_allclose(out["unitmean"], (cos - cos.mean(axis=0)) / np.linalg.norm(g_hat - m_hat, axis=-1),
                               rtol=1e-4, atol=1e-6)
    np.testing.assert_array_equal(np.argmax(out["dot"], axis=0), np.argmax(dots, axis=0))


def test_both_contrasts_pick_a_candidate_above_the_average():
    rng = np.random.default_rng(3)
    for _ in range(50):
        h, g = rng.standard_normal((3, 8)), rng.standard_normal((6, 3, 8))
        out = gc.gloss_contrast_scores(h, g)
        dots = np.einsum("ld,nld->nl", h, g)
        for layer in range(3):
            k = np.argmax(out["glossmean"][:, layer])
            assert dots[k, layer] >= dots[:, layer].mean()
            k = np.argmax(out["unitmean"][:, layer])
            assert out["baseline"][k, layer] >= out["baseline"][:, layer].mean()


def test_two_candidates_glossmean_is_the_dot_product_and_unitmean_is_the_baseline():
    rng = np.random.default_rng(1)
    for _ in range(50):
        h, g = rng.standard_normal((4, 8)), rng.standard_normal((2, 4, 8))
        out = gc.gloss_contrast_scores(h, g)
        np.testing.assert_allclose(out["glossmean"][0], -out["glossmean"][1], rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(np.argmax(out["glossmean"], axis=0), np.argmax(out["dot"], axis=0))
        np.testing.assert_array_equal(np.argmax(out["unitmean"], axis=0), np.argmax(out["baseline"], axis=0))


def test_the_gloss_centring_cancels_in_glossmean_only():
    h, g, mus = _random()
    mu_s, mu_g = mus
    a = gc.gloss_contrast_scores(h, g, mu_s, mu_g)
    b = gc.gloss_contrast_scores(h, g, mu_s, None)
    np.testing.assert_allclose(a["glossmean"], b["glossmean"], rtol=1e-4, atol=1e-6)
    assert not np.allclose(a["unitmean"], b["unitmean"], atol=1e-3)


def test_leaving_the_candidate_out_of_its_own_mean_gives_the_same_cosine():
    h, g, _ = _random(n=4)
    out = gc.gloss_contrast_scores(h, g)["glossmean"]
    for k in range(4):
        others = np.delete(g, k, axis=0).mean(axis=0)
        for layer in range(g.shape[1]):
            assert abs(_cos(h[layer], g[k, layer] - others[layer]) - out[k, layer]) < 1e-5


def test_alpha_family_ends_and_a_single_candidate():
    h, g, mus = _random()
    out = gc.gloss_contrast_scores(h, g, *mus, alphas=(0.0, 1.0, 0.5))
    np.testing.assert_array_equal(out[gc.alpha_key(0.0)], out["baseline"])
    np.testing.assert_array_equal(out[gc.alpha_key(1.0)], out["glossmean"])
    assert gc.alpha_key(-0.0) == "alpha=0" and gc.alpha_key(0.5) == "alpha=0.5"
    one = gc.gloss_contrast_scores(h, g[:1], *mus)
    assert np.all(one["glossmean"] == 0) and np.all(one["unitmean"] == 0)


def test_layer_scores_match_the_evaluation_module():
    rng = np.random.default_rng(2)
    names = [f"s{i}" for i in range(7)]
    bank = ev.GlossBank(names, rng.standard_normal((7, 3, 8)).astype(np.float16))
    es = ev.EvalSet(instance_ids=["a", "b"], clusters=["x", "y"], candidates=[names[:3], names[2:7]],
                    gold=[[0], [1]], pos=["n", "v"])
    sent = rng.standard_normal((2, 3, 8)).astype(np.float16)
    centering = ev.Centering(rng.standard_normal((3, 8)).astype(np.float32), rng.standard_normal((3, 8)).astype(np.float32))
    out = gc.gloss_contrast_layer_scores(sent, bank, es, centering, alphas=(0.5,))
    old = ev.layer_scores(sent, bank, es, centering)
    for a, b in zip(out["baseline"], old):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    assert [v.shape for v in out["glossmean"]] == [(3, 3), (5, 3)]
    direct = gc.gloss_contrast_scores(sent[1], bank.get(names[2:7]), centering.mu_sentence, centering.mu_gloss,
                                      alphas=(0.5,))
    for name in direct:
        np.testing.assert_array_equal(out[name][1], direct[name])
