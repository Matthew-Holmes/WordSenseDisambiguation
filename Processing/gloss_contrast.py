"""Gloss-side contrast: remove what a target's candidate glosses share before matching.

All of a lemma's definition prompts start with "lemma:", so the gloss vectors of its candidate senses share a
component: the lemma, and the shape of a definition. Subtracting the mean of the candidates' gloss vectors keeps
only what tells the senses apart. The algebra below is worked through, with numbers, in
Analysis/gloss_mean_methods.ipynb.

Notation, per layer, for one target with n candidate senses (vectors in R^D):
    h            the target word's vector
    g_1..g_n     the candidates' gloss vectors
    mu_s, mu_g   centring means fitted on dev (Processing/evaluation.fit_centering), exactly as in the baseline
    x~ = x - mu (centred),  x^ = x~ / ||x~|| (unit length),  m = mean_j g~_j,  m^ = mean_j g^_j

Scores for candidate k; the highest-scoring candidate is predicted.
    baseline   cos(h~, g~_k)                        the layer study's score
    glossmean  cos(h~, g~_k - m)                    remove the candidates' mean gloss vector
    unitmean   cos(h~, g^_k - m^)                   normalise each gloss vector, then remove the mean
    dot        <h^, g~_k> / mean_j ||g~_j||          diagnostic: the dot product, scaled to the size of a cosine
    alpha=a    cos(h~, g~_k - a m)                  partial removal: a = 0 is the baseline, a = 1 is glossmean

Consequences (each checked numerically in the notebook and in tests/test_gloss_contrast.py):
  * Subtracting at the score level, cos(h~, g~_k) - mean_j cos(h~, g~_j), never changes a ranking, because the
    subtracted term is the same for every candidate. That stays true over a band of layers. Only subtracting
    vectors can matter.
  * glossmean_k = (<h^, g~_k> - <h^, m>) / ||g~_k - m||. The term <h^, m> is the same for every candidate, so
    glossmean ranks by the dot product above the candidates' average, divided by how far each gloss vector is
    from the mean. Only the denominator differs from the dot product.
  * unitmean_k = (cos_k - mean_j cos_j) / ||g^_k - m^||, with cos_k the baseline score: the baseline's
    above-average margin, divided by how far each unit gloss vector is from the mean.
  * Both therefore pick a candidate that is above the candidates' average: on the dot product for glossmean,
    on the baseline cosine for unitmean.
  * With n = 2, g~_1 - m = -(g~_2 - m), so glossmean ranks like the dot product and unitmean ranks exactly
    like the baseline.
  * mu_g cancels in glossmean, since g~_k - m = g_k - mean_j g_j, but not in unitmean, whose unit vectors
    depend on the centring.
  * Leaving candidate k out of its own mean gives the same cosine: g~_k - mean_{j != k} g~_j = n/(n-1) (g~_k - m).
  * With a single candidate there is nothing to contrast: g~_1 - m = 0, and the score is 0.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from Processing.contrastive import unit
from Processing.evaluation import Centering, EvalSet, GlossBank

SCORES = ("baseline", "glossmean", "unitmean", "dot")
_EPS = 1e-12


def alpha_key(a: float) -> str:
    """Name of the partial-removal score with weight ``a``, e.g. 'alpha=0.5'."""
    return f"alpha={float(a) + 0.0:g}"


def gloss_contrast_scores(h: np.ndarray, g: np.ndarray, mu_s: np.ndarray | None = None,
                          mu_g: np.ndarray | None = None, alphas: Sequence[float] = ()) -> dict[str, np.ndarray]:
    """Per-layer scores of ``n`` candidates under every score in SCORES, plus one per partial-removal weight.

    h : (L, D) the target's vector.
    g : (n, L, D) the candidates' gloss vectors.
    mu_s, mu_g : (L, D) centring means, or None for no centring.
    alphas : weights a for cos(h~, g~_k - a m); each adds an entry named alpha_key(a).
    Returns {name: (n, L) float32}.
    """
    h = np.asarray(h, dtype=np.float32)
    g = np.asarray(g, dtype=np.float32)
    if mu_s is not None:
        h = h - mu_s
    if mu_g is not None:
        g = g - mu_g
    h_hat = unit(h)  # (L, D)
    g_hat = unit(g)  # (n, L, D)
    m = g.mean(axis=0)  # (L, D): the candidates' mean gloss vector
    m_hat = g_hat.mean(axis=0)  # the mean of the unit gloss vectors

    def cos_with(v: np.ndarray) -> np.ndarray:  # cos(h~, v_k) for every candidate: (n, L)
        return np.einsum("ld,nld->nl", h_hat, unit(v))

    out = {
        "baseline": np.einsum("ld,nld->nl", h_hat, g_hat),
        "glossmean": cos_with(g - m),
        "unitmean": cos_with(g_hat - m_hat),
        "dot": np.einsum("ld,nld->nl", h_hat, g) / np.maximum(np.linalg.norm(g, axis=-1).mean(axis=0), _EPS),
    }
    for a in alphas:
        out[alpha_key(a)] = cos_with(g - np.float32(a) * m)
    return {k: v.astype(np.float32) for k, v in out.items()}


def gloss_contrast_layer_scores(sent_vecs: np.ndarray, bank: GlossBank, es: EvalSet, centering: Centering | None,
                                alphas: Sequence[float] = ()) -> dict[str, list[np.ndarray]]:
    """Per-layer scores for every instance of ``es``: {name: [(n_i, L) per instance]}.

    sent_vecs : (N, L, D) target vectors, rows aligned with ``es``.
    centering : the baseline's Centering (fit on dev), or None for plain cosines.
    alphas : as in gloss_contrast_scores.
    """
    if len(sent_vecs) != len(es):
        raise ValueError(f"{len(sent_vecs)} sentence vectors for {len(es)} instances")
    mu_s = None if centering is None else centering.mu_sentence.astype(np.float32)
    mu_g = None if centering is None else centering.mu_gloss.astype(np.float32)
    out: dict[str, list[np.ndarray]] = {}
    for i, cands in enumerate(es.candidates):
        for name, v in gloss_contrast_scores(sent_vecs[i], bank.get(cands), mu_s, mu_g, alphas).items():
            out.setdefault(name, []).append(v)
    return out
