"""Similarity metrics and cheap post-processing for gloss matching (the metrics study).

The layer study scores a candidate sense by the cosine between the centred target vector h and the centred gloss
vector g_k. This module provides the alternatives the metrics study compares. Each one is worked through, with
numbers, in Analysis/metrics_methods.ipynb.

Every function takes vectors that are already centred with the layer study's dev means. Scores are per layer,
shape (n, L) for n candidates; the highest score wins. A band of layers averages per-layer scores, so each score is
scaled to be comparable across layers.

Metrics, for one target h (L, D) and its candidates g (n, L, D):
    cosine     <h^, g^_k>                                  the layer study's score
    euclidean  -||h - g_k|| / mean_j ||h - g_j||          negative distance, relative to the candidates' average
    manhattan  -|h - g_k|_1 / mean_j |h - g_j|_1          the same with the L1 distance
    pearson    cos(h - mean(h), g_k - mean(g_k))           correlation across the D dimensions
    spearman   pearson on the ranks of the D entries       ignores how large the outlier dimensions are
    dot        <h^, g_k> / mean_j ||g_k||                  rewards long gloss vectors
Dividing by a quantity shared by all candidates (the averages above) keeps each layer's ranking and only sets the
scale used when a band is averaged. A Euclidean distance between unit vectors would rank like the cosine, so the
distances use the centred vectors as they are.

Re-shaping the space before the cosine, fitted per layer on reference vectors (fit_space):
    standardise   divide every dimension by its standard deviation (a diagonal whitening)
    abtt          remove the projection on the top k principal directions ("all-but-the-top")
    whiten        shrink the top principal directions towards the average variance: direction i is scaled by
                  sqrt(ebar / (e_i + ebar)), where e_i is its variance and ebar the average variance per dimension

Post-processing of the cosines:
    csls          2 cos(h, g_k) - r(g_k), where r(g) is the mean cosine between g and its k nearest vectors in a
                  pool of reference targets. It discounts "hub" glosses that are close to many targets.
    gloss_z       (cos(h, g_k) - mu(g_k)) / sd(g_k), with mu and sd over the reference pool: a gloss is judged
                  against how similar it usually is to targets.
    layer_z       at each layer, z-score the candidates' scores before averaging a band, so that every layer in the
                  band counts equally. Within a single layer the ranking is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from Processing.contrastive import unit

METRICS = ("cosine", "euclidean", "manhattan", "pearson", "spearman", "dot")
_EPS = 1e-12


# ---------------------------------------------------------------------------------------------- metrics
def centred_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks of the entries along the last axis (0 = smallest), minus the mean rank. Float32, ties broken by order."""
    x = np.asarray(x)
    ranks = np.empty(x.shape, dtype=np.float32)
    order = np.argsort(x, axis=-1, kind="stable")
    np.put_along_axis(ranks, order, np.arange(x.shape[-1], dtype=np.float32), axis=-1)
    return ranks - (x.shape[-1] - 1) / 2.0


def metric_scores(h: np.ndarray, g: np.ndarray, metric: str) -> np.ndarray:
    """Per-layer scores of n candidates under one of METRICS. h (L, D) and g (n, L, D) are centred."""
    h = np.asarray(h, dtype=np.float32)
    g = np.asarray(g, dtype=np.float32)
    if metric == "cosine":
        return np.einsum("ld,nld->nl", unit(h), unit(g))
    if metric in ("euclidean", "manhattan"):
        diff = g - h[None]
        d = np.linalg.norm(diff, axis=-1) if metric == "euclidean" else np.abs(diff).sum(-1)  # (n, L)
        return -d / np.maximum(d.mean(axis=0, keepdims=True), _EPS)
    if metric == "pearson":
        return np.einsum("ld,nld->nl", unit(h - h.mean(-1, keepdims=True)), unit(g - g.mean(-1, keepdims=True)))
    if metric == "spearman":
        return np.einsum("ld,nld->nl", unit(centred_ranks(h)), unit(centred_ranks(g)))
    if metric == "dot":
        return np.einsum("ld,nld->nl", unit(h), g) / np.maximum(np.linalg.norm(g, axis=-1).mean(0), _EPS)
    raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")


# ------------------------------------------------------------------------------------ re-shaping the space
@dataclass
class SpaceFit:
    """Per-layer statistics of a set of reference vectors, for standardise / abtt / whiten."""

    std: np.ndarray  # (L, D) standard deviation of every dimension
    components: np.ndarray  # (L, r, D) principal directions, largest variance first
    variances: np.ndarray  # (L, r) variance along each direction
    mean_variance: np.ndarray  # (L,) total variance / D: the average variance per dimension


def fit_space(X: np.ndarray, n_components: int = 128) -> SpaceFit:
    """Fit SpaceFit on reference vectors X (n, L, D), which should already be centred."""
    X = np.asarray(X, dtype=np.float32)
    n, L, D = X.shape
    r = min(n_components, n, D)
    std = np.empty((L, D), np.float32)
    comps = np.empty((L, r, D), np.float32)
    var = np.empty((L, r), np.float32)
    mean_var = np.empty(L, np.float32)
    for layer in range(L):
        xl = X[:, layer] - X[:, layer].mean(axis=0)
        std[layer] = xl.std(axis=0)
        _, s, vt = np.linalg.svd(xl, full_matrices=False)
        comps[layer], var[layer] = vt[:r], (s[:r] ** 2) / n
        mean_var[layer] = (s ** 2).sum() / n / D
    return SpaceFit(std, comps, var, mean_var)


def standardise(x: np.ndarray, fit: SpaceFit) -> np.ndarray:
    """Divide every dimension by its reference standard deviation. x (..., L, D)."""
    return np.asarray(x, dtype=np.float32) / np.maximum(fit.std, _EPS)


def _coefficients(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Coordinates of x (..., L, D) along the directions v (L, k, D): (..., L, k). One matrix product per layer."""
    flat = x.reshape(int(np.prod(x.shape[:-2])), *x.shape[-2:])
    out = np.empty((flat.shape[0], v.shape[0], v.shape[1]), np.float32)
    for layer in range(v.shape[0]):
        out[:, layer] = flat[:, layer] @ v[layer].T
    return out.reshape(*x.shape[:-2], v.shape[0], v.shape[1])


def _combine(coef: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The vectors with coordinates coef (..., L, k) along v (L, k, D): (..., L, D)."""
    flat = coef.reshape(int(np.prod(coef.shape[:-2])), *coef.shape[-2:])
    out = np.empty((flat.shape[0], v.shape[0], v.shape[2]), np.float32)
    for layer in range(v.shape[0]):
        out[:, layer] = flat[:, layer] @ v[layer]
    return out.reshape(*coef.shape[:-2], v.shape[0], v.shape[2])


def abtt(x: np.ndarray, fit: SpaceFit, k: int) -> np.ndarray:
    """Remove the projection on the top-k principal directions at every layer. x (..., L, D)."""
    x = np.asarray(x, dtype=np.float32)
    v = np.ascontiguousarray(fit.components[:, :k])  # (L, k, D)
    return x - _combine(_coefficients(x, v), v)


def whiten(x: np.ndarray, fit: SpaceFit) -> np.ndarray:
    """Soft whitening: scale principal direction i by sqrt(ebar / (e_i + ebar)), leaving the rest unchanged."""
    x = np.asarray(x, dtype=np.float32)
    v = fit.components  # (L, r, D)
    shrink = np.sqrt(fit.mean_variance[:, None] / (fit.variances + fit.mean_variance[:, None])) - 1.0  # (L, r)
    return x + _combine(_coefficients(x, v) * shrink, v)


# -------------------------------------------------------------------------------------- post-processing
@dataclass
class GlossStats:
    """How similar each gloss usually is to targets: cosines with a reference pool of target vectors."""

    mean: np.ndarray  # (G, L)
    sd: np.ndarray  # (G, L)
    knn: dict[int, np.ndarray]  # k -> (G, L): mean of the k largest cosines


def gloss_stats(pool: np.ndarray, glosses: np.ndarray, ks: tuple[int, ...] = (10,)) -> GlossStats:
    """Statistics of cos(t, g) over pool targets t, for every gloss g. pool (P, L, D), glosses (G, L, D)."""
    p, gl = unit(pool), unit(glosses)
    L = gl.shape[1]
    mean = np.empty((gl.shape[0], L), np.float32)
    sd = np.empty_like(mean)
    knn = {k: np.empty_like(mean) for k in ks}
    for layer in range(L):
        c = gl[:, layer] @ p[:, layer].T  # (G, P)
        mean[:, layer], sd[:, layer] = c.mean(axis=1), c.std(axis=1)
        top = -np.sort(-c, axis=1)
        for k in ks:
            knn[k][:, layer] = top[:, :k].mean(axis=1)
    return GlossStats(mean, sd, knn)


def csls(cos: np.ndarray, knn: np.ndarray) -> np.ndarray:
    """CSLS for one target's candidates: 2 cos(h, g_k) - r(g_k). cos and knn are (n, L)."""
    return 2.0 * cos - knn


def gloss_z(cos: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """(cos(h, g_k) - mu(g_k)) / sd(g_k), all (n, L)."""
    return (cos - mean) / np.maximum(sd, 1e-6)


def layer_z(scores: np.ndarray) -> np.ndarray:
    """z-score the candidates' scores at each layer (columns of an (n, L) array). Ties at a layer give zeros."""
    s = np.asarray(scores, dtype=np.float32)
    return (s - s.mean(axis=0, keepdims=True)) / np.maximum(s.std(axis=0, keepdims=True), 1e-6)
