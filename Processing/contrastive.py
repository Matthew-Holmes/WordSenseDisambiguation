"""Contrastive scores: compare a target's hidden state with the same slot holding a random word.

The swapped sentences come from Processing/swaps.py. The algebra below is worked through, with numbers, in
Analysis/swap_methods.ipynb.

Notation, per layer (vectors in R^D):
    h            the target word's vector in the original sentence
    s_1..s_K     the vector at the same slot in K swapped sentences (the study's main run uses K = 1)
    g_k          the gloss vector of candidate sense k
    mu_s, mu_g   centring means fitted on dev (Processing/evaluation.fit_centering), exactly as in the baseline
    x~ = x - mu (centred),  x^ = x~ / ||x~|| (unit length),  cos(a, b) = <a, b> / (||a|| ||b||)

Scores for candidate k; the highest-scoring candidate is predicted. Using cosine distance (1 - cos) instead and
taking the lowest gives the same ranking for every method.
    baseline   cos(h~, g~_k)                              the layer study's score
    diff       cos(h~ - mean_j s~_j, g~_k)                method 1: subtract the activations, then cosine
    normdiff   cos(h^ - mean_j s^_j, g~_k)                method 2: normalise, subtract, then cosine
    scorediff  cos(h~, g~_k) - mean_j cos(s~_j, g~_k)     method 3: cosine with each, then subtract the two
    swap       mean_j cos(s~_j, g~_k)                     diagnostic: the slot with a random word, alone

The subtraction is always original minus swapped. The reverse order flips the sign of every score and so
reverses the ranking.

Consequences (each checked numerically in the notebook):
  * At any single layer, normdiff and scorediff rank candidates identically. With d = h^ - mean_j s^_j,
    scorediff_k = <d, g^_k> and normdiff_k = <d, g^_k> / ||d||, and ||d|| is the same for every candidate.
  * diff ranks like ||h~|| cos(h~, g~_k) - ||s~|| cos(s~, g~_k) (K = 1): it is scorediff with each cosine
    weighted by its vector's length.
  * mu_s cancels in diff (h~ - s~ = h - s) but not in normdiff or scorediff.
  * A band of layers averages per-layer scores (evaluation.config_scores). Over a band, normdiff weights
    layer l by 1 / ||d_l|| relative to scorediff, so the two can rank differently there.
  * If h~ = s~ at some layer, then d = 0 and every candidate scores 0 there; ties are broken at random.
    This happens at layer 0 when the target and the replacement end in the same sub-token.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from Processing.evaluation import Centering, EvalSet, GlossBank

METHODS = ("baseline", "diff", "normdiff", "scorediff", "swap")
_EPS = 1e-12


def unit(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """``x`` scaled to unit length along ``axis``; a zero vector stays zero."""
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), _EPS)


def contrastive_scores(h: np.ndarray, s: np.ndarray, g: np.ndarray, mu_s: np.ndarray | None = None,
                       mu_g: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Per-layer scores of ``n`` candidates under every method in METHODS.

    h : (L, D) the target in the original sentence.
    s : (K, L, D) or (L, D) the same slot in K swapped sentences.
    g : (n, L, D) candidate gloss vectors.
    mu_s, mu_g : (L, D) centring means, or None for no centring.
    Returns {method: (n, L) float32}.
    """
    h = np.asarray(h, dtype=np.float32)
    s = np.asarray(s, dtype=np.float32)
    g = np.asarray(g, dtype=np.float32)
    if s.ndim == 2:
        s = s[None]
    if mu_s is not None:
        h = h - mu_s
        s = s - mu_s
    if mu_g is not None:
        g = g - mu_g
    g_hat = unit(g)  # (n, L, D)
    h_hat, s_hat = unit(h), unit(s)  # (L, D), (K, L, D)
    cos_h = np.einsum("ld,nld->nl", h_hat, g_hat)  # cos(h~, g~_k)
    cos_s = np.einsum("kld,nld->nl", s_hat, g_hat) / s.shape[0]  # mean_j cos(s~_j, g~_k)
    d_raw = h - s.mean(axis=0)  # method 1: h~ - mean_j s~_j
    d_unit = h_hat - s_hat.mean(axis=0)  # method 2: h^ - mean_j s^_j
    return {
        "baseline": cos_h,
        "diff": np.einsum("ld,nld->nl", unit(d_raw), g_hat),
        "normdiff": np.einsum("ld,nld->nl", unit(d_unit), g_hat),
        "scorediff": cos_h - cos_s,
        "swap": cos_s,
    }


def contrastive_layer_scores(orig_vecs: np.ndarray, swap_vecs, bank: GlossBank, es: EvalSet,
                             centering: Centering | None, cand_pos: Sequence[Sequence[str]] | None = None
                             ) -> dict[str, list[np.ndarray]]:
    """Per-layer scores for every instance under every method: {method: [(n_i, L) per instance]}.

    orig_vecs : (N, L, D) target vectors, rows aligned with ``es``.
    swap_vecs : (N, K, L, D) or (N, L, D) when all of an instance's candidates share one swap set (always the
                case for v2 data). Otherwise a list with one {POS: (K, L, D)} dict per instance, together with
                ``cand_pos``, the POS of each candidate (Processing.swaps.synset_pos); each candidate is then
                scored against the swaps made for its own POS.
    centering : the baseline's Centering (fit on dev), or None for plain cosines.
    """
    if len(orig_vecs) != len(es) or len(swap_vecs) != len(es):
        raise ValueError(f"{len(orig_vecs)} original and {len(swap_vecs)} swapped rows for {len(es)} instances")
    mu_s = None if centering is None else centering.mu_sentence.astype(np.float32)
    mu_g = None if centering is None else centering.mu_gloss.astype(np.float32)
    out: dict[str, list[np.ndarray]] = {m: [] for m in METHODS}
    for i, cands in enumerate(es.candidates):
        g = bank.get(cands)
        h = orig_vecs[i]
        if isinstance(swap_vecs[i], dict):
            if cand_pos is None:
                raise ValueError("cand_pos is required when swap vectors are given per part of speech")
            pos_i = np.asarray(cand_pos[i])
            missing = set(pos_i.tolist()) - set(swap_vecs[i])
            if missing:
                raise KeyError(f"{es.instance_ids[i]}: no swap vectors for POS {sorted(missing)}")
            rows = {m: np.empty((len(cands), g.shape[1]), dtype=np.float32) for m in METHODS}
            for pos, s in swap_vecs[i].items():
                sel = np.flatnonzero(pos_i == pos)
                if len(sel):
                    for m, v in contrastive_scores(h, s, g[sel], mu_s, mu_g).items():
                        rows[m][sel] = v
        else:
            rows = contrastive_scores(h, swap_vecs[i], g, mu_s, mu_g)
        for m in METHODS:
            out[m].append(rows[m])
    return out


def with_beta(per_layer: dict[str, list[np.ndarray]], beta: float) -> list[np.ndarray]:
    """The family baseline - beta * swap: beta = 0 is the baseline, beta = 1 is scorediff."""
    return [b - beta * s for b, s in zip(per_layer["baseline"], per_layer["swap"])]
