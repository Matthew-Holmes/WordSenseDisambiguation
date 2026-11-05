"""Evaluation for gloss-matching word-sense disambiguation (pure numpy).

Each evaluated instance has

* a sentence-side vector per layer, ``sent_vecs[i]`` with shape ``(L, D)``, and
* ``n_i`` candidate senses (WordNet synset names, in WordNet order), each with a gloss vector per
  layer, looked up in a :class:`GlossBank` of shape ``(G, L, D)``.

The candidate whose gloss vector is most similar to the sentence vector is predicted. The module is
built so that every choice (layers, representation variant, centring statistics) is made on a dev
split and then applied, unchanged, to a held-out test split:

* :func:`fit_centering` estimates the mean sentence vector and the mean gloss vector on one split.
  The returned :class:`Centering` is passed explicitly to every scoring call, so test data never
  leaks into it.
* :func:`instance_metrics` breaks ties uniformly at random and handles several gold senses. Report
  bug B8: ``np.argsort`` handed ties to the earlier-listed (more frequent) sense.
* :func:`bootstrap_ci` and :func:`paired_bootstrap` resample *clusters* (sentences), because targets
  from the same sentence are not independent.
* :func:`select_best` picks a configuration from dev results only (report bug B9).

Arrays may be float16; all arithmetic is done in float32 or float64.

Metric definitions, per instance with ``n`` candidates and gold set ``G``:

``acc1``
    1 if the top-scoring candidate is gold. Ties for the top score are broken uniformly at random.
``mrr``
    ``1 / r``, where ``r`` is the 1-based rank of the best-ranked gold candidate. Ties are resolved
    in expectation. Suppose the best gold score is shared by ``t`` candidates, ``k`` of them gold,
    and ``g`` candidates score strictly higher. Then ``r = g + (t + 1) / (k + 1)``, which is the
    expected position of the first gold under a random tie order. For ``k = 1`` this is the usual
    average rank.
``norm_rank``
    ``1 - (r - 1) / (n - 1)``: 1 means gold ranked first, 0 means last, and 0.5 is chance (single
    gold). This matches the convention in the Analysis notebooks. Defined as 1.0 when ``n == 1``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from itertools import chain
from typing import Sequence

import numpy as np

METRICS = ("acc1", "mrr", "norm_rank")
_EPS = 1e-12
_BAND_NAME = re.compile(r"L(\d+)(?:-(\d+))?$")


# ---------------------------------------------------------------------------------------- data
@dataclass(frozen=True)
class EvalSet:
    """The instances to evaluate, in the same row order as the sentence-vector array.

    clusters    : sentence id of each instance; the unit resampled by the bootstrap.
    candidates  : synset names per instance, in WordNet order (``candidates[i][0]`` is the first sense).
    gold        : indices into ``candidates[i]`` of the gold sense(s); at least one per instance.
    pos         : WordNet POS letter per instance ('n', 'v', 'a', 'r'), for breakdowns.
    """

    instance_ids: list[str]
    clusters: list[str]
    candidates: list[list[str]]
    gold: list[list[int]]
    pos: list[str]

    def __post_init__(self) -> None:
        n = len(self.instance_ids)
        for field in ("clusters", "candidates", "gold", "pos"):
            if len(getattr(self, field)) != n:
                raise ValueError(f"EvalSet.{field} has {len(getattr(self, field))} entries, expected {n}")
        for i, (cands, gold) in enumerate(zip(self.candidates, self.gold)):
            if len(cands) == 0:
                raise ValueError(f"instance {self.instance_ids[i]!r} has no candidates")
            if len(gold) == 0:
                raise ValueError(f"instance {self.instance_ids[i]!r} has no gold index")
            if any(not 0 <= g < len(cands) for g in gold):
                raise ValueError(
                    f"instance {self.instance_ids[i]!r}: gold {list(gold)} out of range for {len(cands)} candidates"
                )

    def __len__(self) -> int:
        return len(self.instance_ids)

    @property
    def n_candidates(self) -> np.ndarray:
        return np.array([len(c) for c in self.candidates], dtype=np.int64)

    def gold_mask(self, i: int) -> np.ndarray:
        """Boolean mask over instance ``i``'s candidates marking the gold sense(s)."""
        mask = np.zeros(len(self.candidates[i]), dtype=bool)
        mask[list(self.gold[i])] = True
        return mask

    def subset(self, idx: Sequence[int]) -> "EvalSet":
        """A new EvalSet holding the given rows, in the given order."""
        idx = list(idx)
        return EvalSet(
            instance_ids=[self.instance_ids[i] for i in idx],
            clusters=[self.clusters[i] for i in idx],
            candidates=[self.candidates[i] for i in idx],
            gold=[self.gold[i] for i in idx],
            pos=[self.pos[i] for i in idx],
        )


class GlossBank:
    """Gloss vectors keyed by synset name: ``vectors[k]`` (shape ``(L, D)``) belongs to ``names[k]``."""

    def __init__(self, names: list[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors)
        if vectors.ndim != 3:
            raise ValueError(f"vectors must have shape (G, L, D), got {vectors.shape}")
        if len(names) != vectors.shape[0]:
            raise ValueError(f"{len(names)} names for {vectors.shape[0]} gloss vectors")
        self.names = list(names)
        self.vectors = vectors
        self._index = {name: k for k, name in enumerate(self.names)}
        if len(self._index) != len(self.names):
            raise ValueError("GlossBank names must be unique")

    def __len__(self) -> int:
        return len(self.names)

    def __contains__(self, name: str) -> bool:
        return name in self._index

    @property
    def n_layers(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[2])

    def indices(self, names: Sequence[str]) -> np.ndarray:
        """Row indices of ``names``. Raises KeyError for a synset that is not in the bank."""
        try:
            return np.array([self._index[name] for name in names], dtype=np.int64)
        except KeyError as exc:
            raise KeyError(f"synset {exc.args[0]!r} is not in the GlossBank") from None

    def get(self, names: Sequence[str]) -> np.ndarray:
        """Gloss vectors for ``names`` in the given order: an ``(n, L, D)`` copy, in the bank's dtype."""
        return self.vectors[self.indices(names)]


@dataclass
class Centering:
    """Mean vectors subtracted before taking cosines; fit on one split, applied to any split."""

    mu_sentence: np.ndarray  # (L, D)
    mu_gloss: np.ndarray  # (L, D)


# ----------------------------------------------------------------------------------- centring
def _mean_rows(array: np.ndarray, rows: np.ndarray | None = None, chunk: int = 128) -> np.ndarray:
    """Mean over axis 0 (optionally of selected rows), accumulated in float64, chunk by chunk.

    This avoids materialising a full float32 copy of a large float16 array.
    """
    n = array.shape[0] if rows is None else len(rows)
    if n == 0:
        raise ValueError("cannot take the mean of zero rows")
    total = np.zeros(array.shape[1:], dtype=np.float64)
    for start in range(0, n, chunk):
        block = array[start : start + chunk] if rows is None else array[rows[start : start + chunk]]
        total += block.astype(np.float64).sum(axis=0)
    return (total / n).astype(np.float32)


def _check_sent_vecs(sent_vecs: np.ndarray, es: EvalSet, bank: GlossBank) -> np.ndarray:
    sv = np.asarray(sent_vecs)
    if sv.ndim != 3:
        raise ValueError(f"sentence vectors must have shape (N, L, D), got {sv.shape}")
    if sv.shape[0] != len(es):
        raise ValueError(f"{sv.shape[0]} sentence vectors for {len(es)} instances")
    if sv.shape[1:] != bank.vectors.shape[1:]:
        raise ValueError(f"sentence vectors are {sv.shape[1:]} per instance, gloss vectors {bank.vectors.shape[1:]}")
    return sv


def _check_centering(centering: Centering, bank: GlossBank) -> None:
    expected = bank.vectors.shape[1:]
    for name in ("mu_sentence", "mu_gloss"):
        if getattr(centering, name).shape != expected:
            raise ValueError(f"Centering.{name} has shape {getattr(centering, name).shape}, expected {expected}")


def fit_centering(sent_vecs: np.ndarray, bank: GlossBank, es: EvalSet) -> Centering:
    """Means for centring, estimated on this split only.

    The gloss mean is taken over the *unique* candidate synsets of ``es``, so frequent lemmas,
    whose candidate lists recur across many instances, do not dominate it.
    """
    sent_vecs = _check_sent_vecs(sent_vecs, es, bank)
    unique = bank.indices(sorted(set(chain.from_iterable(es.candidates))))
    unique.sort()
    return Centering(mu_sentence=_mean_rows(sent_vecs), mu_gloss=_mean_rows(bank.vectors, unique))


def no_centering(L: int, D: int) -> Centering:
    """Zero means, i.e. plain cosine similarity."""
    zeros = np.zeros((L, D), dtype=np.float32)
    return Centering(mu_sentence=zeros, mu_gloss=zeros.copy())


# ------------------------------------------------------------------------------------- scoring
def _cosine(g: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Per-layer cosine between candidates ``g`` (n, L, D) and one sentence vector ``s`` (L, D) -> (n, L)."""
    num = np.einsum("nld,ld->nl", g, s)
    den = np.linalg.norm(g, axis=-1) * np.linalg.norm(s, axis=-1)[None, :]
    return (num / np.maximum(den, _EPS)).astype(np.float32)


def layer_scores(
    sent_vecs: np.ndarray, bank: GlossBank, es: EvalSet, centering: Centering
) -> list[np.ndarray]:
    """Centred cosine between each instance's sentence vector and its candidates' gloss vectors.

    Returns one ``(n_i, L)`` float32 array per instance: rows are candidates (WordNet order), columns
    are layers.
    """
    sent_vecs = _check_sent_vecs(sent_vecs, es, bank)
    _check_centering(centering, bank)
    mu_s = centering.mu_sentence.astype(np.float32)
    mu_g = centering.mu_gloss.astype(np.float32)
    out = []
    for i, cands in enumerate(es.candidates):
        s = sent_vecs[i].astype(np.float32) - mu_s
        g = bank.get(cands).astype(np.float32) - mu_g
        out.append(_cosine(g, s))
    return out


def _as_layers(layers: Sequence[int]) -> list[int]:
    layers = [int(layer) for layer in layers]
    if not layers:
        raise ValueError("a configuration needs at least one layer")
    return layers


def config_scores(per_layer: list[np.ndarray], layers: Sequence[int]) -> list[np.ndarray]:
    """Collapse per-layer scores to one score per candidate: the mean over ``layers``."""
    layers = _as_layers(layers)
    return [np.asarray(s)[:, layers].mean(axis=1) for s in per_layer]


# ------------------------------------------------------------------------------------- metrics
def _rank_stats(scores: np.ndarray, gold_mask: np.ndarray, keys: np.ndarray):
    """Metrics for a batch of score vectors over the same candidates.

    scores : (B, n) float64; ``-inf`` is allowed, NaN is not.
    gold_mask : (n,) bool.
    keys : (B, n) uniform random tie-break keys in [0, 1).

    Returns ``(acc1, mrr, norm_rank)``, each of shape (B,).
    """
    if np.isnan(scores).any():
        raise ValueError("scores contain NaN")
    n = scores.shape[1]
    top = scores.max(axis=1, keepdims=True)
    pick = np.where(scores == top, keys, -1.0).argmax(axis=1)  # uniform among tied maxima
    acc1 = gold_mask[pick].astype(np.float64)

    best_gold = np.where(gold_mask[None, :], scores, -np.inf).max(axis=1, keepdims=True)
    above = (scores > best_gold).sum(axis=1)
    tied = scores == best_gold
    t = tied.sum(axis=1)
    k = (tied & gold_mask[None, :]).sum(axis=1)
    rank = above + (t + 1) / (k + 1)  # expected 1-based position of the best gold
    mrr = 1.0 / rank
    norm_rank = np.ones_like(rank) if n == 1 else 1.0 - (rank - 1.0) / (n - 1)
    return acc1, mrr, norm_rank


def instance_metrics(scores: list[np.ndarray], es: EvalSet, seed: int = 0) -> dict[str, np.ndarray]:
    """Per-instance ``acc1``, ``mrr`` and ``norm_rank`` (see the module docstring) for one score per candidate.

    ``seed`` drives the tie-breaking for ``acc1``. Tie-break keys are drawn for every instance,
    whether or not it has ties, so results are reproducible for a fixed seed and input.
    """
    if len(scores) != len(es):
        raise ValueError(f"{len(scores)} score vectors for {len(es)} instances")
    rng = np.random.default_rng(seed)
    out = {m: np.empty(len(es), dtype=np.float64) for m in METRICS}
    for i, sc in enumerate(scores):
        sc = np.asarray(sc, dtype=np.float64)
        if sc.ndim != 1 or sc.shape[0] != len(es.candidates[i]):
            raise ValueError(
                f"instance {es.instance_ids[i]!r}: scores of shape {sc.shape} for {len(es.candidates[i])} candidates"
            )
        keys = rng.random(sc.shape[0])
        a, m, r = _rank_stats(sc[None, :], es.gold_mask(i), keys[None, :])
        out["acc1"][i], out["mrr"][i], out["norm_rank"][i] = a[0], m[0], r[0]
    return out


# ----------------------------------------------------------------------------------- bootstrap
def bootstrap_distribution(
    values: np.ndarray, clusters: Sequence[str], n_boot: int = 2000, seed: int = 0, block: int = 250
) -> np.ndarray:
    """Bootstrap distribution of the mean, resampling whole clusters with replacement.

    Each replicate is the pooled mean of all instances in the drawn clusters, so a large cluster
    weighs more than a small one, exactly as in the full-sample mean.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1:
        raise ValueError(f"values must be 1-D, got shape {v.shape}")
    if len(clusters) != len(v):
        raise ValueError(f"{len(clusters)} cluster labels for {len(v)} values")
    if len(v) == 0:
        raise ValueError("cannot bootstrap an empty array")
    if not np.all(np.isfinite(v)):
        raise ValueError("values must be finite")
    _, codes = np.unique(np.asarray(clusters), return_inverse=True)
    n_clusters = int(codes.max()) + 1
    sums = np.bincount(codes, weights=v, minlength=n_clusters)
    counts = np.bincount(codes, minlength=n_clusters).astype(np.float64)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, block):
        draws = rng.integers(0, n_clusters, size=(min(block, n_boot - start), n_clusters))
        out[start : start + len(draws)] = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return out


def bootstrap_ci(
    values: np.ndarray, clusters: Sequence[str], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> dict:
    """Mean and percentile ``(1 - alpha)`` cluster-bootstrap CI: ``{"mean", "lo", "hi"}``."""
    v = np.asarray(values, dtype=np.float64)
    dist = bootstrap_distribution(v, clusters, n_boot=n_boot, seed=seed)
    lo, hi = np.quantile(dist, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi)}


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, clusters: Sequence[str], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> dict:
    """Paired comparison of two systems scored on the same instances.

    Returns ``{"diff", "lo", "hi", "p_two_sided"}``. ``diff`` is ``mean(a - b)`` and the CI is a
    percentile cluster-bootstrap CI of it. The p-value is
    ``2 * min(P*(diff <= 0), P*(diff >= 0))``, with +1 smoothing and capped at 1, where ``P*`` is
    taken over the bootstrap replicates.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays differ in shape: {a.shape} vs {b.shape}")
    d = a - b
    dist = bootstrap_distribution(d, clusters, n_boot=n_boot, seed=seed)
    lo, hi = np.quantile(dist, [alpha / 2, 1 - alpha / 2])
    below = (np.sum(dist <= 0) + 1) / (n_boot + 1)
    above = (np.sum(dist >= 0) + 1) / (n_boot + 1)
    return {
        "diff": float(d.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "p_two_sided": float(min(1.0, 2.0 * min(below, above))),
    }


def summarize(
    metrics: dict[str, np.ndarray], clusters: Sequence[str], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> dict[str, dict]:
    """``bootstrap_ci`` for every metric in a per-instance metrics dict."""
    return {name: bootstrap_ci(v, clusters, n_boot=n_boot, seed=seed, alpha=alpha) for name, v in metrics.items()}


# ----------------------------------------------------------------------------------- baselines
def random_baseline(es: EvalSet, n_draws: int = 200, seed: int = 0) -> dict[str, np.ndarray]:
    """Per-instance expected metrics under random scores, by Monte Carlo through the same metric code.

    See :func:`random_baseline_exact` for the closed forms; the two agree up to Monte Carlo error.
    """
    rng = np.random.default_rng(seed)
    out = {m: np.empty(len(es), dtype=np.float64) for m in METRICS}
    for i, cands in enumerate(es.candidates):
        n = len(cands)
        a, m, r = _rank_stats(rng.random((n_draws, n)), es.gold_mask(i), rng.random((n_draws, n)))
        out["acc1"][i], out["mrr"][i], out["norm_rank"][i] = a.mean(), m.mean(), r.mean()
    return out


def random_baseline_exact(es: EvalSet) -> dict[str, np.ndarray]:
    """Closed-form expectations under a uniformly random ranking.

    With ``k`` distinct gold senses among ``n`` candidates, the best gold's position ``R`` satisfies
    ``P(R = r) = C(n - r, k - 1) / C(n, k)``. Then ``acc1 = k / n``, ``E[R] = (n + 1) / (k + 1)``,
    ``norm_rank = 1 - (E[R] - 1) / (n - 1)`` and ``mrr = E[1 / R]``.
    """
    out = {m: np.empty(len(es), dtype=np.float64) for m in METRICS}
    for i, (cands, gold) in enumerate(zip(es.candidates, es.gold)):
        n, k = len(cands), len(set(gold))
        total = math.comb(n, k)
        out["acc1"][i] = k / n
        out["mrr"][i] = sum(math.comb(n - r, k - 1) / total / r for r in range(1, n - k + 2))
        out["norm_rank"][i] = 1.0 if n == 1 else 1.0 - ((n + 1) / (k + 1) - 1.0) / (n - 1)
    return out


def first_sense_baseline(es: EvalSet) -> dict[str, np.ndarray]:
    """Always predict ``candidates[0]`` and rank the rest in WordNet order.

    WordNet orders senses by SemCor tag counts, so on SemCor this baseline is optimistic.
    """
    scores = [-np.arange(len(c), dtype=np.float64) for c in es.candidates]  # strictly decreasing: no ties
    return instance_metrics(scores, es, seed=0)


def _derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """A uniformly random permutation with no fixed points (rejection sampling; ~e tries on average)."""
    identity = np.arange(n)
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == identity):
            return perm


def shuffled_sentence_metrics(
    sent_vecs: np.ndarray,
    bank: GlossBank,
    es: EvalSet,
    centering: Centering,
    layers: Sequence[int],
    n_perm: int = 20,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Control: score each instance's candidates against *another* instance's sentence vector.

    Each of ``n_perm`` random derangements pairs instance ``i`` with a partner ``j != i``. Scores
    use the same centred cosine and the same layer averaging as :func:`layer_scores` followed by
    :func:`config_scores`. Per-instance metrics are averaged over the permutations. If a system beats
    this control only by chance, its signal does not come from the sentence.
    """
    sent_vecs = _check_sent_vecs(sent_vecs, es, bank)
    _check_centering(centering, bank)
    layers = _as_layers(layers)
    n_inst = len(es)
    if n_inst < 2:
        raise ValueError("need at least two instances to shuffle sentence vectors")
    rng = np.random.default_rng(seed)
    perms = np.stack([_derangement(n_inst, rng) for _ in range(n_perm)])  # (P, N)

    mu_s = centering.mu_sentence[layers].astype(np.float32)
    mu_g = centering.mu_gloss[layers].astype(np.float32)
    s = sent_vecs[:, layers].astype(np.float32) - mu_s  # (N, l, D)
    s /= np.maximum(np.linalg.norm(s, axis=-1, keepdims=True), _EPS)

    out = {m: np.empty(n_inst, dtype=np.float64) for m in METRICS}
    for i, cands in enumerate(es.candidates):
        g = bank.get(cands)[:, layers].astype(np.float32) - mu_g  # (n, l, D)
        g /= np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), _EPS)
        partners = s[perms[:, i]]  # (P, l, D); never instance i itself
        sc = np.einsum("pld,nld->pnl", partners, g).mean(axis=2).astype(np.float64)  # (P, n)
        a, m, r = _rank_stats(sc, es.gold_mask(i), rng.random(sc.shape))
        out["acc1"][i], out["mrr"][i], out["norm_rank"][i] = a.mean(), m.mean(), r.mean()
    return out


# ------------------------------------------------------------------------- configs & selection
def contiguous_bands(n_layers: int, widths: Sequence[int] = (1, 2, 3, 4)) -> dict[str, list[int]]:
    """Every contiguous band of layers of the given widths: ``{"L5": [5], "L4-6": [4, 5, 6], ...}``.

    Ordered by width, then start layer.
    """
    if n_layers < 1:
        raise ValueError("n_layers must be positive")
    bands: dict[str, list[int]] = {}
    for width in widths:
        if not 1 <= width <= n_layers:
            raise ValueError(f"band width {width} is invalid for {n_layers} layers")
        for start in range(n_layers - width + 1):
            stop = start + width - 1
            bands[f"L{start}" if width == 1 else f"L{start}-{stop}"] = list(range(start, stop + 1))
    return bands


def evaluate_configs(
    per_layer: list[np.ndarray], es: EvalSet, configs: dict[str, Sequence[int]], seed: int = 0
) -> dict[str, dict[str, np.ndarray]]:
    """Per-instance metrics for every named layer configuration (same tie-break seed for each)."""
    return {name: instance_metrics(config_scores(per_layer, layers), es, seed=seed) for name, layers in configs.items()}


def _band_key(name: str, configs: dict[str, Sequence[int]] | None) -> tuple[float, float]:
    """(band width, first layer) for tie-breaking; unknown names sort last."""
    if configs is not None and name in configs:
        layers = _as_layers(configs[name])
        return float(len(layers)), float(min(layers))
    match = _BAND_NAME.search(name)
    if match:
        start = int(match.group(1))
        stop = int(match.group(2)) if match.group(2) else start
        return float(stop - start + 1), float(start)
    return math.inf, math.inf


def select_best(
    results: dict[str, dict[str, np.ndarray]],
    metric: str = "acc1",
    configs: dict[str, Sequence[int]] | None = None,
    tol: float = 1e-12,
) -> str:
    """The configuration with the highest mean ``metric``. Call it on dev results only.

    Ties (within ``tol``) go to the narrower band, then to the lower starting layer, then to the
    alphabetically first name. Band width and start are read from ``configs`` when given, or else
    parsed from names ending in ``L<a>`` or ``L<a>-<b>`` (prefixes such as ``"echo/L4-6"`` are
    allowed).
    """
    if not results:
        raise ValueError("no results to select from")
    means = {name: float(np.mean(r[metric])) for name, r in results.items()}
    best = max(means.values())
    tied = [name for name, value in means.items() if value >= best - tol]
    return min(tied, key=lambda name: (*_band_key(name, configs), name))


def by_group(values: np.ndarray, groups: Sequence[str]) -> dict[str, dict]:
    """Mean and count of ``values`` per group label (e.g. per POS): ``{group: {"mean", "n"}}``."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) != len(groups):
        raise ValueError(f"{len(values)} values for {len(groups)} group labels")
    labels, codes = np.unique(np.asarray(groups), return_inverse=True)
    sums = np.bincount(codes, weights=values, minlength=len(labels))
    counts = np.bincount(codes, minlength=len(labels))
    return {str(label): {"mean": float(s / c), "n": int(c)} for label, s, c in zip(labels, sums, counts)}
