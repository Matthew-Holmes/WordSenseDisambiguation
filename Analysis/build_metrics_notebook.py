"""Writes Analysis/metrics_methods.ipynb (the worked-examples notebook for the metrics study).

Regenerate and execute with:
    .venv/bin/python Analysis/build_metrics_notebook.py
    (cd Analysis && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace metrics_methods.ipynb)
"""

from pathlib import Path

import nbformat as nbf

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("md", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


# =====================================================================================================
md(r"""
# Metrics and post-processing for gloss matching: every function, with worked examples

The layer study scores each candidate sense by the **cosine** between the centred target vector $h$ and the
centred gloss vector $g_k$, and predicts the highest. The metrics study asks whether another similarity measure,
or a cheap correction applied afterwards, picks out the right sense more often on **sense-balanced** data. This
notebook documents each function in `Processing/similarity.py`, with examples small enough to check by hand, and
then looks at the real Llama vectors.

| Code | What it does |
|---|---|
| `Processing/similarity.py` | the metrics (Part 1), the re-shaped spaces (Part 2) and the post-processing (Part 3) |
| `Analysis/metrics_study.py` | runs the experiment; results are in `Report/METRICS_STUDY.md` |
| `tests/test_similarity.py` | the same checks as unit tests |

All vectors below are already centred, as in the layer study. Outputs were produced by executing the notebook top
to bottom.
""")

code(r"""
import sys
from pathlib import Path

import numpy as np

REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "Processing").is_dir())
for p in (REPO, REPO / "Analysis"):
    sys.path.insert(0, str(p))

from Processing import evaluation as ev
from Processing import similarity as sim
from Processing.contrastive import unit

np.set_printoptions(precision=4, suppress=True)


def show(scores, labels="AB"):
    return "   ".join(f"{labels[k]} {v:+.4f}" for k, v in enumerate(scores)) + f"   -> picks {labels[int(np.argmax(scores))]}"
""")

# =====================================================================================================
md(r"""
---
## Part 1. Metrics (`metric_scores`)

For one target $h$ (shape $L \times D$) and $n$ candidates $g$ ($n \times L \times D$), each metric returns an
$n \times L$ array of per-layer scores; the highest wins.

| Metric | Score for candidate $k$ |
|---|---|
| `cosine` | $\langle \hat h, \hat g_k\rangle$ |
| `euclidean` | $-\lVert h - g_k\rVert \,/\, \frac1n\sum_j \lVert h - g_j\rVert$ |
| `manhattan` | $-\lVert h - g_k\rVert_1 \,/\, \frac1n\sum_j \lVert h - g_j\rVert_1$ |
| `pearson` | the correlation between the $D$ entries of $h$ and of $g_k$ |
| `spearman` | the correlation between the *ranks* of those entries |
| `dot` | $\langle \hat h, g_k\rangle \,/\, \frac1n\sum_j \lVert g_j\rVert$ |

The divisions by an average over candidates don't change any layer's ranking. They only put every layer on a
similar scale, so that averaging a band of layers doesn't let one layer dominate.

### 1.1 A hand example

$h = (3, 0, 0, 1)$, $g_A = (1, 0, 0, 1)$, $g_B = (0, 2, 0, 0)$, one layer.

- cosine: $\langle h, g_A\rangle = 4$, $\lVert h\rVert = \sqrt{10}$, $\lVert g_A\rVert = \sqrt2$, so $4/\sqrt{20} = 0.8944$; B is orthogonal to $h$, so 0.
- euclidean: $\lVert h - g_A\rVert = \lVert(2,0,0,0)\rVert = 2$ and $\lVert h - g_B\rVert = \lVert(3,-2,0,1)\rVert = \sqrt{14} = 3.7417$.
  Their mean is 2.8708, so the scores are $-2/2.8708 = -0.6967$ and $-3.7417/2.8708 = -1.3033$.
- manhattan: distances 2 and $3+2+0+1 = 6$, mean 4, scores $-0.5$ and $-1.5$.
- dot: $\langle \hat h, g\rangle = 4/\sqrt{10},\ 0$, divided by the mean gloss length $(\sqrt2 + 2)/2 = 1.7071$: 0.7410 and 0.
""")

code(r"""
h = np.array([[3.0, 0.0, 0.0, 1.0]])
g = np.array([[[1.0, 0.0, 0.0, 1.0]], [[0.0, 2.0, 0.0, 0.0]]])
for metric in sim.METRICS:
    print(f"{metric:10s}", show(sim.metric_scores(h, g, metric)[:, 0]))
""")

md(r"""
### 1.2 Distances between unit vectors rank like the cosine

For unit vectors, $\lVert \hat h - \hat g\rVert^2 = 2 - 2\cos(h, g)$, so a smaller distance is exactly a larger
cosine. That is why the study's `euclidean` uses the centred vectors as they are: on unit vectors it would just
be the cosine again. The difference from the cosine is that it also penalises a gloss vector for being long or
short, since $\lVert h - g\rVert^2 = \lVert h\rVert^2 + \lVert g\rVert^2 - 2\langle h, g\rangle$.
""")

code(r"""
rng = np.random.default_rng(0)
h_r, g_r = rng.standard_normal((3, 12)), rng.standard_normal((6, 3, 12))  # L = 3 layers, D = 12, n = 6
same = np.array_equal(np.argsort(sim.metric_scores(unit(h_r), unit(g_r), "euclidean"), 0),
                      np.argsort(sim.metric_scores(h_r, g_r, "cosine"), 0))
print("euclidean on unit vectors orders the candidates exactly like the cosine:", same)
""")

md(r"""
### 1.3 Pearson and Spearman: correlations across the dimensions

`pearson` subtracts each vector's own mean over its $D$ entries, then takes the cosine: it is the correlation
coefficient between the two lists of entries. `spearman` does the same with the entries' *ranks*. It only asks
whether the dimensions that are high in $h$ are also high in $g$, not by how much.

That matters when a few dimensions are much larger than the rest, which is common in LLM hidden states. In the
example below, one dimension dominates. Gloss A shares that dimension but disagrees on every other one; gloss B
lacks it but agrees on the rest. The cosine picks A, and the rank correlation picks B. The ranks are
$h \to (5, 4, 3, 1, 0, 2)$, $A \to (5, 0, 1, 3, 4, 2)$ and $B \to (0, 5, 4, 2, 1, 3)$. After subtracting the mean
rank of 2.5, the products sum to $-2.5$ for A and $+2.5$ for B. Each list has squared length 17.5, so the
correlations are $\mp 0.1429$.
""")

code(r"""
h = np.array([[20.0, 1.0, 0.5, -0.5, -1.0, 0.2]])
g = np.array([[[20.0, -1.0, -0.5, 0.5, 1.0, -0.2]],   # A: shares the big dimension, disagrees on the rest
              [[-1.0, 1.1, 0.4, -0.6, -0.9, 0.3]]])   # B: lacks it, agrees on the rest
for metric in ("cosine", "pearson", "spearman"):
    print(f"{metric:9s}", show(sim.metric_scores(h, g, metric)[:, 0]))
print("ranks of h:", sim.centred_ranks(h[0]) + 2.5, " A:", sim.centred_ranks(g[0, 0]) + 2.5, " B:", sim.centred_ranks(g[1, 0]) + 2.5)
print("spearman is unchanged by any increasing transform (here, cubing):",
      np.allclose(sim.metric_scores(h ** 3, g ** 3, "spearman"), sim.metric_scores(h, g, "spearman")))
""")

# =====================================================================================================
md(r"""
---
## Part 2. Re-shaping the space before the cosine (`fit_space`, `standardise`, `abtt`, `whiten`)

These transforms change *which directions count* before taking the cosine. Their statistics are fitted, per
layer, on reference vectors: in the study, dev_natural's target vectors for the sentence side and its candidates'
gloss vectors for the gloss side. `fit_space` stores each dimension's standard deviation, the principal
directions $v_i$ with their variances $e_i$, and the average variance per dimension $\bar e$.

- **`standardise`** divides every dimension by its standard deviation, so a dimension that varies a lot no longer
  dominates. It is a diagonal whitening.
- **`abtt`** ("all-but-the-top") removes the projection on the top $k$ principal directions. Those usually carry
  variation shared by everything in the space rather than meaning.
- **`whiten`** shrinks each top direction instead of deleting it: direction $i$ is scaled by
  $\sqrt{\bar e / (e_i + \bar e)}$. A direction with typical variance is barely touched; a direction with ten times
  the average variance is scaled by $\sqrt{1/11} = 0.30$.

A synthetic example: 500 vectors in 4 dimensions, where one direction has variance 100 and the rest 1.
""")

code(r"""
rng = np.random.default_rng(1)
X = rng.standard_normal((500, 1, 4)) * np.array([10.0, 1.0, 1.0, 1.0])  # (n, L=1, D=4): dimension 0 dominates
X -= X.mean(axis=0)
fit = sim.fit_space(X, n_components=4)
print("variances of the principal directions:", fit.variances[0].round(2), " average per dimension:", fit.mean_variance.round(2))
print("top direction:", fit.components[0, 0].round(3))
print("whitening scale per direction, sqrt(ebar / (e_i + ebar)):",
      np.sqrt(fit.mean_variance[0] / (fit.variances[0] + fit.mean_variance[0])).round(3))

x = np.array([[[5.0, 1.0, -1.0, 0.5]]])
print("\nx              ", x[0, 0])
print("standardise(x) ", sim.standardise(x, fit)[0, 0].round(3), " (each dimension divided by its sd:", fit.std[0].round(2), ")")
print("abtt(x, k=1)   ", sim.abtt(x, fit, 1)[0, 0].round(3), " (the dominant dimension is removed)")
print("whiten(x)      ", sim.whiten(x, fit)[0, 0].round(3), " (the dominant dimension is shrunk)")
""")

# =====================================================================================================
md(r"""
---
## Part 3. Post-processing the cosines (`gloss_stats`, `csls`, `gloss_z`, `layer_z`)

Some gloss vectors are close to *many* targets, whatever the context: they are "hubs". With the plain cosine, a
hub tends to win wherever it is a candidate. The first two corrections compare each gloss with how similar it
usually is to targets. They use a reference pool of target vectors: dev_natural's targets, unlabelled.

- **`csls`**: $2\cos(h, g_k) - r(g_k)$, where $r(g)$ is the mean cosine between $g$ and its $k$ nearest pool
  targets. This is the hubness correction from bilingual word-embedding retrieval (Conneau et al., 2018). Its
  other term, $r(h)$, is the same for every candidate of a target, so it is dropped.
- **`gloss_z`**: $(\cos(h, g_k) - \mu(g_k)) / \sigma(g_k)$, with $\mu$ and $\sigma$ the mean and standard
  deviation of $\cos(t, g_k)$ over the pool. A gloss then scores well only if $h$ is unusually close to it.

A toy example in two dimensions. The pool is spread around the circle, gloss A is a hub, close to most of the pool,
and gloss B is not. The target is slightly closer to A by raw cosine, but much closer to B relative to what each
gloss usually scores.
""")

code(r"""
angles = np.linspace(-0.6, 0.6, 9)
pool = np.stack([np.cos(angles), np.sin(angles)], axis=-1)[:, None, :]  # (P=9, L=1, D=2): targets near the x-axis
gl = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])  # A on the x-axis (a hub for this pool), B on the y-axis
st = sim.gloss_stats(pool, gl, ks=(3,))
h = np.array([[0.75, 0.66]])
cos = sim.metric_scores(h, gl, "cosine")
print("cosine  ", show(cos[:, 0]))
print("mu(g)   ", st.mean[:, 0].round(4), "   sd(g)", st.sd[:, 0].round(4), "   r(g), k=3:", st.knn[3][:, 0].round(4))
print("csls    ", show(sim.csls(cos, st.knn[3])[:, 0]))
print("gloss_z ", show(sim.gloss_z(cos, st.mean, st.sd)[:, 0]))
""")

md(r"""
**`layer_z`** works across layers rather than across glosses. A band score is the average of the per-layer
scores, so a layer where the candidates' scores are spread widely outweighs a layer where they are close.
`layer_z` z-scores the candidates at each layer first, so every layer in the band counts equally. Within a single
layer the ranking can't change. Over a band it can:
""")

code(r"""
s = np.array([[0.30, 0.10], [0.20, 0.12], [0.00, 0.11]])  # 3 candidates x 2 layers
print("per-layer winners, raw and z-scored:", np.argmax(s, 0), np.argmax(sim.layer_z(s), 0))
print("band average, raw      ", s.mean(1).round(4), "-> candidate", int(np.argmax(s.mean(1))))
print("band average, layer_z  ", sim.layer_z(s).mean(1).round(4), "-> candidate", int(np.argmax(sim.layer_z(s).mean(1))))
""")

md(r"""
The **ensemble** in the study averages `layer_z` scores over all six combinations of target representation
(*plain_last*, *echo_last*, *echo_mean*) and gloss representation (*mean*, *last*). It is the same idea across
representations instead of layers.
""")

# =====================================================================================================
md(r"""
---
## Part 4. The real vectors (Llama-3.2-1B)

These use the layer study's vectors and centring. The question for each correction is whether its target is
there: are a few directions dominant, and are some glosses hubs?
""")

code(r"""
import json

import layer_study as ls
from Processing.script import load_sample, open_stores
from Processing.semcor_data import load_dataset

ds = load_dataset()
sent_store, gloss_store = open_stores()
nat_ids = load_sample("dev_natural_80")["instance_ids"]
dev_ids = list(dict.fromkeys(nat_ids + load_sample("dev_balanced_200")["instance_ids"]))
dev_all, nat = ls.eval_set(ds, dev_ids), ls.eval_set(ds, nat_ids)
keys = sorted({k for c in dev_all.candidates for k in c})
bank = ev.GlossBank(keys, gloss_store.get(keys, "mean"))
centering = ev.fit_centering(sent_store.get(dev_ids, "echo_last"), bank, dev_all)  # the layer study's centring
targets = sent_store.get(nat_ids, "echo_last").astype(np.float32) - centering.mu_sentence
ref_keys = sorted({k for c in nat.candidates for k in c})
glosses = bank.get(ref_keys).astype(np.float32) - centering.mu_gloss
print(f"reference: {len(targets)} dev_natural targets and {len(glosses)} of their candidates' glosses")
""")

md(r"""
### 4.1 Are a few directions dominant?

The share of the total variance carried by the top principal directions, and by the single largest dimension, at
a few layers. If the variance were spread evenly, the top 10 of 2,048 directions would carry about 0.5%.
""")

code(r"""
sfit, gfit = sim.fit_space(targets), sim.fit_space(glosses)
print(f"{'layer':>5s} | {'targets: top 1 / top 10 PCs / top dim':>36s} | {'glosses: top 1 / top 10 PCs / top dim':>36s}")
for layer in (0, 3, 6, 9, 12, 16):
    row = []
    for f in (sfit, gfit):
        total = f.mean_variance[layer] * 2048
        row.append(f"{f.variances[layer, 0] / total:9.1%} {f.variances[layer, :10].sum() / total:9.1%} "
                   f"{(f.std[layer] ** 2).max() / total:9.1%}")
    print(f"{layer:5d} | {row[0]:>36s} | {row[1]:>36s}")
""")

md(r"""
### 4.2 Are some glosses hubs?

$\mu(g)$, the mean cosine between a gloss and the 411 dev_natural targets at layer 6, varies from gloss to gloss.
If it were the same for every gloss, `gloss_z` and `csls` could change nothing. The last line checks the link to
the gloss-mean study's finding: are wordier glosses bigger hubs?
""")

code(r"""
layer = 6
st = sim.gloss_stats(targets[:, [layer]], glosses[:, [layer]], ks=(10,))
mu = st.mean[:, 0]
print(f"mu(g) at layer {layer}: mean {mu.mean():+.3f}, sd {mu.std():.3f}, range {mu.min():+.3f} to {mu.max():+.3f}")
words = np.array([len(ds.synsets[k.split('|', 1)[1]]["gloss"].split()) for k in ref_keys])
rank = lambda v: np.argsort(np.argsort(v))
print(f"Spearman correlation between a gloss's word count and mu(g): {np.corrcoef(rank(words), rank(mu))[0, 1]:+.3f}")
for i in np.argsort(-mu)[:3]:
    print(f"  hub  {mu[i]:+.3f}  {ref_keys[i]:32s} {ds.synsets[ref_keys[i].split('|', 1)[1]]['gloss'][:60]}")
for i in np.argsort(mu)[:3]:
    print(f"  anti {mu[i]:+.3f}  {ref_keys[i]:32s} {ds.synsets[ref_keys[i].split('|', 1)[1]]['gloss'][:60]}")
""")

md(r"""
### 4.3 Cross-check against the layer study's scorer

`metric_scores(..., "cosine")` on the centred vectors must reproduce `evaluation.layer_scores`.
""")

code(r"""
es = ls.eval_set(ds, load_sample("test_balanced_200")["instance_ids"][:20])
sv = sent_store.get(es.instance_ids, "echo_last")
test_keys = sorted({k for c in es.candidates for k in c})
test_bank = ev.GlossBank(test_keys, gloss_store.get(test_keys, "mean"))
old = ev.layer_scores(sv, test_bank, es, centering)
new = [sim.metric_scores(sv[i].astype(np.float32) - centering.mu_sentence,
                         test_bank.get(c).astype(np.float32) - centering.mu_gloss, "cosine") for i, c in enumerate(es.candidates)]
print("max |cosine - layer_scores| over 20 test_balanced instances:", max(np.abs(a - b).max() for a, b in zip(new, old)))
""")

md(r"""
---
## Where the experiment lives

- `Analysis/metrics_study.py` runs every method at the layer study's configuration on the pooled sense-balanced
  samples, and again with each method's own layers chosen on dev_balanced.
- `Report/METRICS_STUDY.md` has the results.
- `tests/test_similarity.py` repeats the checks in Parts 1–3 as unit tests:
  `.venv/bin/python -m tests.run_tests similarity`.
""")


def build(path: Path) -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [nbf.v4.new_markdown_cell(t) if kind == "md" else nbf.v4.new_code_cell(t) for kind, t in CELLS]
    nbf.write(nb, path)
    print(f"wrote {path} ({len(CELLS)} cells)")


if __name__ == "__main__":
    build(Path(__file__).resolve().parent / "metrics_methods.ipynb")
