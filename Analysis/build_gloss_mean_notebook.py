"""Writes Analysis/gloss_mean_methods.ipynb (the worked-examples notebook for the gloss-mean study).

The notebook is generated so that its text and code stay in one reviewable file. Regenerate and execute with:
    .venv/bin/python Analysis/build_gloss_mean_notebook.py
    (cd Analysis && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace gloss_mean_methods.ipynb)
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
# Removing the candidates' mean gloss vector: every function, with worked examples

**The idea.** Every definition prompt for a word starts with the word itself, as in *"plant: buildings for carrying
on industrial labor"*. So the gloss vectors of a word's candidate senses share a component: the lemma, and the
shape of a definition. The cosine between the target's vector and a gloss vector is partly a cosine with that
shared component, which says nothing about which sense is meant. Subtracting the mean of the candidates' gloss
vectors before the cosine should leave only what tells the senses apart. This is the gloss-side counterpart of the
swapped-word contrasts in `swap_methods.ipynb`.

This notebook documents the function that implements it, with an example small enough to check by hand, the
identities that say what the subtraction can and cannot change, and real examples from Llama-3.2-1B.

| Code | What it does |
|---|---|
| `Processing/gloss_contrast.py` | computes the scores (Part 1) |
| `Analysis/gloss_mean_study.py` | runs the experiment; results are in `Report/GLOSS_MEAN_STUDY.md` |
| `tests/test_gloss_contrast.py` | the same checks as unit tests |

Outputs below were produced by executing the notebook top to bottom.
""")

code(r"""
import sys
from pathlib import Path

import numpy as np

REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "Processing").is_dir())
for p in (REPO, REPO / "Analysis"):
    sys.path.insert(0, str(p))

from Processing import evaluation as ev
from Processing import gloss_contrast as gc
from Processing.contrastive import unit

np.set_printoptions(precision=4, suppress=True)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
""")

# =====================================================================================================
md(r"""
---
## Part 1. The scores (`gloss_contrast_scores`)

Per layer, for one target with $n$ candidate senses:

| Symbol | Meaning |
|---|---|
| $h$ | the target word's vector |
| $g_1, \dots, g_n$ | the candidates' gloss vectors |
| $\mu_s,\ \mu_g$ | mean sentence and gloss vectors on dev; the layer study's centring |
| $\tilde x = x - \mu$ | centred vector; $\hat x = \tilde x / \lVert \tilde x \rVert$ is centred and unit length |
| $m = \frac1n \sum_j \tilde g_j$ | the candidates' mean gloss vector |
| $\hat m = \frac1n \sum_j \hat g_j$ | the mean of the unit gloss vectors |

| Score | Formula for candidate $k$ | |
|---|---|---|
| `baseline` | $\cos(\tilde h, \tilde g_k)$ | the layer study's score |
| `glossmean` | $\cos(\tilde h, \tilde g_k - m)$ | remove the candidates' mean gloss vector |
| `unitmean` | $\cos(\tilde h, \hat g_k - \hat m)$ | normalise each gloss vector, then remove the mean |
| `dot` | $\langle \hat h, \tilde g_k \rangle / \frac1n\sum_j \lVert \tilde g_j \rVert$ | diagnostic: the dot product, scaled to the size of a cosine |
| `alpha=a` | $\cos(\tilde h, \tilde g_k - a\,m)$ | partial removal: $a = 0$ is the baseline, $a = 1$ is `glossmean` |

The highest score is predicted. A band of layers averages the per-layer scores, as in the layer study.

### 1.1 A toy example to check by hand

Four dimensions: senses A, B and C, and a *lemma* dimension shared by every prompt for this word. The target means
A, but, being the word itself, its vector is dominated by the lemma. Each gloss carries the same lemma load (2).
Gloss B is short and generic, so it is almost all lemma.

| | A | B | C | lemma |
|---|---:|---:|---:|---:|
| $h$ | 1 | 0 | 0 | 4 |
| $g_A$ | 2 | 0 | 0 | 2 |
| $g_B$ | 0 | 0.3 | 0 | 2 |
| $g_C$ | 0 | 0 | 1 | 2 |

No centring here, so $\tilde x = x$. $\lVert h\rVert = \sqrt{17} = 4.1231$.

**Baseline.** $\langle h, g\rangle = 10, 8, 8$ and $\lVert g\rVert = \sqrt8, \sqrt{4.09}, \sqrt5 = 2.8284, 2.0224,
2.2361$, so the cosines are $10/11.662 = 0.8575$, $8/8.339 = 0.9594$ and $8/9.220 = 0.8677$. **B wins**, because its
direction is almost the lemma's, and so is the target's.

**Remove the mean.** $m = (2/3,\ 0.1,\ 1/3,\ 2)$, so the lemma dimension cancels in every $g_k - m$:

| | $g_k - m$ | $\langle h, g_k - m\rangle$ | $\lVert g_k - m\rVert$ | `glossmean` |
|---|---|---:|---:|---:|
| A | $(4/3,\ -0.1,\ -1/3,\ 0)$ | 1.3333 | $\sqrt{1.8989} = 1.3780$ | $1.3333 / (4.1231 \times 1.3780) = 0.2347$ |
| B | $(-2/3,\ 0.2,\ -1/3,\ 0)$ | −0.6667 | $\sqrt{0.5956} = 0.7717$ | $-0.6667 / 3.1819 = -0.2095$ |
| C | $(-2/3,\ -0.1,\ 2/3,\ 0)$ | −0.6667 | $\sqrt{0.8989} = 0.9481$ | $-0.6667 / 3.9093 = -0.1705$ |

**A wins.** With the shared lemma gone, only the sense dimensions are compared.
""")

code(r"""
h = np.array([[1.0, 0.0, 0.0, 4.0]])  # (L=1, D=4)
g = np.array([[[2.0, 0.0, 0.0, 2.0]], [[0.0, 0.3, 0.0, 2.0]], [[0.0, 0.0, 1.0, 2.0]]])  # (n=3, L=1, D=4)
toy = {k: v[:, 0] for k, v in gc.gloss_contrast_scores(h, g).items()}
for name, v in toy.items():
    print(f"{name:10s} A {v[0]:+.4f}   B {v[1]:+.4f}   C {v[2]:+.4f}   -> picks {'ABC'[int(np.argmax(v))]}")

hand = {"baseline": [0.8575, 0.9594, 0.8677], "glossmean": [0.2347, -0.2095, -0.1705]}
print("\nmatches the hand calculation to 4 decimals:", all(np.allclose(toy[k], v, atol=5e-5) for k, v in hand.items()))
""")

md(r"""
`dot` also picks A here: $\langle \hat h, g_k\rangle = 10, 8, 8$ over $\sqrt{17}$, divided by the mean gloss
length $2.3623$. `unitmean` keeps B, and section 1.3 shows why.

### 1.2 Subtracting at the score level does nothing

The obvious score-level version, $\cos(\tilde h, \tilde g_k) - \frac1n\sum_j \cos(\tilde h, \tilde g_j)$, subtracts the
same number from every candidate. It can never change a ranking, at one layer or averaged over a band. So unlike
the swap study's method 3, it is not a method here. Only subtracting *vectors* can matter.
""")

code(r"""
rng = np.random.default_rng(0)
h_r, g_r = rng.standard_normal((4, 16)), rng.standard_normal((6, 4, 16))  # L = 4 layers, D = 16, n = 6
base = gc.gloss_contrast_scores(h_r, g_r)["baseline"]  # (n, L)
shifted = base - base.mean(axis=0, keepdims=True)
print("same order at every layer:", np.array_equal(np.argsort(-base, axis=0), np.argsort(-shifted, axis=0)))
print("same pick over the band of layers 1-3:", np.argmax(base[:, 1:].mean(1)) == np.argmax(shifted[:, 1:].mean(1)))
""")

md(r"""
### 1.3 What the vector subtraction actually changes

Expand the cosine. Because $\lVert \hat h\rVert = 1$,

$$\texttt{glossmean}_k = \frac{\langle \hat h, \tilde g_k\rangle - \langle \hat h, m\rangle}{\lVert \tilde g_k - m\rVert},
\qquad
\texttt{unitmean}_k = \frac{\cos_k - \frac1n\sum_j \cos_j}{\lVert \hat g_k - \hat m\rVert},$$

where $\cos_k$ is the baseline score. In both, the subtracted term in the numerator is the same for every
candidate, so the numerator ranks exactly like the dot product (`glossmean`) or like the baseline (`unitmean`).
**Everything new is in the denominator.**

- `glossmean` ranks by *how far the dot product is above the candidates' average*, divided by how far the gloss
  vector is from the candidates' mean. It is the dot product with a different length correction. The baseline
  divides by $\lVert \tilde g_k\rVert$, which includes the shared component; `glossmean` divides by the length of
  the distinctive part only.
- `unitmean` is the baseline's above-average margin, divided by how unusual the gloss's direction is.
- Every candidate below the average gets a negative score, so both always pick a candidate that is above the
  average: on the dot product for `glossmean`, on the cosine for `unitmean`. That is why `unitmean` keeps B in
  the toy: B is the only gloss whose cosine is above the mean (0.9594 against 0.8949).
""")

code(r"""
out = gc.gloss_contrast_scores(h_r, g_r)
h_hat = h_r / np.linalg.norm(h_r, axis=-1, keepdims=True)
g_hat = g_r / np.linalg.norm(g_r, axis=-1, keepdims=True)
m, m_hat = g_r.mean(0), g_hat.mean(0)
dots = np.einsum("ld,nld->nl", h_hat, g_r)
gm_closed = (dots - np.einsum("ld,ld->l", h_hat, m)) / np.linalg.norm(g_r - m, axis=-1)
um_closed = (out["baseline"] - out["baseline"].mean(0)) / np.linalg.norm(g_hat - m_hat, axis=-1)
print("max |glossmean - closed form|:", np.abs(out["glossmean"] - gm_closed).max())
print("max |unitmean - closed form| :", np.abs(out["unitmean"] - um_closed).max())

above = all(dots[np.argmax(out["glossmean"][:, l]), l] >= dots[:, l].mean()
            and out["baseline"][np.argmax(out["unitmean"][:, l]), l] >= out["baseline"][:, l].mean() for l in range(4))
print("both picks are above the average at every layer:", above)
""")

md(r"""
### 1.4 Two candidates: `glossmean` is the dot product, `unitmean` is the baseline

With $n = 2$, $\tilde g_1 - m = (\tilde g_1 - \tilde g_2)/2 = -(\tilde g_2 - m)$. Both differences have the same
length, so the denominators are equal, and the two scores are each other's negatives. `glossmean` then ranks by
the dot product, and `unitmean` (by the same argument on unit vectors) ranks exactly like the baseline. Any
change `unitmean` makes comes from targets with three or more candidates. In the study's four samples, 12.5–18%
of targets have exactly two candidates.
""")

code(r"""
same_dot = same_base = 0
for seed in range(200):
    r = np.random.default_rng(seed)
    o = gc.gloss_contrast_scores(r.standard_normal((4, 16)), r.standard_normal((2, 4, 16)))
    same_dot += np.array_equal(np.argmax(o["glossmean"], 0), np.argmax(o["dot"], 0))
    same_base += np.array_equal(np.argmax(o["unitmean"], 0), np.argmax(o["baseline"], 0))
print(f"glossmean picks like the dot product in {same_dot}/200 random two-candidate cases")
print(f"unitmean picks like the baseline in {same_base}/200")
print("the two glossmean scores are negatives of each other:", np.allclose(o["glossmean"][0], -o["glossmean"][1], atol=1e-6))
""")

md(r"""
### 1.5 Centring, leaving a candidate out of its own mean, and a single candidate

- **The gloss centring cancels in `glossmean`.** $\tilde g_k - m = (g_k - \mu_g) - (\bar g - \mu_g) = g_k - \bar g$.
  The sentence centring $\mu_s$ still applies to $h$. `unitmean` normalises the centred vectors first, so its
  result depends on $\mu_g$.
- **Leave-one-out gives the same cosine.** Subtracting the mean of the *other* candidates instead,
  $\tilde g_k - \frac{1}{n-1}\sum_{j\ne k}\tilde g_j = \frac{n}{n-1}(\tilde g_k - m)$, only rescales the vector.
- **One candidate.** $\tilde g_1 - m = 0$, so the score is 0. Every target in the study has at least two.
""")

code(r"""
mu_s, mu_g = rng.standard_normal((2, 4, 16))
with_mu = gc.gloss_contrast_scores(h_r, g_r, mu_s, mu_g)
without_mu_g = gc.gloss_contrast_scores(h_r, g_r, mu_s, None)
print("glossmean unchanged by mu_g:", np.allclose(with_mu["glossmean"], without_mu_g["glossmean"], atol=1e-5))
print("unitmean changed by mu_g:   ", not np.allclose(with_mu["unitmean"], without_mu_g["unitmean"], atol=1e-3))

k, l = 2, 1
loo = g_r[k, l] - np.delete(g_r, k, axis=0)[:, l].mean(0)
print(f"leave-one-out cosine {cos(h_r[l], loo):.6f} = glossmean {out['glossmean'][k, l]:.6f}")
print("one candidate scores:", gc.gloss_contrast_scores(h_r, g_r[:1])["glossmean"][0])
""")

md(r"""
### 1.6 Partial removal

`alpha=a` subtracts only a fraction of the mean: $\cos(\tilde h, \tilde g_k - a\,m)$. At $a = 0$ it is exactly the
baseline, and at $a = 1$ exactly `glossmean`. The study tunes $a$ on dev. In the toy, the pick switches from B to A
partway along:
""")

code(r"""
alphas = np.round(np.arange(0, 1.01, 0.1), 2)
fam = gc.gloss_contrast_scores(h, g, alphas=alphas)
print("alpha=0 is the baseline:", np.array_equal(fam[gc.alpha_key(0)], fam["baseline"]))
print("alpha=1 is glossmean:   ", np.array_equal(fam[gc.alpha_key(1)], fam["glossmean"]))
print("toy pick by alpha:", "  ".join(f"{a:g}:{'ABC'[int(np.argmax(fam[gc.alpha_key(a)][:, 0]))]}" for a in alphas))
""")

# =====================================================================================================
md(r"""
---
## Part 2. Real examples (Llama-3.2-1B)

These use the vectors already extracted for the layer study. Scores use its frozen configuration, *echo_last ×
mean gloss, layers 5–7*, and its centring fitted on dev. All examples are from **dev** documents.
""")

code(r"""
import json

import layer_study as ls
from Processing.script import load_sample, open_stores
from Processing.semcor_data import load_dataset

ds = load_dataset()
sent_store, gloss_store = open_stores()
frozen = json.loads((REPO / "Report/layer_study/results.json").read_text())["dev_selection"]["winner"]
rep, band_name = frozen.split("/")
s_var, g_var = rep.split("+")
band = ev.contiguous_bands(17)[band_name]
print("frozen configuration:", frozen, "-> layers", band)

dev_ids = load_sample("dev_natural_80")["instance_ids"]
dev_all_ids = list(dict.fromkeys(dev_ids + load_sample("dev_balanced_200")["instance_ids"]))
dev_all = ls.eval_set(ds, dev_all_ids)
keys = sorted({k for c in dev_all.candidates for k in c})
gbank = ev.GlossBank(keys, gloss_store.get(keys, g_var))
centering = ev.fit_centering(sent_store.get(dev_all_ids, s_var), gbank, dev_all)  # the layer study's centring

es = ls.eval_set(ds, dev_ids)  # dev_natural
sent = sent_store.get(es.instance_ids, s_var)
scores = gc.gloss_contrast_layer_scores(sent, gbank, es, centering)
band_scores = {name: ev.config_scores(v, band) for name, v in scores.items()}
print(f"{len(es)} dev_natural instances scored; {np.mean(es.n_candidates):.1f} candidates on average")
""")

md(r"""
### 2.1 Cross-check against the layer study's own scorer

`baseline` must reproduce `evaluation.layer_scores`, the code behind the layer study's results.
""")

code(r"""
old = ev.layer_scores(sent, gbank, es, centering)
print("max |baseline - layer_scores|:", max(np.abs(a - b).max() for a, b in zip(scores["baseline"], old)))
""")

md(r"""
### 2.2 How much do a word's glosses share?

These are averages over dev_natural targets, taken over each target's candidates:

- $\cos(\tilde g_k, m)$: how closely each centred gloss vector points along the candidates' mean;
- $\lVert \tilde g_k - m\rVert / \lVert \tilde g_k\rVert$: the share of its length left after removing the mean;
- $\cos(\tilde h, m)$: how closely the target points along that mean.

If the shared part were small, removing it could change little.
""")

code(r"""
mu_g, mu_s = centering.mu_gloss.astype(np.float32), centering.mu_sentence.astype(np.float32)
cg, rr, ch = [], [], []
for i, cands in enumerate(es.candidates):
    gk = gbank.get(cands).astype(np.float32) - mu_g
    mk = gk.mean(0)
    cg.append((unit(gk) * unit(mk)).sum(-1).mean(0))
    rr.append((np.linalg.norm(gk - mk, axis=-1) / np.linalg.norm(gk, axis=-1)).mean(0))
    ch.append((unit(sent[i].astype(np.float32) - mu_s) * unit(mk)).sum(-1))
cg, rr, ch = np.mean(cg, 0), np.mean(rr, 0), np.mean(ch, 0)
print(f"{'layer':>5s}{'cos(g~, m)':>12s}{'|g~ - m|/|g~|':>15s}{'cos(h~, m)':>12s}")
for l in range(17):
    print(f"{l:5d}{cg[l]:12.3f}{rr[l]:15.3f}{ch[l]:12.3f}")
""")

md(r"""
### 2.3 One instance, candidate by candidate

Each row is a candidate sense, scored with the band mean over layers 5–7. `*` marks the gold sense and `<` marks
each score's pick. The examples are the first dev instance where removing the mean fixes the baseline's pick, and
the first where it breaks a correct one.
""")

code(r"""
NAMES = ("baseline", "glossmean", "unitmean", "dot")


def show(i):
    x = ds.instances[es.instance_ids[i]]
    print(f"{es.instance_ids[i]}  {x['word']!r} ({x['pos']}), {len(x['candidates'])} candidates")
    print(f"  {ds.sentences[x['sentence_id']]['text']}")
    picks = {m: int(np.argmax(band_scores[m][i])) for m in NAMES}
    print(f"  {'':3s}{'candidate':22s}" + "".join(f"{m:>10s} " for m in NAMES) + "  gloss")
    for k, syn in enumerate(x["candidates"]):
        gold = "*" if k in x["gold_indices"] else " "
        row = "".join(f"{band_scores[m][i][k]:10.3f}{'<' if picks[m] == k else ' '}" for m in NAMES)
        print(f"  {gold}  {syn:22s}{row}  {ds.synsets[syn]['gloss'][:55]}")
    print()


base_pick = np.array([int(np.argmax(s)) for s in band_scores["baseline"]])
gm_pick = np.array([int(np.argmax(s)) for s in band_scores["glossmean"]])
right = np.array([p in g for p, g in zip(base_pick, es.gold)])
gm_right = np.array([p in g for p, g in zip(gm_pick, es.gold)])
show(int(np.flatnonzero(~right & gm_right)[0]))
show(int(np.flatnonzero(right & ~gm_right)[0]))
""")

md(r"""
### 2.4 How often the pick changes on dev

At the frozen configuration, on dev_natural. The study reports the same counts for every sample.
""")

code(r"""
print(f"glossmean changes the baseline's pick for {np.mean(gm_pick != base_pick):.1%} of {len(es)} instances: "
      f"{int((~right & gm_right).sum())} fixed, {int((right & ~gm_right).sum())} broken")
for m in ("unitmean", "dot"):
    p = np.array([int(np.argmax(s)) for s in band_scores[m]])
    ok = np.array([q in g for q, g in zip(p, es.gold)])
    print(f"{m:9s} changes it for {np.mean(p != base_pick):.1%}: {int((~right & ok).sum())} fixed, {int((right & ~ok).sum())} broken")
two = es.n_candidates == 2
dot_pick = np.array([int(np.argmax(s)) for s in band_scores["dot"]])
print(f"\ntwo-candidate targets: glossmean picks like the dot product for {np.mean(gm_pick[two] == dot_pick[two]):.0%} of {two.sum()}")
""")

md(r"""
---
## Where the experiment lives

- `Analysis/gloss_mean_study.py` runs the protocol. It searches layers and representations on dev for each score,
  validates on held-out test documents, and runs the secondary analyses (each score at the baseline's layers, and
  partial removal).
- `Report/GLOSS_MEAN_STUDY.md` has the results.
- `tests/test_gloss_contrast.py` repeats the checks in this notebook as unit tests:
  `.venv/bin/python -m tests.run_tests gloss_contrast`.
""")


def build(path: Path) -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [nbf.v4.new_markdown_cell(t) if kind == "md" else nbf.v4.new_code_cell(t) for kind, t in CELLS]
    nbf.write(nb, path)
    print(f"wrote {path} ({len(CELLS)} cells)")


if __name__ == "__main__":
    build(Path(__file__).resolve().parent / "gloss_mean_methods.ipynb")
