"""Writes Analysis/swap_methods.ipynb (the worked-examples notebook for the swap study).

The notebook is generated so that its text and code stay in one reviewable file. Regenerate and execute with:
    .venv/bin/python Analysis/build_swap_notebook.py
    .venv/bin/jupyter nbconvert --to notebook --execute --inplace Analysis/swap_methods.ipynb
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
# Swapped-word contrasts: every function, with worked examples

**The idea.** The cosine between a target word's hidden state and a gloss vector may mix two things: how close
the *meanings* are, and how similar the *grammatical* signals are (part of speech, syntactic role, position). To
separate them, each sentence is run twice. The second run swaps the target for a random word of the same part
of speech. That keeps the grammatical slot but changes the meaning, so contrasting the two runs should cancel
the grammar. Three ways of contrasting them are tested:

1. **Subtract the activations, then take the cosine** with the gloss vector.
2. **Normalise both activations, subtract, then take the cosine.**
3. **Take the cosine of each with the gloss vector, then subtract** the two numbers.

This notebook documents each function that implements this. Every function comes with examples small enough to
check by hand, the identities that link the three methods, and a few real examples from Llama-3.2-1B.

| Code | What it does |
|---|---|
| `Processing/swaps.py` | builds the swapped sentences (Part 1) |
| `Processing/contrastive.py` | computes the three scores (Part 2) |
| `Analysis/swap_study.py` | runs the experiment; results are in `Report/SWAP_STUDY.md` |
| `tests/test_swaps.py`, `tests/test_contrastive.py` | the same checks as unit tests |

Outputs below were produced by executing the notebook top to bottom.
""")

code(r"""
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "Processing").is_dir())
for p in (REPO, REPO / "Analysis"):
    sys.path.insert(0, str(p))

from Processing import contrastive as ct
from Processing import evaluation as ev
from Processing import swaps as sw
from Processing.semcor_data import load_dataset

np.set_printoptions(precision=4, suppress=True)
ds = load_dataset()
print(f"{len(ds.instances):,} sense-tagged words in {len(ds.sentences):,} sentences")
""")

# =====================================================================================================
md(r"""
---
## Part 1. Building the swapped sentences (`Processing/swaps.py`)

### 1.1 `synset_pos`, `needed_pos`: which swaps an instance needs

Every candidate sense should be compared against a slot holding a word of *its own* part of speech. So an instance
needs one swapped sentence per distinct part of speech among its candidates. `synset_pos` reads the POS from a
WordNet synset name and folds adjective satellites (`s`) into adjectives (`a`). `needed_pos` lists the distinct
parts of speech.

In the v2 dataset, candidates are restricted to the target's own part of speech, so every instance needs exactly
one swap. The mixed case below is what the old, pre-v2 data would have needed.
""")

code(r"""
print(sw.synset_pos("long.a.01"), sw.synset_pos("long.s.03"), sw.synset_pos("st._john's_wort.n.01"))
inst = ds.instances["d000.s000.t000"]
print(f"{inst['word']!r} ({inst['pos']}): candidates {inst['candidates']}")
print("needed:", sw.needed_pos(inst["candidates"]))
print("a mixed candidate list needs:", sw.needed_pos(["long.a.01", "long.v.01", "long.r.01", "long.s.03"]))
print("v2 instances needing more than one swap:",
      sum(len(sw.needed_pos(x["candidates"])) > 1 for x in ds.instances.values()))
""")

md(r"""
### 1.2 `usable_form`, `build_pool`: the replacement words

The pool is every distinct surface form of a sense-tagged word in the dataset, grouped by part of speech. Only
lower-case alphabetic single words are kept, with hyphens and apostrophes allowed inside. This keeps out names
("Fulton"), numbers and multiword expressions ("New York"). The pool also records every form seen for each
(lemma, POS), so that a target is never replaced by a form of its own lemma.
""")

code(r"""
pool = sw.build_pool(ds.instances.values())
print("pool sizes:", pool.sizes())
for pos in "nvar":
    words = pool.by_pos[pos]
    print(f"  {pos}: {words[1000:1006]} ... {words[-3:]}")
print({w: sw.usable_form(w) for w in ["well-known", "o'clock", "Fulton", "New York", "3", "'s", "a"]})
""")

md(r"""
### 1.3 `excluded_forms`, `draw_replacement`: the random draw

`draw_replacement` draws **uniformly over distinct forms**: every allowed word is equally likely, however often it
occurs. It uses rejection sampling, drawing from the whole list and redrawing when it hits an excluded form. That
gives each allowed form probability $1/(\text{pool size} - \text{excluded})$.

Below:
- the excluded forms for the target *objectives*;
- ten draws for it;
- a frequency check on a four-word pool, where the target "cat" must never come out and the other three should
  each come out about a third of the time.
""")

code(r"""
inst = ds.instances["d000.s000.t003"]
print(f"target {inst['word']!r}, lemma {inst['lemma']!r}: excluded {sorted(sw.excluded_forms(pool, inst['word'], inst['lemma'], 'n'))}")
rng = random.Random(0)
print("ten draws:", [sw.draw_replacement(pool, inst["word"], inst["lemma"], "n", rng) for _ in range(10)])

toy = sw.build_pool([{"word": w, "lemma": w, "pos": "n"} for w in ["cat", "dog", "hen", "owl"]])
rng = random.Random(1)
counts = Counter(sw.draw_replacement(toy, "cat", "cat", "n", rng) for _ in range(30_000))
print("30,000 draws for 'cat' from {cat, dog, hen, owl}:", dict(sorted(counts.items())), "(expected 10,000 each)")
""")

md(r"""
### 1.4 `word_form`: inflection classes (only for the `pos+form` variant)

The design as specified matches only the part of speech, so a past-tense verb can be replaced by an *-ing* form
("since you **reviewed** the" → "since you **dangling** the"). The secondary variant `match="pos+form"` also
matches a coarse inflection class, read off how the word differs from its lemma. It's a heuristic: for example,
*was* counts as an *-s* form. When a class has fewer than 50 forms, the draw falls back to the whole
part-of-speech pool.
""")

code(r"""
for word, lemma, pos in [("objectives", "objective", "n"), ("program", "program", "n"), ("reviewed", "review", "v"),
                         ("reviewing", "review", "v"), ("reviews", "review", "v"), ("took", "take", "v"),
                         ("longer", "long", "a"), ("longest", "long", "a"), ("worse", "bad", "a")]:
    print(f"{word:>11} ({lemma}, {pos}) -> {sw.word_form(word, lemma, pos)}")
print({f"{p}/{f}": len(v) for (p, f), v in sorted(pool.by_pos_form.items())})
""")

md(r"""
### 1.5 `match_case`, `replace_span`: editing the text

The replacement copies the target's capitalisation. `replace_span` swaps characters `[start, end)` and returns
the new word's span. `make_swap` adds a space when the target is written against a neighbouring word. That
happens with the clitics SemCor tags separately, such as *'s* (be) in *It's*.
""")

code(r"""
print(sw.match_case("ripe", "How"), sw.match_case("toad", "NASA"), sw.match_case("toad", "how"))
text, span = sw.replace_span("How long has it been", (4, 8), "ripe")
print(repr(text), span, repr(text[span[0]:span[1]]))
clitic = sw.make_swap(ds, pool, "d191.s074.t000", "v", 0)
print(f"{ds.sentences[clitic.sentence_id]['text'][:40]!r} -> {clitic.text[:40]!r}")
""")

md(r"""
### 1.6 `make_swap`, `plan_swaps`: one swapped sentence per target and draw

Each draw is seeded by `(seed, instance id, POS, draw, match mode)` alone. So the same target always gets the same
word, whatever else is in the run, and adding draws never changes earlier ones. Below is a dev sentence with every
target swapped in turn. Only one word changes per swapped sentence. The swaps match `plan.json`, the record
written next to the vectors that were actually extracted.
""")

code(r"""
from Processing.script import load_sample, open_swap_store

dev_ids = load_sample("dev_natural_80")["instance_ids"]
sid = ds.instances[dev_ids[0]]["sentence_id"]
ids = [i for i in dev_ids if ds.instances[i]["sentence_id"] == sid]
print("original:", ds.sentences[sid]["text"], "\n")
plan = sw.plan_swaps(ds, ids, pool=pool)
for s in plan:
    print(f"{s.original!r:>16} -> {s.replacement!r:<16} {s.text}")

swap_store = open_swap_store()
recorded = {s.key: s for s in sw.load_plan(swap_store.root / "plan.json")}
print("\nidentical to the recorded plan:", all(recorded[s.key] == s for s in plan))
print("pos+form draws for the same targets:",
      [s.replacement for s in sw.plan_swaps(ds, ids, pool=pool, match="pos+form")])
""")

md(r"""
### 1.7 `swap_inputs`: where the vectors are read

Swapped sentences go through the same echo prompt as the originals: `[BOS] text text`. The vector is read from
the tokens of the replacement:

- **`plain_last`**: its last sub-token in copy 1, which sees only the left context;
- **`echo_last`**: its last sub-token in copy 2, which has read the whole sentence once;
- **`echo_mean`**: the mean over its sub-tokens in copy 2.

Brackets mark those tokens below. The tokens before the replacement in copy 1 are identical to the original
sentence's. Llama is causal, so the hidden states there are identical too.
""")

code(r"""
from Processing.script import build_inputs, load_tokenizer

tok = load_tokenizer()
for s in (plan[3], plan[7]):  # a one-token and a multi-token replacement
    x = sw.swap_inputs(tok, [s])[0]
    (pf, pl), (ef, el) = x.targets[s.key]["plain"], x.targets[s.key]["echo"]
    pieces = [tok.decode([t]) for t in x.token_ids]
    print("".join(f"[{p}]" if pf <= i <= pl or ef <= i <= el else p for i, p in enumerate(pieces)))
    print(f"plain_last reads token {pl} {pieces[pl]!r}; echo_last token {el} {pieces[el]!r}; "
          f"echo_mean averages tokens {ef}-{el} {pieces[ef:el + 1]}")
    orig = build_inputs(ds, tok, [s.instance_id])[0][0]
    print("tokens before the target identical to the original sentence:", orig.token_ids[:pf] == x.token_ids[:pf], "\n")
""")

# =====================================================================================================
md(r"""
---
## Part 2. The scores (`Processing/contrastive.py`)

### 2.1 Notation

For one target at one layer:

| Symbol | Meaning |
|---|---|
| $h$ | the target's vector in the original sentence |
| $s$ | the vector at the same slot in the swapped sentence |
| $g_k$ | the gloss vector of candidate sense $k$ |
| $\mu_s,\ \mu_g$ | mean sentence and gloss vectors on dev; the layer study's centring |
| $\tilde x = x - \mu$ | centred vector |
| $\hat x = \tilde x / \lVert\tilde x\rVert$ | centred, unit-length vector (`unit`) |
| $\cos(a,b) = \langle a,b\rangle / (\lVert a\rVert\,\lVert b\rVert)$ | cosine similarity |

The scores, with the highest score predicted:

| Score | Formula | Code name |
|---|---|---|
| baseline (layer study) | $\cos(\tilde h, \tilde g_k)$ | `baseline` |
| method 1 | $\cos(\tilde h - \tilde s,\ \tilde g_k)$ | `diff` |
| method 2 | $\cos(\hat h - \hat s,\ \tilde g_k)$ | `normdiff` |
| method 3 | $\cos(\tilde h, \tilde g_k) - \cos(\tilde s, \tilde g_k)$ | `scorediff` |
| diagnostic | $\cos(\tilde s, \tilde g_k)$: the slot holding a random word, alone | `swap` |

**Similarity or distance.** Cosine *distance* is $1-\cos$. Ranking by smallest distance gives exactly the same
order as ranking by largest similarity, and that holds for method 3 too:
$(1-\cos_h) - (1-\cos_s) = -(\cos_h - \cos_s)$. The code uses similarity throughout.

**Order of subtraction.** It is always original minus swapped. Swapping the order flips the sign of every score,
which reverses the ranking.

`unit(x)` divides by the length and leaves a zero vector at zero, so a zero difference gives scores of 0 rather
than NaN.
""")

code(r"""
print(ct.unit(np.array([3.0, 4.0])), ct.unit(np.zeros(2)), np.linalg.norm(ct.unit(np.random.randn(5, 7)), axis=-1))
""")

md(r"""
### 2.2 A toy example worked by hand

Three dimensions: **(sense A, sense B, grammar)**. The target means sense A and carries a strong grammatical
component. The random word shares the grammar and has a little sense B. Gloss B happens to carry more of the
grammatical direction than gloss A.

$$h = (1,\ 0,\ 3),\quad s = (0,\ 0.5,\ 3),\quad g_A = (1,\ 0,\ 0.5),\quad g_B = (0,\ 1,\ 2)$$

The lengths are $\lVert h\rVert^2 = 10$, $\lVert s\rVert^2 = 9.25$, $\lVert g_A\rVert^2 = 1.25$ and $\lVert g_B\rVert^2 = 5$. There is no centring in the toy.

| | candidate A | candidate B | pick |
|---|---|---|---|
| **baseline** $\cos(h,g)$ | $2.5/\sqrt{12.5} = 0.7071$ | $6/\sqrt{50} = 0.8485$ | **B** (fooled by grammar) |
| **method 1**: $d = h - s = (1, -0.5, 0)$, $\lVert d\rVert^2 = 1.25$ | $1/\sqrt{1.25 \cdot 1.25} = 0.8$ | $-0.5/\sqrt{1.25 \cdot 5} = -0.2$ | **A** |
| $\cos(s,g)$ | $1.5/\sqrt{11.5625} = 0.4411$ | $6.5/\sqrt{46.25} = 0.9558$ | |
| **method 3**: $\cos(h,g)-\cos(s,g)$ | $0.7071 - 0.4411 = 0.2660$ | $0.8485 - 0.9558 = -0.1073$ | **A** |
| **method 2**: $d = \hat h - \hat s = (0.3162,\ -0.1644,\ -0.0377)$, $\lVert d\rVert = 0.3584$ | $0.2974/(0.3584 \cdot 1.1180) = 0.7421$ | $-0.2398/(0.3584 \cdot 2.2361) = -0.2993$ | **A** |

The code below computes the same numbers with `contrastive_scores`. The toy has a single layer and one draw, so
`h` and `s` have shape `(L=1, D=3)` and `g` has shape `(n=2, L=1, D=3)`.
""")

code(r"""
h = np.array([[1.0, 0.0, 3.0]])
s = np.array([[0.0, 0.5, 3.0]])
g = np.array([[[1.0, 0.0, 0.5]], [[0.0, 1.0, 2.0]]])
toy_scores = ct.contrastive_scores(h, s, g)
print(f"{'':10s}{'A':>9s}{'B':>9s}  pick")
for m, v in toy_scores.items():
    print(f"{m:10s}{v[0, 0]:9.4f}{v[1, 0]:9.4f}  {'AB'[int(np.argmax(v[:, 0]))]}")
""")

md(r"""
### 2.3 Identity 1: methods 2 and 3 always rank candidates the same way, at any single layer

Write $d = \hat h - \hat s$. Then

$$\text{method 3} = \langle \hat h, \hat g_k\rangle - \langle \hat s, \hat g_k\rangle = \langle d, \hat g_k\rangle,
\qquad \text{method 2} = \cos(d, \tilde g_k) = \frac{\langle d, \hat g_k\rangle}{\lVert d\rVert}.$$

$\lVert d\rVert$ is the same for every candidate of an instance, so **method 2 = method 3 / $\lVert d\rVert$**.
That is a positive rescaling, which never changes the order. In the toy, $0.2660 / 0.7421 = 0.3584 =
\lVert d\rVert$.

The check below uses random 2,048-dimensional vectors with 17 layers and 8 candidates, centred as in the study.
""")

code(r"""
rng = np.random.default_rng(0)
L, D, n = 17, 2048, 8
h, s, g = rng.standard_normal((L, D)), rng.standard_normal((L, D)), rng.standard_normal((n, L, D))
mu_s, mu_g = rng.standard_normal((L, D)), rng.standard_normal((L, D))
out = ct.contrastive_scores(h, s, g, mu_s, mu_g)
d = ct.unit(h - mu_s) - ct.unit(s - mu_s)
print("toy: method 3 / method 2 =", toy_scores["scorediff"][:, 0] / toy_scores["normdiff"][:, 0])
print("max |method 3 - |d| * method 2| over 17 layers x 8 candidates:",
      np.abs(out["scorediff"] - np.linalg.norm(d, axis=-1) * out["normdiff"]).max())
print("same candidate order at every layer:",
      (np.argsort(-out["scorediff"], axis=0) == np.argsort(-out["normdiff"], axis=0)).all())
""")

md(r"""
### 2.4 Identity 2: method 1 is method 3 with each cosine weighted by its vector's length

$$\cos(\tilde h - \tilde s,\ \tilde g_k) = \frac{\langle \tilde h - \tilde s, \hat g_k\rangle}{\lVert\tilde h - \tilde s\rVert}
= \frac{\lVert\tilde h\rVert\cos(\tilde h,\tilde g_k) - \lVert\tilde s\rVert\cos(\tilde s,\tilde g_k)}{\lVert\tilde h - \tilde s\rVert}.$$

The denominator is the same for every candidate. So method 1 ranks like
$\cos(\tilde h,\tilde g_k) - \beta_i \cos(\tilde s,\tilde g_k)$ with $\beta_i = \lVert\tilde s\rVert / \lVert\tilde h\rVert$,
which is method 3 when the two vectors have equal length. In the toy, $\sqrt{10}\cdot 0.7071 - \sqrt{9.25}\cdot
0.4411 = 0.8944$, and $0.8944 / \lVert d\rVert = 0.8944/1.1180 = 0.8$.
""")

code(r"""
cos = lambda a, b: (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
hc, sc, gc = h - mu_s, s - mu_s, g - mu_g
weighted = np.linalg.norm(hc, axis=-1) * cos(hc[None], gc) - np.linalg.norm(sc, axis=-1) * cos(sc[None], gc)
print("max |method 1 - weighted / |h~ - s~||:", np.abs(out["diff"] - weighted / np.linalg.norm(hc - sc, axis=-1)).max())
beta_i = np.linalg.norm(sc, axis=-1) / np.linalg.norm(hc, axis=-1)
family = out["baseline"] - beta_i * out["swap"]
print("method 1 ranks like baseline - beta_i * swap:",
      (np.argsort(-family, axis=0) == np.argsort(-out["diff"], axis=0)).all(), "| beta_i at layers 0-3:", beta_i[:4])
""")

md(r"""
### 2.5 Centring: it cancels in method 1, not in methods 2 and 3

Every score centres the sentence-side vectors with the same dev mean $\mu_s$ as the baseline, and centres the
glosses with $\mu_g$. Llama's hidden states share a large common component (anisotropy), so uncentred cosines
are all high and barely discriminate. Centring removes that common part.

In method 1, $\tilde h - \tilde s = (h-\mu_s) - (s-\mu_s) = h - s$, so $\mu_s$ drops out. Subtracting the
swapped run removes the common component by itself. In methods 2 and 3, $\mu_s$ changes the unit vectors
$\hat h$ and $\hat s$, so it matters.
""")

code(r"""
plain = ct.contrastive_scores(h, s, g, None, mu_g)
for m in ("diff", "normdiff", "scorediff"):
    reordered = int((np.argsort(-out[m], axis=0) != np.argsort(-plain[m], axis=0)).any(axis=0).sum())
    print(f"{m:9s} max change from dropping mu_s: {np.abs(out[m] - plain[m]).max():.2e}; "
          f"candidate order changes at {reordered} of 17 layers")
""")

md(r"""
### 2.6 Bands of layers: where methods 2 and 3 *can* differ

A configuration such as "layers 5–7" averages the **per-layer scores** (`evaluation.config_scores`), exactly as in
the layer study. From 2.3, method 3 at layer $l$ is $\lVert d_l\rVert$ times method 2. So over a band, method 3
weights each layer by $\lVert d_l\rVert$ and method 2 weights them equally.

The example below has two layers in two dimensions, chosen to make them disagree:

- **Layer 1:** $h$ and $s$ point almost the same way, so $d_1$ is short ($\lVert d_1\rVert = 0.1$). It points at
  candidate B (cosines A = −1, B = +1).
- **Layer 2:** $h$ and $s$ are far apart, so $d_2$ is long ($\lVert d_2\rVert = 1.41$). It mildly favours A
  (cosines A = 0.5, B = 0).

The result:

- **Method 2:** A $= (-1 + 0.5)/2 = -0.25$ and B $= (1 + 0)/2 = 0.5$, so it picks **B**.
- **Method 3:** A $= (-0.1 + 0.71)/2 = 0.30$ and B $= (0.1 + 0)/2 = 0.05$, so it picks **A**.

The study reports how often this happens on real data. It is rare, because neighbouring layers have similar
$\lVert d_l\rVert$.
""")

code(r"""
def at(angle):
    return np.array([np.cos(angle), np.sin(angle)])

h1, s1 = at(np.pi / 2 - 0.05), at(np.pi / 2 + 0.05)  # d1 = h1 - s1 = (2 sin 0.05, 0), length 0.1
h2, s2 = at(0.0), at(-np.pi / 2)  # d2 = (1, 1), length sqrt(2)
d2_angle = np.arctan2(*(h2 - s2)[::-1])
g_A = [at(np.pi), at(d2_angle + np.pi / 3)]  # layer 1: opposite d1; layer 2: 60 degrees from d2
g_B = [at(0.0), at(d2_angle + np.pi / 2)]  # layer 1: along d1; layer 2: perpendicular to d2
band = ct.contrastive_scores(np.stack([h1, h2]), np.stack([s1, s2]), np.array([g_A, g_B]))
for m in ("normdiff", "scorediff"):
    per_layer, mean = band[m], band[m].mean(axis=1)
    print(f"{m:9s} layer 1 (A, B) = {per_layer[:, 0]}, layer 2 = {per_layer[:, 1]}, band mean = {mean} -> {'AB'[int(np.argmax(mean))]}")
""")

md(r"""
### 2.7 Edge case: identical vectors

If $\tilde h = \tilde s$ at a layer, then $d = 0$ and every candidate scores exactly 0 there. Scores at other
layers in the band still decide. If every layer of the band ties, `evaluation.instance_metrics` breaks the tie
uniformly at random, so the expected accuracy is $1/n$.

On real data this can only happen at layer 0, the embeddings, with a `last`-sub-token vector, when the target and
the replacement end in the same sub-token. Section 3.3 counts it for the dev examples.
""")

code(r"""
same = ct.contrastive_scores(h, s=h, g=g)
print({m: float(np.abs(same[m]).max()) for m in ("diff", "normdiff", "scorediff")})
tie_es = ev.EvalSet(["i"], ["c"], [["a", "b", "c", "d"]], [[0]], ["n"])
picks = [ev.instance_metrics([np.zeros(4)], tie_es, seed=k)["acc1"][0] for k in range(4000)]
print("accuracy over 4,000 tie-breaks of a 4-way tie:", np.mean(picks), "(expected 0.25)")
""")

md(r"""
### 2.8 Several draws, and the β family

With $K$ swapped sentences, each method uses the average swapped side:

| Method | With $K$ draws |
|---|---|
| method 1 | $\cos(\tilde h - \overline{\tilde s},\ \tilde g_k)$, subtracting the mean vector |
| method 2 | $\cos(\hat h - \overline{\hat s},\ \tilde g_k)$, subtracting the mean of the unit vectors |
| method 3 | $\cos(\tilde h,\tilde g_k) - \overline{\cos(\tilde s_j,\tilde g_k)}$, subtracting the mean cosine |

Identity 1 still holds with $d = \hat h - \overline{\hat s}$.

The **β family** $\cos(\tilde h,\tilde g_k) - \beta\cos(\tilde s,\tilde g_k)$ (`with_beta`) links the three:

| β | Equivalent to |
|---|---|
| 0 | the baseline |
| 1 | method 3, and method 2's ranking |
| $\lVert\tilde s\rVert/\lVert\tilde h\rVert$ per instance | method 1's ranking (2.4) |

A negative β *adds* the swapped slot's similarity, which would help if the context carries sense information.
The study tunes β on dev as a secondary analysis.
""")

code(r"""
s3 = rng.standard_normal((3, L, D))
three = ct.contrastive_scores(h, s3, g, mu_s, mu_g)
ones = [ct.contrastive_scores(h, s3[j], g, mu_s, mu_g) for j in range(3)]
print("K=3 swap = mean of the three single-draw swaps:", np.allclose(three["swap"], np.mean([o["swap"] for o in ones], 0), atol=1e-6))
print("K=3 method 1 = method 1 against the mean vector:",
      np.allclose(three["diff"], ct.contrastive_scores(h, s3.mean(0), g, mu_s, mu_g)["diff"], atol=1e-6))
pl = {m: [v] for m, v in toy_scores.items()}
for beta in (0.0, 0.5, 1.0):
    print(f"toy, beta={beta}: {ct.with_beta(pl, beta)[0][:, 0]}")
""")

md(r"""
### 2.9 `contrastive_layer_scores`: a whole evaluation set, including mixed parts of speech

This applies `contrastive_scores` to every instance of an `EvalSet` and returns `{method: [(n_i, L) per
instance]}`, the format the layer study's band search consumes. For v2 data the swapped vectors are one array
`(N, K, L, D)`.

For candidate lists that mix parts of speech, pass one `{POS: (K, L, D)}` dict per instance plus each candidate's
POS. Each candidate is then scored against the swaps made for its own POS. In the toy below, the verb candidate
must get exactly the scores it would get alone with the verb swaps.
""")

code(r"""
names = ["bank|bank.n.01", "bank|bank.v.01", "bank|bank.n.02"]
bank = ev.GlossBank(names, rng.standard_normal((3, 4, 8)))
es_toy = ev.EvalSet(["i0"], ["c0"], [names], [[0]], ["n"])
orig = rng.standard_normal((1, 4, 8))
s_noun, s_verb = rng.standard_normal((1, 4, 8)), rng.standard_normal((1, 4, 8))
mixed = ct.contrastive_layer_scores(orig, [{"n": s_noun, "v": s_verb}], bank, es_toy, None, cand_pos=[["n", "v", "n"]])
alone = ct.contrastive_scores(orig[0], s_verb, bank.get(["bank|bank.v.01"]))
print("verb candidate, scored with the verb swaps:", np.allclose(mixed["scorediff"][0][1], alone["scorediff"][0]))
""")

# =====================================================================================================
md(r"""
---
## Part 3. Real examples (Llama-3.2-1B)

These use the vectors extracted for the study. The original-sentence and gloss vectors come from the layer
study, and the swapped-sentence vectors come from draw 0 of the POS-matched swaps. Scores use the layer study's
frozen configuration, *echo_last × mean gloss, layers 5–7*, and its centring fitted on dev. All examples are from
**dev** documents.
""")

code(r"""
import json

import layer_study as ls
from Processing.script import open_stores

sent_store, gloss_store = open_stores()
frozen = json.loads((REPO / "Report/layer_study/results.json").read_text())["dev_selection"]["winner"]
rep, band_name = frozen.split("/")
s_var, g_var = rep.split("+")
band = ev.contiguous_bands(17)[band_name]
print("frozen configuration:", frozen, "-> layers", band)

dev_all_ids = list(dict.fromkeys(load_sample("dev_natural_80")["instance_ids"] + load_sample("dev_balanced_200")["instance_ids"]))
dev_es = ls.eval_set(ds, dev_all_ids)
keys = sorted({k for c in dev_es.candidates for k in c})
gbank = ev.GlossBank(keys, gloss_store.get(keys, g_var))
centering = ev.fit_centering(sent_store.get(dev_all_ids, s_var), gbank, dev_es)  # the layer study's centring

es = ls.eval_set(ds, dev_ids)  # dev_natural
orig = sent_store.get(es.instance_ids, s_var)
swapv = swap_store.get([sw.swap_key(i, p, 0) for i, p in zip(es.instance_ids, es.pos)], s_var)
scores = ct.contrastive_layer_scores(orig, swapv, gbank, es, centering)
band_scores = {m: ev.config_scores(v, band) for m, v in scores.items()}
print(f"{len(es)} dev_natural instances scored")
""")

md(r"""
### 3.1 Cross-check against the layer study's own scorer

Method 3 must equal *the layer study's cosine on the original sentence minus the same cosine on the swapped
sentence*, computed by `evaluation.layer_scores`, the code behind the earlier results. Identity 1 must also hold
on real vectors.
""")

code(r"""
base_old = ev.layer_scores(orig, gbank, es, centering)
swap_old = ev.layer_scores(swapv, gbank, es, centering)
print("max |baseline - layer_scores(original)|:", max(np.abs(a - b).max() for a, b in zip(scores["baseline"], base_old)))
print("max |method 3 - (layer_scores(original) - layer_scores(swapped))|:",
      max(np.abs(m3 - (a - b)).max() for m3, a, b in zip(scores["scorediff"], base_old, swap_old)))
mu = centering.mu_sentence
d_norm = np.linalg.norm(ct.unit(orig.astype(np.float32) - mu) - ct.unit(swapv.astype(np.float32) - mu), axis=-1)  # (N, L)
print("max |method 3 - |d| * method 2| over all instances and layers:",
      max(np.abs(m3 - dn[None] * m2).max() for m3, m2, dn in zip(scores["scorediff"], scores["normdiff"], d_norm)))
""")

md(r"""
### 3.2 One instance, candidate by candidate

Each row is a candidate sense, scored with the band mean over layers 5–7. `*` marks the gold sense and `<` marks
each method's pick. The first example is the first dev instance where method 3 and the baseline disagree. The
second is the first where all methods agree.
""")

code(r"""
def show(i):
    iid = es.instance_ids[i]
    x = ds.instances[iid]
    swap = sw.make_swap(ds, pool, iid, x["pos"], 0)
    print(f"{iid}  {x['word']!r} ({x['pos']})")
    print(f"  original: {ds.sentences[x['sentence_id']]['text']}")
    print(f"  swapped : {swap.text}   ({swap.original!r} -> {swap.replacement!r})")
    picks = {m: int(np.argmax(v[i])) for m, v in band_scores.items()}
    header = "".join(f"{name:>8s} " for name in ("baseline", "swap", "m1 diff", "m2 norm", "m3 score"))
    print(f"  {'':3s}{'candidate':22s}{header}  gloss")
    for k, syn in enumerate(x["candidates"]):
        gold = "*" if k in x["gold_indices"] else " "
        row = "".join(f"{band_scores[m][i][k]:8.3f}{'<' if picks[m] == k else ' '}"
                      for m in ("baseline", "swap", "diff", "normdiff", "scorediff"))
        print(f"  {gold}  {syn:22s}{row}  {ds.synsets[syn]['gloss'][:60]}")
    print()

disagree = [i for i in range(len(es)) if np.argmax(band_scores["baseline"][i]) != np.argmax(band_scores["scorediff"][i])]
agree = [i for i in range(len(es)) if len({int(np.argmax(band_scores[m][i])) for m in ("baseline", "diff", "normdiff", "scorediff")}) == 1]
print(f"methods 3 and baseline pick different senses for {len(disagree)} of {len(es)} dev_natural instances\n")
show(disagree[0])
show(agree[0])
""")

md(r"""
### 3.3 How different is the swapped slot?

These are per-layer averages over the dev_natural instances of:

- the raw cosine between $h$ and $s$;
- the centred cosine between $\tilde h$ and $\tilde s$;
- the length ratio $\lVert\tilde s\rVert/\lVert\tilde h\rVert$.

If the slot (grammar, position, context) dominated the target's vector, $h$ and $s$ would be close. Method 1's
implicit $\beta_i$ from 2.4 is the length ratio.
""")

code(r"""
h32, s32 = orig.astype(np.float32), swapv.astype(np.float32)
raw = (ct.unit(h32) * ct.unit(s32)).sum(-1).mean(0)
cen = (ct.unit(h32 - mu) * ct.unit(s32 - mu)).sum(-1).mean(0)
ratio = (np.linalg.norm(s32 - mu, axis=-1) / np.linalg.norm(h32 - mu, axis=-1)).mean(0)
print(f"{'layer':>5s}{'cos raw':>9s}{'cos centred':>13s}{'|s~|/|h~|':>11s}")
for l in range(17):
    print(f"{l:5d}{raw[l]:9.3f}{cen[l]:13.3f}{ratio[l]:11.3f}")
zero0 = np.mean(np.all(orig[:, 0] == swapv[:, 0], axis=-1))
print(f"\nlayer 0: target and replacement share their last sub-token (d = 0) for {zero0:.1%} of instances")
""")

md(r"""
---
## Where the experiment lives

- `Analysis/swap_study.py` runs the protocol. It searches layers and representations on dev for each method,
  validates on held-out test documents, and runs the secondary analyses (inflection-matched swaps, two draws, the
  β family).
- `Report/SWAP_STUDY.md` has the results.
- `tests/test_swaps.py` and `tests/test_contrastive.py` repeat the checks in this notebook as unit tests:
  `.venv/bin/python -m tests.run_tests swaps contrastive`.
""")


def build(path: Path) -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [nbf.v4.new_markdown_cell(t) if kind == "md" else nbf.v4.new_code_cell(t) for kind, t in CELLS]
    nbf.write(nb, path)
    print(f"wrote {path} ({len(CELLS)} cells)")


if __name__ == "__main__":
    build(Path(__file__).resolve().parent / "swap_methods.ipynb")
