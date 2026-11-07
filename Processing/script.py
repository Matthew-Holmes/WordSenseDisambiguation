#!/usr/bin/env python3
"""v2 activation pipeline: sample SemCor instances, then extract target and gloss vectors with JAX Llama.

Run from the repository root (in WSL, with the project venv):

    # 1. build the v2 dataset once (POS-restricted candidates, document splits, clean text)
    python -m Processing.build_dataset

    # 2. draw samples (document-level dev/test splits; "natural" = whole sentences, "balanced" = sense-balanced)
    python -m Processing.script sample --split dev --kind natural --n 80
    python -m Processing.script sample --split test --kind balanced --n 200

    # 3. see what extraction will cost, then run it (resumable; vectors are cached per instance / gloss)
    python -m Processing.script estimate --samples dev_natural_80 test_balanced_200
    python -m Processing.script extract  --samples dev_natural_80 test_balanced_200

    # 4. optional: swapped sentences (target replaced by a random word of the same POS; Processing/swaps.py).
    #    Each spec is MATCH:DRAWS with MATCH = pos | pos+form and DRAWS = 0, 1 or a range such as 0-2;
    #    specs run in the order given, so the first one finishes first.
    python -m Processing.script extract --samples dev_natural_80 --swaps pos:0 pos+form:0 pos:1

Vectors land in Data/Processed/v2/activations/{sentences,glosses}/ (see Processing/extract.py), keyed by
instance id (variants plain_last, echo_last, echo_mean) and by "lemma|synset" (variants mean, last).
Swapped-sentence vectors land in activations/swapped_seed<S>/, keyed by Processing.swaps.swap_key, with the
swapped texts in plan.json beside them. Each is a (17, 2048) float16 array: index 0 = embeddings, index k =
output of block k.

This replaces the March 2025 chunk script; the fixes are listed in Report/REPORT.md (B1-B16).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import warnings
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from Processing.prompts import build_gloss_input, build_sentence_input, gloss_key  # noqa: E402
from Processing.semcor_data import DEFAULT_ROOT, load_dataset, sample_balanced, sample_natural  # noqa: E402

CKPT = Path(os.environ.get("LLAMA_CKPT_ROOT", "/home/matt/.llama/checkpoints"))
DEFAULT_TOKENIZER = Path(os.environ.get("LLAMA_TOKENIZER", CKPT / "Llama3.2-1B-hf-tok" / "tokenizer.model"))
DEFAULT_WEIGHTS = REPO / "Data" / "ModelWeights" / "llama_jax_weights.pkl"
SAMPLES_DIR = DEFAULT_ROOT / "samples"
ACTIVATIONS_DIR = DEFAULT_ROOT / "activations"
STORE_META = {"model": "Llama-3.2-1B", "compute_dtype": "float32", "sentence_prompt": "bos+text+space+text",
              "gloss_prompt": "bos+'{lemma}: {gloss}'", "layers": "embeddings+16 blocks"}
TOKENS_PER_SECOND = 32.0  # measured on the 4-core development CPU, float32 (see Report/LAYER_STUDY.md)


# ---------------------------------------------------------------------------
# samples
# ---------------------------------------------------------------------------
def sample_path(name: str) -> Path:
    return SAMPLES_DIR / f"{name}.json"


def make_sample(ds, split: str, kind: str, n: int, seed: int = 0, name: str | None = None) -> dict:
    if kind == "natural":
        ids = sample_natural(ds, split, n_sentences=n, seed=seed)
    elif kind == "balanced":
        ids = sample_balanced(ds, split, n_instances=n, seed=seed)
    else:
        raise ValueError(kind)
    sample = {"name": name or f"{split}_{kind}_{n}", "split": split, "kind": kind, "n": n, "seed": seed,
              "instance_ids": ids}
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    sample_path(sample["name"]).write_text(json.dumps(sample, indent=1))
    return sample


def load_sample(name: str) -> dict:
    return json.loads(sample_path(name).read_text())


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def load_tokenizer(path: str | Path = DEFAULT_TOKENIZER):
    from LLM.llama.tokenizer import Tokenizer

    return Tokenizer(str(path))


def build_inputs(ds, tok, instance_ids: list[str]):
    """Sentence inputs (one per sentence, covering the requested targets) and unique gloss inputs."""
    by_sentence: dict[str, dict[str, tuple[int, int]]] = {}
    gloss_specs: dict[str, tuple[str, str]] = {}
    for iid in instance_ids:
        inst = ds.instances[iid]
        by_sentence.setdefault(inst["sentence_id"], {})[iid] = ds.target_span(iid)
        for syn in inst["candidates"]:
            gloss_specs.setdefault(gloss_key(inst["lemma"], syn), (inst["lemma"], syn))
    sentences = [build_sentence_input(tok, sid, ds.sentences[sid]["text"], targets)
                 for sid, targets in by_sentence.items()]
    glosses = [build_gloss_input(tok, lemma, syn, ds.synsets[syn]["gloss"]) for lemma, syn in gloss_specs.values()]
    return sentences, glosses


def open_stores(root: Path = ACTIVATIONS_DIR):
    from Processing.extract import GLOSS_VARIANTS, SENTENCE_VARIANTS, VectorStore

    return (VectorStore(root / "sentences", SENTENCE_VARIANTS, STORE_META),
            VectorStore(root / "glosses", GLOSS_VARIANTS, STORE_META))


SWAP_SEED = 0


def parse_swap_specs(specs: list[str]) -> list[tuple[str, int]]:
    """'pos:0', 'pos+form:0-2' -> [(match, draw), ...] in the order given."""
    from Processing.swaps import MATCH_MODES

    out = []
    for spec in specs:
        match, _, draws = spec.rpartition(":")
        if match not in MATCH_MODES or not draws:
            raise ValueError(f"bad swap spec {spec!r}: expected MATCH:DRAWS with MATCH in {MATCH_MODES}")
        first, _, last = draws.partition("-")
        out += [(match, j) for j in range(int(first), int(last or first) + 1)]
    return list(dict.fromkeys(out))


def open_swap_store(root: Path = ACTIVATIONS_DIR, seed: int = SWAP_SEED):
    from Processing.extract import SENTENCE_VARIANTS, VectorStore
    from Processing.swaps import POOL_VERSION

    meta = {**STORE_META, "swap_pool_version": POOL_VERSION, "swap_seed": seed}
    return VectorStore(root / f"swapped_seed{seed}", SENTENCE_VARIANTS, meta)


def swap_plans(ds, instance_ids: list[str], specs: list[tuple[str, int]], seed: int = SWAP_SEED):
    """[((match, draw), [Swap, ...]), ...] in spec order (see Processing/swaps.py)."""
    from Processing.swaps import build_pool, plan_swaps

    pool = build_pool(ds.instances.values())
    return [((m, j), plan_swaps(ds, instance_ids, draws=(j,), seed=seed, match=m, pool=pool)) for m, j in specs]


def _swap_todo(ds, tok, instance_ids, specs, root, seed, save: bool):
    from Processing.swaps import save_plan, swap_inputs

    store = open_swap_store(root, seed)
    todo = []
    for spec, plan in swap_plans(ds, instance_ids, specs, seed):
        if save:
            save_plan(store.root / "plan.json", plan)
        todo.append((spec, [x for x in swap_inputs(tok, plan) if x.sentence_id not in store]))
    return store, todo


def estimate(ds, tok, instance_ids: list[str], root: Path = ACTIVATIONS_DIR, swaps: list[tuple[str, int]] = (),
             swap_seed: int = SWAP_SEED) -> dict:
    """Token counts still to be computed (padding included) and a runtime estimate."""
    from Processing.extract import Extractor

    sentences, glosses = build_inputs(ds, tok, instance_ids)
    sent_store, gloss_store = open_stores(root)
    todo_s = [s for s in sentences if any(t not in sent_store for t in s.targets)]
    todo_g = [g for g in glosses if g.key not in gloss_store]
    todo_w = [x for _, xs in _swap_todo(ds, tok, instance_ids, swaps, root, swap_seed, save=False)[1] for x in xs] \
        if swaps else []

    def padded(seqs):
        ex = Extractor(params=None, args=None)
        return sum(max(1, ex.tokens_per_batch // T) * T for T, _ in ex.batches([len(s) for s in seqs]))

    s_tok, g_tok = padded([s.token_ids for s in todo_s]), padded([g.token_ids for g in todo_g])
    w_tok = padded([x.token_ids for x in todo_w])
    total = s_tok + g_tok + w_tok
    return {"instances": len(instance_ids), "sentences_to_run": len(todo_s), "glosses_to_run": len(todo_g),
            "swapped_sentences_to_run": len(todo_w), "padded_tokens": total,
            "estimated_minutes": round(total / TOKENS_PER_SECOND / 60, 1)}


def extract(ds, tok, instance_ids: list[str], weights: Path = DEFAULT_WEIGHTS, root: Path = ACTIVATIONS_DIR,
            dtype: str = "float32", log=print, swaps: list[tuple[str, int]] = (), swap_seed: int = SWAP_SEED) -> dict:
    """Compute and cache all vectors needed for `instance_ids` (and their swapped sentences). Returns timing stats."""
    import jax.numpy as jnp

    from LLM.llama_jax.weights import load_params
    from Processing.extract import Extractor, extract_glosses, extract_sentences

    sentences, glosses = build_inputs(ds, tok, instance_ids)
    sent_store, gloss_store = open_stores(root)
    swap_store, swap_todo = _swap_todo(ds, tok, instance_ids, swaps, root, swap_seed, save=True) if swaps \
        else (None, [])
    if (not any(any(t not in sent_store for t in s.targets) for s in sentences)
            and not gloss_store.missing(g.key for g in glosses) and not any(xs for _, xs in swap_todo)):
        log("everything already cached")
        return {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # legacy-pickle repair warning; the repair is tested
        params, args = load_params(weights, dtype=jnp.float32 if dtype == "float32" else jnp.bfloat16)
    ex = Extractor(params, args, log=log)
    extract_sentences(ex, sentences, sent_store)
    extract_glosses(ex, glosses, gloss_store)
    for (match, draw), inputs in swap_todo:
        log(f"swapped sentences, match={match} draw={draw}: {len(inputs)} to run")
        extract_sentences(ex, inputs, swap_store)
        log(f"swapped sentences, match={match} draw={draw}: done ({json.dumps(ex.stats)})")
    return ex.stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("sample", help="draw a sample of instance ids")
    sp.add_argument("--split", choices=["dev", "test"], required=True)
    sp.add_argument("--kind", choices=["natural", "balanced"], required=True)
    sp.add_argument("--n", type=int, required=True, help="sentences (natural) or instances (balanced)")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--name")
    for name in ("estimate", "extract"):
        p = sub.add_parser(name)
        p.add_argument("--samples", nargs="+", required=True, help="sample names in Data/Processed/v2/samples")
        p.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
        p.add_argument("--swaps", nargs="*", default=[],
                       help="swapped sentences to add, e.g. pos:0 pos+form:0 pos:1-2 (see Processing/swaps.py)")
        p.add_argument("--swap-seed", type=int, default=SWAP_SEED)
        if name == "extract":
            p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
            p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    a = parser.parse_args()

    ds = load_dataset()
    if a.cmd == "sample":
        s = make_sample(ds, a.split, a.kind, a.n, a.seed, a.name)
        print(f"wrote {sample_path(s['name'])}: {len(s['instance_ids'])} instances")
        return
    tok = load_tokenizer(a.tokenizer)
    ids = list(dict.fromkeys(i for name in a.samples for i in load_sample(name)["instance_ids"]))
    specs = parse_swap_specs(a.swaps)
    if a.cmd == "estimate":
        print(json.dumps(estimate(ds, tok, ids, swaps=specs, swap_seed=a.swap_seed), indent=1))
    else:
        print(json.dumps(extract(ds, tok, ids, Path(a.weights), dtype=a.dtype, swaps=specs, swap_seed=a.swap_seed),
                         indent=1), flush=True)


# ---------------------------------------------------------------------------
# legacy helpers, kept so the March 2025 analysis notebooks still import
# ---------------------------------------------------------------------------
def load_jax_array(key: str, store_path: str):
    """Load one pickled array from a v1 jax_store directory."""
    file_path = os.path.join(store_path, f"{key}.pkl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JAX array file not found: {file_path}")
    with open(file_path, "rb") as f:
        return pickle.load(f)


def unzip_jax_store(zip_path: str, extract_to: str | None = None) -> None:
    """Extract a v1 jax_store zip (fixed: strips the .zip suffix, not the characters '.', 'z', 'i', 'p')."""
    if not zip_path.endswith(".zip"):
        raise ValueError("Provided file is not a .zip archive")
    extract_to = extract_to or zip_path.removesuffix(".zip")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"JAX store unzipped to {extract_to}")


if __name__ == "__main__":
    main()
