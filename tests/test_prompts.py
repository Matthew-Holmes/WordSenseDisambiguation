"""Target alignment and prompt construction (fixes B1, B6, B13). Needs the tokenizer and the v2 dataset."""

from __future__ import annotations

import os
import random
import unittest
from functools import lru_cache
from pathlib import Path

from LLM.llama_jax.model import BOS_ID
from Processing import prompts

TOKENIZER = Path(os.environ.get("LLAMA_TOKENIZER", "/home/matt/.llama/checkpoints/Llama3.2-1B-hf-tok/tokenizer.model"))


@lru_cache(maxsize=1)
def _tok():
    if not TOKENIZER.exists():
        raise unittest.SkipTest(f"missing tokenizer {TOKENIZER}")
    from LLM.llama.tokenizer import Tokenizer

    return Tokenizer(str(TOKENIZER))


@lru_cache(maxsize=1)
def _ds():
    from Processing.semcor_data import DEFAULT_ROOT, load_dataset

    if not (DEFAULT_ROOT / "instances.jsonl").exists():
        raise unittest.SkipTest("v2 dataset not built (python -m Processing.build_dataset)")
    return load_dataset()


def _decoded(tok, ids, first, last):
    return tok.decode(ids[first : last + 1])


def test_offsets_round_trip_and_token_range():
    tok = _tok()
    text = 'Is your purchasing agent offering too much "free" service?'
    ids, spans = prompts.encode_with_offsets(tok, text)
    assert "".join(text[s:e] for s, e in spans) == text
    s = text.index("offering")
    first, last = prompts.token_range(spans, s, s + len("offering"))
    assert tok.decode(ids[first : last + 1]).strip() == "offering"


def test_the_b1_example_now_points_at_the_target():
    """'purchasing agent' is one SemCor word but two tokens; the old code read ' agent' for 'offering'."""
    tok = _tok()
    text = "Is your purchasing agent offering too much free service?"
    s = text.index("offering")
    inp = prompts.build_sentence_input(tok, "x", text, {"t": (s, s + len("offering"))})
    assert inp.token_ids[0] == BOS_ID
    for copy in ("plain", "echo"):
        first, last = inp.targets["t"][copy]
        assert _decoded(tok, inp.token_ids, first, last).strip() == "offering", copy
    assert inp.targets["t"]["plain"][1] < inp.targets["t"]["echo"][0]


def test_first_word_and_punctuation_heavy_targets():
    tok = _tok()
    text = 'U.S. officials said the 3.5% rise was "unprecedented" at 10:30 a.m.'
    for word in ("U.S.", "3.5", "unprecedented", "10:30"):
        s = text.index(word)
        inp = prompts.build_sentence_input(tok, "x", text, {"t": (s, s + len(word))})
        for copy in ("plain", "echo"):
            got = _decoded(tok, inp.token_ids, *inp.targets["t"][copy]).strip()
            assert word in got and len(got) <= len(word) + 2, (word, copy, got)


def test_gloss_prompts_never_raise_and_span_the_gloss():
    """B13: lemmas ending in punctuation used to raise 'colon absorbed'; now they just work."""
    tok = _tok()
    for lemma in ("bank", "jr.", "u.s.", "add_up", "null"):
        gloss = "a sloping land (especially the slope beside a body of water)"
        g = prompts.build_gloss_input(tok, lemma, "bank.n.01", gloss)
        assert g.token_ids[0] == BOS_ID
        a, b = g.span
        assert tok.decode(g.token_ids[a:b]).strip() == gloss
        assert g.key == f"{lemma}|bank.n.01"


def test_alignment_on_random_semcor_sentences():
    """Every target in 1,500 random sentences decodes to its own word in both copies."""
    tok, ds = _tok(), _ds()
    by_sentence: dict[str, dict[str, tuple[int, int]]] = {}
    for iid, inst in ds.instances.items():
        by_sentence.setdefault(inst["sentence_id"], {})[iid] = ds.target_span(iid)
    rng = random.Random(0)
    checked = mismatched = 0
    for sid in rng.sample(sorted(by_sentence), 1500):
        text = ds.sentences[sid]["text"]
        inp = prompts.build_sentence_input(tok, sid, text, by_sentence[sid])
        for iid, (s, e) in by_sentence[sid].items():
            word = text[s:e]
            for copy in ("plain", "echo"):
                got = _decoded(tok, inp.token_ids, *inp.targets[iid][copy])
                checked += 1
                # the tokens must cover the word and add at most a leading space or adjacent punctuation
                if word not in got or len(got.strip()) > len(word) + 2:
                    mismatched += 1
    print(f"checked {checked} target spans, {mismatched} mismatches")
    assert mismatched == 0
