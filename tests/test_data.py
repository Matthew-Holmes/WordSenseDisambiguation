"""Tests for the v2 SemCor dataset (Processing/semcor_data.py).

Run from the repository root:  .venv/bin/python -m tests.run_tests data

The detokeniser tests need nothing. The corpus-level tests need the built dataset and are skipped
if it is missing (build it with `.venv/bin/python -m Processing.build_dataset`).
"""

from __future__ import annotations

import functools
import unittest
from collections import defaultdict

from Processing.semcor_data import (
    DEFAULT_ROOT,
    detokenize,
    load_dataset,
    make_splits,
    normalise_token,
    sample_balanced,
    sample_natural,
)


@functools.lru_cache(maxsize=1)
def _load():
    return load_dataset()


def _ds():
    if not (DEFAULT_ROOT / "instances.jsonl").exists():
        raise unittest.SkipTest("v2 dataset not built: run `python -m Processing.build_dataset`")
    return _load()


def _check_spans(tokens: list[str], text: str, spans: list[tuple[int, int]]) -> None:
    assert len(spans) == len(tokens)
    prev_end = 0
    for i, (tok, (s, e)) in enumerate(zip(tokens, spans)):
        assert s >= prev_end, f"overlapping span at token {i}"
        assert text[s:e] == normalise_token(tok, tokens[i - 1] if i else None), (tok, text[s:e])
        prev_end = e


# --------------------------------------------------------------------------- detokeniser
def test_detokenize_quotes_and_punctuation():
    tokens = ["``", "Hello", ",", "''", "he", "said", "."]
    text, spans = detokenize(tokens)
    assert text == '"Hello," he said.'
    _check_spans(tokens, text, spans)


def test_detokenize_clitics_and_possessives():
    cases = {
        ("John", "'s", "dog", "does", "n't", "bark", "."): "John's dog doesn't bark.",
        ("ca", "n't", "you", "'ve got", "it", "?"): "can't you've got it?",
        ("thin", "'em", "out"): "thin 'em out",  # not a clitic we attach
        ("the", "players", "'", "union"): "the players' union",
        ("I", "'d", "say", "we", "'re", "fine"): "I'd say we're fine",
    }
    for tokens, expected in cases.items():
        text, spans = detokenize(list(tokens))
        assert text == expected, (text, expected)
        _check_spans(list(tokens), text, spans)


def test_detokenize_never_alters_token_characters():
    tokens = ["It", "costs", "$", "3.5", "million", "at", "10:30", "for", "the", "U.S.", "army", "and", "Jr.", "."]
    text, spans = detokenize(tokens)
    assert text == "It costs $3.5 million at 10:30 for the U.S. army and Jr.."
    _check_spans(tokens, text, spans)
    for tok in ("3.5", "10:30", "U.S.", "Jr."):
        assert tok in text


def test_detokenize_brackets_percent_and_multiword_tokens():
    tokens = ["(", "about", "3", "%", ")", "a", "purchasing agent", "[", "sic", "]"]
    text, spans = detokenize(tokens)
    assert text == "(about 3%) a purchasing agent [sic]"
    _check_spans(tokens, text, spans)
    s, e = spans[6]
    assert text[s:e] == "purchasing agent"


def test_detokenize_backtick_is_degree_after_number_else_quote():
    text, spans = detokenize(["rose", "to", "105", "`", "C", "."])
    assert text == "rose to 105° C."
    text2, _ = detokenize(["he", "said", "`", "maybe", "'"])
    assert text2 == "he said 'maybe'"
    _check_spans(["rose", "to", "105", "`", "C", "."], text, spans)


def test_detokenize_merged_quote_token():
    tokens = ["toast", "of", "two", "continents", "''.", "``", "Next"]
    text, spans = detokenize(tokens)
    assert text == 'toast of two continents". "Next'
    _check_spans(tokens, text, spans)


# --------------------------------------------------------------------------- corpus checks
def test_corpus_spans_match_tokens():
    ds = _ds()
    for s in ds.sentences.values():
        _check_spans(s["tokens"], s["text"], s["spans"])


def test_instances_point_at_their_word():
    ds = _ds()
    for iid, inst in ds.instances.items():
        sent = ds.sentences[inst["sentence_id"]]
        assert sent["tokens"][inst["token_index"]] == inst["word"], iid
        s, e = ds.target_span(iid)
        prev = sent["tokens"][inst["token_index"] - 1] if inst["token_index"] else None
        assert sent["text"][s:e] == normalise_token(inst["word"], prev), iid
        assert inst["doc_id"] == sent["doc_id"]


def test_gold_is_subset_of_candidates():
    ds = _ds()
    for iid, inst in ds.instances.items():
        assert inst["gold_synsets"], iid
        assert len(inst["gold_indices"]) == len(inst["gold_synsets"]), iid
        for k, g in zip(inst["gold_indices"], inst["gold_synsets"]):
            assert inst["candidates"][k] == g, iid
        assert len(set(inst["candidates"])) == len(inst["candidates"]), f"duplicate candidates in {iid}"
        for name in inst["candidates"]:
            assert name in ds.synsets, name


def test_candidates_share_pos_and_contain_lemma():
    ds = _ds()
    for iid, inst in ds.instances.items():
        assert inst["pos"] in ("n", "v", "a", "r"), iid
        for name in inst["candidates"]:
            syn = ds.synsets[name]
            assert syn["pos"].replace("s", "a") == inst["pos"], (iid, name)
            assert inst["lemma"].lower() in {lem.lower() for lem in syn["lemmas"]}, (iid, name)


def test_null_word_preserved():
    ds = _ds()
    nulls = [inst for inst in ds.instances.values() if inst["word"] == "null"]
    assert nulls, "the SemCor word 'null' should survive the round trip as a string"
    assert all(isinstance(inst["word"], str) for inst in ds.instances.values())


def test_splits_disjoint_cover_and_stratified():
    ds = _ds()
    dev, test = set(ds.splits["dev"]), set(ds.splits["test"])
    all_docs = {s["doc_id"] for s in ds.sentences.values()}
    assert not dev & test
    assert dev | test == all_docs
    assert len(all_docs) == 352
    genre = {s["doc_id"]: s["doc_source"].split("-")[1][0] for s in ds.sentences.values()}
    counts = defaultdict(lambda: [0, 0])
    for d in dev:
        counts[genre[d]][0] += 1
    for d in test:
        counts[genre[d]][1] += 1
    assert all(abs(a - b) <= 1 for a, b in counts.values()), dict(counts)
    sources = {s["doc_id"]: s["doc_source"] for s in ds.sentences.values()}
    assert make_splits(sources, seed=0)["dev"] == ds.splits["dev"]  # deterministic


def test_sample_natural_properties():
    ds = _ds()
    a = sample_natural(ds, "dev", 40, seed=1)
    assert a == sample_natural(ds, "dev", 40, seed=1)
    assert a != sample_natural(ds, "dev", 40, seed=2)
    assert len(a) == len(set(a))
    docs = set(ds.splits["dev"])
    sentences = {ds.instances[i]["sentence_id"] for i in a}
    assert len(sentences) == 40
    for iid in a:
        inst = ds.instances[iid]
        assert inst["doc_id"] in docs and len(inst["candidates"]) >= 2
    # every eligible instance of a sampled sentence is included
    eligible = {iid for iid, inst in ds.instances.items()
                if inst["sentence_id"] in sentences and len(inst["candidates"]) >= 2}
    assert eligible == set(a)
    order = list(ds.instances)
    assert a == sorted(a, key=order.index)


def test_sample_balanced_properties():
    ds = _ds()
    n, max_senses = 120, 4
    b = sample_balanced(ds, "test", n, seed=3, max_senses_per_lemma=max_senses)
    assert b == sample_balanced(ds, "test", n, seed=3, max_senses_per_lemma=max_senses)
    assert len(b) == len(set(b)) and len(b) <= n
    docs = set(ds.splits["test"])
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for iid in b:
        inst = ds.instances[iid]
        assert inst["doc_id"] in docs and len(inst["candidates"]) >= 2
        assert len(inst["gold_synsets"]) == 1
        groups[(inst["lemma"], inst["pos"])].append(inst["gold_synsets"][0])
    for key, senses in groups.items():
        assert 2 <= len(senses) <= max_senses, key
        assert len(set(senses)) == len(senses), f"two instances of one sense for {key}"
