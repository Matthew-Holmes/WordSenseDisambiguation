"""Swapped-sentence construction (Processing/swaps.py). Corpus tests need the v2 dataset; alignment needs the tokenizer."""

from __future__ import annotations

import random
import unittest
from collections import Counter
from functools import lru_cache

from Processing import swaps as sw
from tests.test_prompts import _tok


@lru_cache(maxsize=1)
def _ds():
    from Processing.semcor_data import DEFAULT_ROOT, load_dataset

    if not (DEFAULT_ROOT / "instances.jsonl").exists():
        raise unittest.SkipTest("v2 dataset not built (python -m Processing.build_dataset)")
    return load_dataset()


@lru_cache(maxsize=1)
def _pool():
    return sw.build_pool(_ds().instances.values())


def _toy_instances():
    rows = [("dog", "dog", "n"), ("dogs", "dog", "n"), ("Rex", "rex", "n"), ("cat", "cat", "n"), ("cats", "cat", "n"),
            ("hen", "hen", "n"), ("New York", "new_york", "n"), ("3", "3", "n"), ("ran", "run", "v"),
            ("running", "run", "v"), ("walked", "walk", "v"), ("x", "x", "n")]
    return [{"word": w, "lemma": lem, "pos": p} for w, lem, p in rows]


def test_synset_pos_folds_satellites_and_keeps_dotted_lemmas():
    assert sw.synset_pos("long.a.01") == "a" and sw.synset_pos("long.s.03") == "a"
    assert sw.synset_pos("st._john's_wort.n.01") == "n" and sw.synset_pos("run.v.02") == "v"
    assert sw.needed_pos(["long.a.01", "long.s.03", "long.r.01", "long.a.02"]) == ["a", "r"]


def test_word_form_classes():
    cases = [("objectives", "objective", "n", "plural"), ("objective", "objective", "n", "base"),
             ("reviewed", "review", "v", "ed"), ("reviewing", "review", "v", "ing"), ("reviews", "review", "v", "s"),
             ("took", "take", "v", "irregular"), ("Review", "review", "v", "base"), ("longer", "long", "a", "comparative"),
             ("longest", "long", "a", "superlative"), ("worse", "bad", "a", "irregular"), ("set up", "set_up", "v", "base")]
    for word, lemma, pos, expected in cases:
        assert sw.word_form(word, lemma, pos) == expected, (word, sw.word_form(word, lemma, pos), expected)


def test_pool_keeps_only_lowercase_single_words_and_records_lemma_forms():
    pool = sw.build_pool(_toy_instances())
    assert pool.by_pos["n"] == ["cat", "cats", "dog", "dogs", "hen"]  # no "Rex", "New York", "3", "x"
    assert pool.by_pos["v"] == ["ran", "running", "walked"]
    assert pool.by_pos_form[("n", "plural")] == ["cats", "dogs"]
    assert pool.lemma_forms[("dog", "n")] == {"dog", "dogs"} and pool.lemma_forms[("rex", "n")] == {"rex"}
    assert sw.usable_form("well-known") and sw.usable_form("o'clock") and not sw.usable_form("'s")


def test_draw_never_returns_a_form_of_the_target_lemma_and_reaches_every_other_form():
    pool = sw.build_pool(_toy_instances())
    rng = random.Random(0)
    seen = Counter(sw.draw_replacement(pool, "Dogs", "dog", "n", rng) for _ in range(3000))
    assert set(seen) == {"cat", "cats", "hen"}
    assert min(seen.values()) > 800  # uniform: about 1000 each
    a = [sw.draw_replacement(pool, "dog", "dog", "n", random.Random("k")) for _ in range(3)]
    assert len(set(a)) == 1  # same seed, same draw


def test_draw_pos_form_falls_back_to_the_whole_pos_pool_when_the_class_is_small():
    pool = sw.build_pool(_toy_instances())  # every class here is below MIN_FORM_POOL
    picks = {sw.draw_replacement(pool, "dogs", "dog", "n", random.Random(i), "pos+form") for i in range(200)}
    assert picks == {"cat", "cats", "hen"}
    try:
        sw.draw_replacement(pool, "dog", "dog", "n", random.Random(0), "lemma")
    except ValueError:
        pass
    else:
        raise AssertionError("expected a ValueError for an unknown match mode")


def test_match_case_and_replace_span():
    assert sw.match_case("ripe", "How") == "Ripe" and sw.match_case("toad", "NASA") == "TOAD"
    assert sw.match_case("toad", "how") == "toad" and sw.match_case("toad", "I") == "Toad"
    text, span = sw.replace_span("How long has it been", (4, 8), "ripe")
    assert text == "How ripe has it been" and text[span[0]: span[1]] == "ripe"
    text, span = sw.replace_span("a bc d", (2, 4), "xyz")
    assert text == "a xyz d" and span == (2, 5)


def test_swaps_on_corpus_change_only_the_target():
    ds, pool = _ds(), _pool()
    ids = list(ds.instances)[:3000:7]
    for s in sw.plan_swaps(ds, ids, draws=(0, 1), pool=pool):
        text = ds.sentences[s.sentence_id]["text"]
        a, b = ds.target_span(s.instance_id)
        assert s.text[s.span[0]: s.span[1]] == s.replacement
        assert s.text[: s.span[0]].rstrip() == text[:a].rstrip() and s.text[s.span[1]:].lstrip() == text[b:].lstrip()
        assert s.replacement.lower() not in sw.excluded_forms(pool, ds.instances[s.instance_id]["word"],
                                                              ds.instances[s.instance_id]["lemma"], s.pos)
        assert s.replacement.lower() in pool.by_pos[s.pos]


def test_clitic_targets_are_kept_apart_from_their_neighbour():
    ds, pool = _ds(), _pool()
    s = sw.make_swap(ds, pool, "d191.s074.t000", "v", 0)  # "It's the old ..." with target "'s" (be)
    assert s.original == "'s" and s.text.startswith(f"It {s.replacement} the old")


def test_plan_is_deterministic_order_free_and_draws_are_independent():
    ds, pool = _ds(), _pool()
    ids = list(ds.instances)[100:160]
    full = {s.key: s for s in sw.plan_swaps(ds, ids, draws=(0, 1, 2), pool=pool)}
    part = {s.key: s for s in sw.plan_swaps(ds, ids[::-1][:20], draws=(1,), pool=pool)}
    assert all(full[k] == v for k, v in part.items())
    assert len({(s.instance_id, s.draw) for s in full.values()}) == 3 * len(ids)
    form = sw.plan_swaps(ds, ids, draws=(0,), pool=pool, match="pos+form")
    assert all(s.key.endswith(":0") and ":pos+form:" in s.key for s in form)
    same = sum(full[sw.swap_key(s.instance_id, s.pos, 0)].replacement == s.replacement for s in form)
    assert same < len(form) // 4  # the two modes draw independently


def test_pos_form_matches_the_inflection_class_when_the_class_is_large():
    ds, pool = _ds(), _pool()
    ids = [i for i in list(ds.instances)[:5000] if ds.instances[i]["pos"] in ("n", "v")][:300]
    for s in sw.plan_swaps(ds, ids, pool=pool, match="pos+form"):
        inst = ds.instances[s.instance_id]
        cls = sw.word_form(inst["word"], inst["lemma"], inst["pos"])
        if len(pool.by_pos_form.get((s.pos, cls), [])) >= sw.MIN_FORM_POOL:
            assert s.replacement.lower() in pool.by_pos_form[(s.pos, cls)], (inst["word"], s.replacement, cls)


def test_swap_inputs_point_at_the_replacement_in_both_copies():
    ds, pool, tok = _ds(), _pool(), _tok()
    plan = sw.plan_swaps(ds, list(ds.instances)[:4000:13] + ["d191.s074.t000"], pool=pool)
    for s, x in zip(plan, sw.swap_inputs(tok, plan)):
        assert list(x.targets) == [s.key]
        for copy in ("plain", "echo"):
            first, last = x.targets[s.key][copy]
            span_text = tok.decode(x.token_ids[first: last + 1])
            assert s.replacement in span_text, (copy, s.replacement, span_text)
            assert s.replacement.endswith(tok.decode(x.token_ids[last: last + 1]).strip()), (copy, s.replacement)


def test_plan_round_trips_through_json_and_merges():
    import tempfile
    from pathlib import Path

    ds, pool = _ds(), _pool()
    a = sw.plan_swaps(ds, list(ds.instances)[:5], pool=pool)
    b = sw.plan_swaps(ds, list(ds.instances)[3:8], draws=(1,), pool=pool)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "plan.json"
        sw.save_plan(path, a)
        sw.save_plan(path, b)
        back = {s.key: s for s in sw.load_plan(path)}
    assert set(back) == {s.key for s in a + b} and all(back[s.key] == s for s in a + b)
