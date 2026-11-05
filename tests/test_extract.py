"""VectorStore and batch planning (fast; no model)."""

from __future__ import annotations

import tempfile

import numpy as np

from Processing.extract import BUCKETS, Extractor, VectorStore


def test_vector_store_round_trip_and_resume():
    with tempfile.TemporaryDirectory() as d:
        store = VectorStore(d, ("a", "b"), {"v": 1})
        x = np.random.default_rng(0).standard_normal((3, 4, 5)).astype(np.float32)
        store.add(["k1", "k2", "k3"], {"a": x, "b": -x})
        store.add(["k4"], {"a": x[:1] * 2, "b": x[:1]})
        again = VectorStore(d, ("a", "b"), {"v": 1})  # reopen: index survives
        assert len(again) == 4 and "k4" in again and again.missing(["k1", "k9"]) == ["k9"]
        np.testing.assert_allclose(again.get(["k3", "k1"], "a"), x[[2, 0]].astype(np.float16))
        np.testing.assert_allclose(again.get(["k4"], "b"), x[:1].astype(np.float16))
        assert again.get(["k2"], "a").dtype == np.float16


def test_vector_store_refuses_mismatched_settings():
    with tempfile.TemporaryDirectory() as d:
        VectorStore(d, ("a",), {"prompt": "v1"})
        for variants, meta in ((("b",), {"prompt": "v1"}), (("a",), {"prompt": "v2"})):
            try:
                VectorStore(d, variants, meta)
            except ValueError:
                pass
            else:
                raise AssertionError("expected a ValueError")


def test_batches_cover_every_sequence_once_within_budget():
    ex = Extractor(params=None, args=None)
    lengths = [5, 17, 17, 30, 70, 200, 3, 64, 65]
    plan = ex.batches(lengths)
    seen = sorted(i for _, idx in plan for i in idx)
    assert seen == list(range(len(lengths)))
    for T, idx in plan:
        assert T in BUCKETS and all(lengths[i] <= T for i in idx)
        assert len(idx) <= max(1, ex.tokens_per_batch // T)
