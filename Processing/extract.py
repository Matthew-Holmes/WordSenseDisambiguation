"""Batched activation extraction with the fixed JAX Llama, and a compact resumable vector store.

Fixes the storage/compute issues in Report/REPORT.md (B16): one forward pass per sentence serves all
its targets, only the pooled vectors that are used are kept (17 layers x 2048 dims, float16, ~70 KB
per vector instead of a 590 KB pickle per prompt), and nothing needs JAX to be read back.

Layer convention (HuggingFace): index 0 = token embeddings, index k = output of block k (1..16).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import jax.numpy as jnp
import numpy as np

from LLM.llama_jax import model as llama
from Processing.prompts import GlossInput, SentenceInput

BUCKETS = (16, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 384, 512)
TOKENS_PER_BATCH = 512
SENTENCE_VARIANTS = ("plain_last", "echo_last", "echo_mean")
GLOSS_VARIANTS = ("mean", "last")


# ---------------------------------------------------------------------------
# vector store
# ---------------------------------------------------------------------------
class VectorStore:
    """Append-only directory of float16 vectors, one array file per (part, variant).

    index.json holds the variants, shapes, metadata and, per part, the ids of its rows. Adding rows
    writes a new part and then atomically rewrites index.json, so an interrupted run loses at most
    the batch in flight.
    """

    def __init__(self, root: str | Path, variants: Sequence[str], meta: dict | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        index_path = self.root / "index.json"
        if index_path.exists():
            self.index = json.loads(index_path.read_text())
            if list(self.index["variants"]) != list(variants):
                raise ValueError(f"{root}: variants {self.index['variants']} != {list(variants)}")
            if meta and self.index.get("meta") != meta:
                raise ValueError(f"{root} was built with different settings: {self.index.get('meta')} vs {meta}")
        else:
            self.index = {"variants": list(variants), "meta": meta or {}, "parts": []}
            self._write_index()
        self._row = {i: (p, r) for p, part in enumerate(self.index["parts"]) for r, i in enumerate(part["ids"])}
        self._cache: dict[tuple[int, str], np.ndarray] = {}

    @property
    def variants(self) -> list[str]:
        return list(self.index["variants"])

    def __contains__(self, key: str) -> bool:
        return key in self._row

    def __len__(self) -> int:
        return len(self._row)

    def missing(self, keys: Iterable[str]) -> list[str]:
        return [k for k in dict.fromkeys(keys) if k not in self._row]

    def _write_index(self) -> None:
        tmp = self.root / "index.json.tmp"
        tmp.write_text(json.dumps(self.index))
        os.replace(tmp, self.root / "index.json")

    def add(self, keys: list[str], arrays: dict[str, np.ndarray]) -> None:
        if not keys:
            return
        part = len(self.index["parts"])
        for v in self.variants:
            a = np.asarray(arrays[v], dtype=np.float16)
            assert a.shape[0] == len(keys), (v, a.shape, len(keys))
            np.save(self.root / f"part{part:05d}.{v}.npy", a)
        self.index["parts"].append({"ids": list(keys)})
        self._write_index()
        for r, k in enumerate(keys):
            self._row[k] = (part, r)

    def _part(self, p: int, variant: str) -> np.ndarray:
        if (p, variant) not in self._cache:
            self._cache[(p, variant)] = np.load(self.root / f"part{p:05d}.{variant}.npy", mmap_mode="r")
        return self._cache[(p, variant)]

    def get(self, keys: Sequence[str], variant: str) -> np.ndarray:
        """Stacked vectors for `keys` (float16, shape (n, L, D)); KeyError if any key is missing."""
        rows = []
        for k in keys:
            part, row = self._row[k]
            rows.append(self._part(part, variant)[row])
        return np.stack(rows)


# ---------------------------------------------------------------------------
# model execution
# ---------------------------------------------------------------------------
@dataclass
class Extractor:
    params: dict
    args: llama.ModelArgs
    tokens_per_batch: int = TOKENS_PER_BATCH
    log: Callable[[str], None] = print
    stats: dict = field(default_factory=lambda: {"tokens": 0, "padded_tokens": 0, "seconds": 0.0})

    @staticmethod
    def bucket(n: int) -> int:
        for b in BUCKETS:
            if n <= b:
                return b
        raise ValueError(f"sequence of {n} tokens exceeds the largest bucket ({BUCKETS[-1]})")

    def batches(self, lengths: Sequence[int]) -> list[tuple[int, list[int]]]:
        """Group sequence indices into (bucket_len, indices) batches with a fixed batch size per bucket."""
        groups: dict[int, list[int]] = {}
        for i in np.argsort(lengths, kind="stable"):
            groups.setdefault(self.bucket(lengths[i]), []).append(int(i))
        out = []
        for T, idx in sorted(groups.items()):
            B = max(1, self.tokens_per_batch // T)
            out += [(T, idx[k : k + B]) for k in range(0, len(idx), B)]
        return out

    def forward(self, token_lists: list[list[int]], T: int) -> np.ndarray:
        """Hidden states (n, L+1, T, D) float32 for up to B sequences right-padded to T."""
        B = max(1, self.tokens_per_batch // T)
        tokens = np.full((B, T), llama.PAD_ID, dtype=np.int32)
        lengths = np.ones(B, dtype=np.int32)  # unused rows: a lone BOS
        tokens[:, 0] = llama.BOS_ID
        for i, t in enumerate(token_lists):
            tokens[i, : len(t)] = t
            lengths[i] = len(t)
        states = llama.hidden_states_jit(
            jnp.asarray(tokens), self.params, llama.padding_mask(jnp.asarray(lengths), T),
            self.args.n_heads, self.args.n_kv_heads,
        )
        self.stats["tokens"] += int(sum(len(t) for t in token_lists))
        self.stats["padded_tokens"] += B * T
        return np.asarray(states[: len(token_lists)], dtype=np.float32)

    def _run(self, seqs: list[list[int]], pool, label: str, on_batch) -> None:
        plan = self.batches([len(s) for s in seqs])
        total = sum(len(s) for s in seqs)
        done, start = 0, time.time()
        for b, (T, idx) in enumerate(plan):
            states = self.forward([seqs[i] for i in idx], T)
            on_batch(idx, [pool(i, states[j]) for j, i in enumerate(idx)])
            done += sum(len(seqs[i]) for i in idx)
            elapsed = time.time() - start
            if b % 10 == 0 or b == len(plan) - 1:
                eta = elapsed / max(done, 1) * (total - done)
                self.log(f"  {label}: batch {b + 1}/{len(plan)}  {done}/{total} tokens  "
                         f"{done / max(elapsed, 1e-9):.1f} tok/s  eta {eta / 60:.1f} min")
        self.stats["seconds"] += time.time() - start


def extract_sentences(ex: Extractor, inputs: list[SentenceInput], store: VectorStore, flush_every: int = 200) -> None:
    """Pool plain_last / echo_last / echo_mean vectors for every target of every sentence input."""
    todo = [s for s in inputs if any(t not in store for t in s.targets)]
    if not todo:
        return
    buffer: dict[str, dict[str, np.ndarray]] = {}

    def pool(i: int, h: np.ndarray) -> dict:
        out = {}
        for iid, pos in todo[i].targets.items():
            (pf, pl), (ef, el) = pos["plain"], pos["echo"]
            out[iid] = {"plain_last": h[:, pl], "echo_last": h[:, el], "echo_mean": h[:, ef : el + 1].mean(axis=1)}
        return out

    def on_batch(idx, pooled):
        for d in pooled:
            buffer.update(d)
        if len(buffer) >= flush_every:
            _flush(store, buffer)

    ex._run([s.token_ids for s in todo], pool, "sentences", on_batch)
    _flush(store, buffer)


def extract_glosses(ex: Extractor, inputs: list[GlossInput], store: VectorStore, flush_every: int = 500) -> None:
    """Pool the mean over gloss tokens and the last token's state for every definition prompt."""
    todo = [g for g in inputs if g.key not in store]
    if not todo:
        return
    buffer: dict[str, dict[str, np.ndarray]] = {}

    def pool(i: int, h: np.ndarray) -> dict:
        a, b = todo[i].span
        return {todo[i].key: {"mean": h[:, a:b].mean(axis=1), "last": h[:, b - 1]}}

    def on_batch(idx, pooled):
        for d in pooled:
            buffer.update(d)
        if len(buffer) >= flush_every:
            _flush(store, buffer)

    ex._run([g.token_ids for g in todo], pool, "glosses", on_batch)
    _flush(store, buffer)


def _flush(store: VectorStore, buffer: dict[str, dict[str, np.ndarray]]) -> None:
    keys = [k for k in buffer if k not in store]
    if keys:
        store.add(keys, {v: np.stack([buffer[k][v] for k in keys]) for v in store.variants})
    buffer.clear()
