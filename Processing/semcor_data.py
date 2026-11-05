"""SemCor v2: parsing, detokenisation, POS-restricted candidates, document splits and sampling.

Replaces the CSV pipeline of `Data/SemCore*.ipynb` and fixes the data-side issues in
Report/REPORT.md:

* B13 - everything is JSON lines, so there are no separator-joined strings and pandas never
  turns the word "null" into NaN;
* B14 - candidate senses are restricted to the lemma's part of speech (adjectives include
  satellites), in WordNet sense order;
* B15 - the detokeniser never rewrites a token's own characters ("3.5", "U.S.", "10:30" stay
  intact) and records the exact character span of every SemCor token, so a target word can be
  located in the text without counting words or tokens (B1);
* B7 - candidate lists are real lists, not strings joined with a separator.

It also adds a document-level dev/test split (stratified by Brown genre) so that layer selection
and evaluation never share a document.

Build the files once with `python -m Processing.build_dataset`, then use `load_dataset()`.
"""

from __future__ import annotations

import json
import random
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "Data" / "Processed" / "v2"
RAW_XML = REPO / "Data" / "Raw" / "semcor.data.xml"
RAW_KEYS = REPO / "Data" / "Raw" / "semcor.gold.key.txt"

# Sense-key ss_type -> WordNet POS (satellite adjectives are folded into "a").
SS_TYPE_TO_POS = {"1": "n", "2": "v", "3": "a", "4": "r", "5": "a"}
XML_POS_TO_WN = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}


# --------------------------------------------------------------------------- detokeniser
_NO_SPACE_BEFORE = {",", ".", ";", ":", "!", "?", ")", "]", "}", "%", "'", "''"}
_NO_SPACE_AFTER = {"(", "[", "{", "$", "``"}
_CLITICS = {"'s", "n't", "'re", "'ve", "'ll", "'d", "'m", "'t"}
_NUMBER = re.compile(r"^[\d.,]+$")


def normalise_token(token: str, prev: str | None = None) -> str:
    """The surface string a SemCor token is written as in the detokenised text.

    Only quote conventions change: `` and '' become ", and a lone ` becomes ° after a number
    (SemCor uses it as a degree sign: "105 ` C") or ' otherwise. Every other character is kept.
    """
    if token in ("``", "''"):
        return '"'
    if token == "`":
        return "°" if prev is not None and _NUMBER.match(prev) else "'"
    if "``" in token or "''" in token:  # e.g. the single token "''."
        return token.replace("``", '"').replace("''", '"')
    return token


def _attaches_left(token: str, prev: str | None) -> bool:
    """True if `token` is written without a space before it."""
    if token in _NO_SPACE_BEFORE or token.startswith("''"):
        return True
    if token == "`" and prev is not None and _NUMBER.match(prev):  # degree sign
        return True
    return token.lower().split(" ", 1)[0] in _CLITICS  # "'s", "n't", "'ve got", ...


def _blocks_right(token: str, prev: str | None) -> bool:
    """True if the token after `token` is written without a space before it."""
    if token in _NO_SPACE_AFTER:
        return True
    return token == "`" and not (prev is not None and _NUMBER.match(prev))  # opening single quote


def detokenize(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Join SemCor tokens into natural text and return the character span of every token.

    `text[start:end] == normalise_token(token, previous_token)` holds for every token.
    """
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    pos = 0
    prev: str | None = None
    for i, token in enumerate(tokens):
        surface = normalise_token(token, prev)
        if i > 0 and not (_attaches_left(token, prev) or _blocks_right(prev, tokens[i - 2] if i > 1 else None)):
            parts.append(" ")
            pos += 1
        spans.append((pos, pos + len(surface)))
        parts.append(surface)
        pos += len(surface)
        prev = token
    return "".join(parts), spans


# --------------------------------------------------------------------------- dataset
@dataclass
class SemcorDataset:
    """The v2 dataset as loaded by `load_dataset`.

    `instances` preserves corpus order. Each instance has: instance_id, sentence_id, doc_id,
    token_index, word, lemma, pos (n/v/a/r), gold_keys, gold_synsets, candidates (synset names,
    WordNet sense order) and gold_indices (positions of the gold synsets in `candidates`).
    """

    root: Path
    sentences: dict[str, dict]
    instances: dict[str, dict]
    synsets: dict[str, dict]
    splits: dict[str, list[str]]
    doc_split: dict[str, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.doc_split = {doc: name for name, docs in self.splits.items() for doc in docs}

    def split_of(self, instance_id: str) -> str:
        """Name of the split ("dev"/"test") that an instance's document belongs to."""
        return self.doc_split[self.instances[instance_id]["doc_id"]]

    def instances_in_split(self, split: str) -> list[str]:
        """Instance ids of a split, in corpus order."""
        docs = set(self.splits[split])
        return [iid for iid, inst in self.instances.items() if inst["doc_id"] in docs]

    def target_span(self, instance_id: str) -> tuple[int, int]:
        """Character span of the instance's word inside its sentence's `text`."""
        inst = self.instances[instance_id]
        start, end = self.sentences[inst["sentence_id"]]["spans"][inst["token_index"]]
        return start, end


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_dataset(root: str | Path = DEFAULT_ROOT) -> SemcorDataset:
    """Load the files written by `python -m Processing.build_dataset`."""
    root = Path(root)
    if not root.is_absolute() and not root.exists():
        root = REPO / root
    sentences = {s["sentence_id"]: s for s in _read_jsonl(root / "sentences.jsonl")}
    for s in sentences.values():
        s["spans"] = [tuple(span) for span in s["spans"]]
    instances = {i["instance_id"]: i for i in _read_jsonl(root / "instances.jsonl")}
    with open(root / "synsets.json", encoding="utf-8") as f:
        synsets = json.load(f)
    with open(root / "splits.json", encoding="utf-8") as f:
        raw_splits = json.load(f)
    splits = {k: v for k, v in raw_splits.items() if isinstance(v, list)}
    return SemcorDataset(root=root, sentences=sentences, instances=instances, synsets=synsets, splits=splits)


# --------------------------------------------------------------------------- sampling
def _eligible(inst: dict, min_candidates: int) -> bool:
    return len(inst["candidates"]) >= min_candidates


def sample_natural(
    ds: SemcorDataset, split: str, n_sentences: int, seed: int = 0, min_candidates: int = 2
) -> list[str]:
    """All eligible instances of `n_sentences` randomly chosen sentences from the split.

    Only sentences with at least one eligible instance (>= `min_candidates` candidate senses) are
    drawn. Returns instance ids in corpus order. Deterministic for a given seed.
    """
    docs = set(ds.splits[split])
    by_sentence: dict[str, list[str]] = defaultdict(list)
    for iid, inst in ds.instances.items():
        if inst["doc_id"] in docs and _eligible(inst, min_candidates):
            by_sentence[inst["sentence_id"]].append(iid)
    pool = list(by_sentence)  # corpus order, so the draw is reproducible
    chosen = set(random.Random(seed).sample(pool, min(n_sentences, len(pool))))
    return [iid for sid in pool if sid in chosen for iid in by_sentence[sid]]


def sample_balanced(
    ds: SemcorDataset,
    split: str,
    n_instances: int,
    seed: int = 0,
    max_senses_per_lemma: int = 4,
    min_candidates: int = 2,
) -> list[str]:
    """A sense-balanced sample: one instance per gold sense, for lemmas seen with >= 2 senses.

    Considers eligible single-gold instances of the split, grouped by (lemma, pos). Groups with at
    least two distinct gold synsets contribute up to `max_senses_per_lemma` senses, one randomly
    chosen instance each. Whole groups are added in a seeded random order; a group that would push
    the total past `n_instances` is skipped, so the result never exceeds `n_instances` (it is
    exactly `n_instances` unless the split runs out of groups). Returns ids in corpus order.
    """
    rng = random.Random(seed)
    docs = set(ds.splits[split])
    groups: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for iid, inst in ds.instances.items():
        if inst["doc_id"] in docs and _eligible(inst, min_candidates) and len(inst["gold_synsets"]) == 1:
            groups[(inst["lemma"], inst["pos"])][inst["gold_synsets"][0]].append(iid)
    keys = sorted(k for k, senses in groups.items() if len(senses) >= 2)
    rng.shuffle(keys)
    chosen: list[str] = []
    for key in keys:
        if len(chosen) >= n_instances:
            break
        senses = sorted(groups[key])
        rng.shuffle(senses)
        picks = [rng.choice(groups[key][s]) for s in senses[:max_senses_per_lemma]]
        if len(chosen) + len(picks) <= n_instances:
            chosen.extend(picks)
    order = {iid: i for i, iid in enumerate(ds.instances)}
    return sorted(chosen, key=order.__getitem__)


# --------------------------------------------------------------------------- building
def read_gold_keys(path: Path = RAW_KEYS) -> dict[str, list[str]]:
    """instance_id -> list of gold sense keys."""
    keys: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if parts:
                keys[parts[0]] = parts[1:]
    return keys


def make_splits(doc_sources: dict[str, str], seed: int = 0) -> dict:
    """50/50 document split, stratified by Brown genre letter (source "br-e30" -> genre "e").

    Genres with an odd number of documents alternately give their extra document to dev and test,
    in sorted genre order, so the two halves have equal size when possible.
    """
    by_genre: dict[str, list[str]] = defaultdict(list)
    for doc_id, source in doc_sources.items():
        match = re.match(r"^br-([a-z])\d+$", source)
        if not match:
            raise ValueError(f"unexpected document source {source!r} for {doc_id}")
        by_genre[match.group(1)].append(doc_id)
    rng = random.Random(seed)
    dev: list[str] = []
    test: list[str] = []
    extra_to_dev = True
    for genre in sorted(by_genre):
        docs = sorted(by_genre[genre])
        rng.shuffle(docs)
        half = len(docs) // 2
        if len(docs) % 2:
            half += int(extra_to_dev)
            extra_to_dev = not extra_to_dev
        dev.extend(docs[:half])
        test.extend(docs[half:])
    return {"dev": sorted(dev), "test": sorted(test), "seed": seed, "stratified_by": "brown_genre"}


def build(
    raw_xml: Path = RAW_XML, raw_keys: Path = RAW_KEYS, out_root: Path = DEFAULT_ROOT, seed: int = 0
) -> dict:
    """Parse SemCor, write the v2 files under `out_root` and return the build summary."""
    from nltk.corpus import wordnet as wn

    t0 = time.time()
    out_root.mkdir(parents=True, exist_ok=True)
    keys = read_gold_keys(raw_keys)
    tree = ET.parse(raw_xml)

    candidate_cache: dict[tuple[str, str], list[str]] = {}

    def candidates_for(lemma: str, pos: str) -> list[str]:
        if (lemma, pos) not in candidate_cache:
            names: list[str] = []
            for lem in wn.lemmas(lemma, pos=pos):  # filters to synsets containing the lemma; a includes s
                name = lem.synset().name()
                if name not in names:
                    names.append(name)
            candidate_cache[(lemma, pos)] = names
        return candidate_cache[(lemma, pos)]

    sentences: list[dict] = []
    instances: list[dict] = []
    doc_sources: dict[str, str] = {}
    gold_appended: list[str] = []
    xml_pos_mismatch = 0
    for text_el in tree.getroot().findall("text"):
        doc_id, source = text_el.get("id"), text_el.get("source")
        doc_sources[doc_id] = source
        for sent_el in text_el.findall("sentence"):
            sentence_id = sent_el.get("id")
            elements = list(sent_el)
            tokens = [el.text for el in elements]
            text, spans = detokenize(tokens)
            sentences.append(
                {"sentence_id": sentence_id, "doc_id": doc_id, "doc_source": source, "tokens": tokens,
                 "text": text, "spans": spans}
            )
            for idx, el in enumerate(elements):
                if el.tag != "instance":
                    continue
                iid = el.get("id")
                gold_keys = keys[iid]
                lemma, rest = gold_keys[0].split("%", 1)
                pos = SS_TYPE_TO_POS[rest[0]]
                if XML_POS_TO_WN.get(el.get("pos")) != pos:
                    xml_pos_mismatch += 1
                gold_synsets: list[str] = []
                for key in gold_keys:
                    name = wn.lemma_from_key(key).synset().name()
                    if name not in gold_synsets:
                        gold_synsets.append(name)
                candidates = list(candidates_for(lemma, pos))
                for name in gold_synsets:
                    if name not in candidates:
                        candidates.append(name)
                        gold_appended.append(iid)
                instances.append(
                    {"instance_id": iid, "sentence_id": sentence_id, "doc_id": doc_id, "token_index": idx,
                     "word": el.text, "lemma": lemma, "pos": pos, "gold_keys": gold_keys,
                     "gold_synsets": gold_synsets, "candidates": candidates,
                     "gold_indices": [candidates.index(g) for g in gold_synsets]}
                )

    synset_names = sorted({name for inst in instances for name in inst["candidates"]})
    synsets = {}
    for name in synset_names:
        syn = wn.synset(name)
        synsets[name] = {"gloss": syn.definition(), "pos": syn.pos(), "lemmas": syn.lemma_names(),
                         "examples": syn.examples()}
    splits = make_splits(doc_sources, seed=seed)

    with open(out_root / "sentences.jsonl", "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(out_root / "instances.jsonl", "w", encoding="utf-8") as f:
        for inst in instances:
            f.write(json.dumps(inst, ensure_ascii=False) + "\n")
    with open(out_root / "synsets.json", "w", encoding="utf-8") as f:
        json.dump(synsets, f, ensure_ascii=False)
    with open(out_root / "splits.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=1)

    summary = _summary(sentences, instances, synsets, splits, gold_appended, xml_pos_mismatch)
    summary["build_seconds"] = round(time.time() - t0, 1)
    with open(out_root / "build_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _summary(sentences, instances, synsets, splits, gold_appended, xml_pos_mismatch) -> dict:
    n_cands = [len(i["candidates"]) for i in instances]
    doc_split = {d: name for name in ("dev", "test") for d in splits[name]}
    per_split = {}
    for name in ("dev", "test"):
        insts = [i for i in instances if doc_split[i["doc_id"]] == name]
        per_split[name] = {
            "docs": len(splits[name]),
            "sentences": sum(doc_split[s["doc_id"]] == name for s in sentences),
            "instances": len(insts),
            "eligible_instances_(>=2_candidates)": sum(len(i["candidates"]) >= 2 for i in insts),
        }
    return {
        "docs": len(doc_split),
        "sentences": len(sentences),
        "instances": len(instances),
        "synsets": len(synsets),
        "mean_candidates": round(sum(n_cands) / len(n_cands), 3),
        "mean_candidates_polysemous": round(
            sum(n for n in n_cands if n >= 2) / max(1, sum(n >= 2 for n in n_cands)), 3),
        "monosemous_fraction": round(sum(n == 1 for n in n_cands) / len(n_cands), 4),
        "multi_gold_instances": sum(len(i["gold_synsets"]) > 1 for i in instances),
        "gold_appended_to_candidates": len(gold_appended),
        "gold_appended_examples": gold_appended[:10],
        "xml_pos_mismatches": xml_pos_mismatch,
        "pos_distribution": dict(Counter(i["pos"] for i in instances)),
        "gold_is_first_candidate_fraction": round(
            sum(0 in i["gold_indices"] for i in instances) / len(instances), 4),
        "per_split": per_split,
    }
