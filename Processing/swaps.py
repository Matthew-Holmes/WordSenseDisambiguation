"""Counterfactual ("swapped") sentences: the target word replaced by a random word of the same part of speech.

Idea (Report/SWAP_STUDY.md): a target word's hidden state mixes what the word means in this sentence with what
its slot in the sentence encodes (part of speech, syntactic role, position). Running the sentence again with the
word replaced by an unrelated word of the same part of speech keeps the slot but changes the meaning, so comparing
the two hidden states (Processing/contrastive.py) is meant to cancel the shared, grammatical part.

Replacement words
    Drawn uniformly at random from the distinct surface forms of sense-tagged words anywhere in the v2 dataset
    that have the required part of speech (WordNet n/v/a/r). Only lower-case alphabetic single words are used
    (hyphens and apostrophes allowed inside), which keeps out names, numbers and multiword expressions. Every
    form of the target's own lemma is excluded; accidental synonyms are possible and accepted.

    match="pos" (default) is the design as specified. match="pos+form" also matches a coarse inflection class
    read off the target's form vs its lemma (plural noun, verb -s/-ed/-ing/irregular, comparative or
    superlative adjective), so that, e.g., "reviewed" is replaced by another -ed verb.

    The replacement copies the target's capitalisation ("How" -> "Ripe", "NASA" -> "TOAD"). A target written
    against a neighbouring word (the clitic "'s" in "It's", "was" in "wasn't") gets a space on that side, so
    "It's the old story" becomes "It disputed the old story" rather than "Itdisputed ...".

One swap per (target, part of speech, draw)
    A swap is made for each part of speech among the target's candidate senses, so that each candidate is
    compared against a slot holding a word of its own part of speech. In the v2 dataset every candidate has the
    target's part of speech (Report bug B14), so this is one swapped sentence per target per draw.

Determinism
    Each draw is seeded by (seed, instance id, part of speech, draw index, match mode) alone, so a swap does not
    depend on which other targets are in the run, and adding draws later leaves earlier ones unchanged.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from Processing.prompts import SentenceInput, build_sentence_input

POOL_VERSION = 1  # bump when the pool rules change; stored with the vectors (Processing/script.py)
MATCH_MODES = ("pos", "pos+form")
MIN_FORM_POOL = 50  # pos+form falls back to the whole POS pool when an inflection class has fewer forms
_WORD = re.compile(r"[a-z]+(?:[-'][a-z]+)*")


# ------------------------------------------------------------------------------- parts of speech
def synset_pos(synset: str) -> str:
    """WordNet POS letter of a synset name: 'long.a.01' -> 'a'. Satellites count as adjectives ('s' -> 'a')."""
    pos = synset.rsplit(".", 2)[1]
    return "a" if pos == "s" else pos


def needed_pos(candidates: Sequence[str]) -> list[str]:
    """Distinct parts of speech among an instance's candidate synsets, in first-seen order."""
    return list(dict.fromkeys(synset_pos(c) for c in candidates))


def word_form(word: str, lemma: str, pos: str) -> str:
    """Coarse inflection class of a surface form, from how it differs from its lemma (for match="pos+form").

    Nouns: base / plural. Verbs: base / s / ed / ing / irregular. Adjectives and adverbs: base / comparative /
    superlative / irregular. A heuristic: e.g. "was" counts as an -s verb and "children" as a plural.
    """
    w, base = word.lower(), lemma.replace("_", " ").lower()
    if w == base:
        return "base"
    if pos == "n":
        return "plural"
    if pos == "v":
        for suffix in ("ing", "ed", "s"):
            if w.endswith(suffix):
                return suffix
        return "irregular"
    if w.endswith("est"):
        return "superlative"
    if w.endswith("er"):
        return "comparative"
    return "irregular"


# ------------------------------------------------------------------------------------------- pool
@dataclass(frozen=True)
class ReplacementPool:
    """Candidate replacement words, read from the sense-tagged words of a dataset.

    by_pos       : POS -> sorted distinct surface forms usable as replacements
    by_pos_form  : (POS, inflection class) -> sorted forms (for match="pos+form")
    lemma_forms  : (lemma, POS) -> every lower-cased form of that lemma seen in the data (to exclude)
    """

    by_pos: dict[str, list[str]]
    by_pos_form: dict[tuple[str, str], list[str]]
    lemma_forms: dict[tuple[str, str], frozenset[str]]

    def sizes(self) -> dict[str, int]:
        return {pos: len(words) for pos, words in sorted(self.by_pos.items())}


def usable_form(word: str) -> bool:
    """True for a lower-case alphabetic single word of two or more letters ('well-known' and "o'clock" pass)."""
    return len(word) > 1 and _WORD.fullmatch(word) is not None


def build_pool(instances: Iterable[dict]) -> ReplacementPool:
    """The replacement pool for a collection of instances (pass ``ds.instances.values()`` for the whole dataset)."""
    by_pos: dict[str, set[str]] = defaultdict(set)
    by_pos_form: dict[tuple[str, str], set[str]] = defaultdict(set)
    lemma_forms: dict[tuple[str, str], set[str]] = defaultdict(set)
    for inst in instances:
        word, lemma, pos = inst["word"], inst["lemma"], inst["pos"]
        lemma_forms[(lemma, pos)].add(word.lower())
        if usable_form(word):
            by_pos[pos].add(word)
            by_pos_form[(pos, word_form(word, lemma, pos))].add(word)
    return ReplacementPool(
        by_pos={k: sorted(v) for k, v in by_pos.items()},
        by_pos_form={k: sorted(v) for k, v in by_pos_form.items()},
        lemma_forms={k: frozenset(v) for k, v in lemma_forms.items()},
    )


def excluded_forms(pool: ReplacementPool, word: str, lemma: str, pos: str) -> set[str]:
    """Forms a target may not be replaced by: its own word, its lemma and every form of that lemma."""
    return set(pool.lemma_forms.get((lemma, pos), ())) | {word.lower(), lemma.replace("_", " ").lower()}


def draw_replacement(
    pool: ReplacementPool, word: str, lemma: str, pos: str, rng: random.Random, match: str = "pos"
) -> str:
    """A replacement drawn uniformly from the allowed forms of ``pos`` (see the module docstring)."""
    if match not in MATCH_MODES:
        raise ValueError(f"match must be one of {MATCH_MODES}, got {match!r}")
    options = pool.by_pos.get(pos, [])
    if match == "pos+form":
        same_form = pool.by_pos_form.get((pos, word_form(word, lemma, pos)), [])
        if len(same_form) >= MIN_FORM_POOL:
            options = same_form
    excluded = excluded_forms(pool, word, lemma, pos)
    for _ in range(1000):  # rejection sampling: uniform over the allowed forms
        pick = options[rng.randrange(len(options))] if options else None
        if pick is not None and pick not in excluded:
            return pick
    allowed = [w for w in options if w not in excluded]
    if not allowed:
        raise ValueError(f"no replacement available for {word!r} ({lemma}, {pos})")
    return allowed[rng.randrange(len(allowed))]


# ------------------------------------------------------------------------------------- text edits
def match_case(word: str, template: str) -> str:
    """``word`` written with ``template``'s capitalisation: all caps, first letter capital, or unchanged."""
    if len(template) > 1 and template.isupper():
        return word.upper()
    if template[:1].isupper():
        return word[:1].upper() + word[1:]
    return word


def replace_span(text: str, span: tuple[int, int], replacement: str) -> tuple[str, tuple[int, int]]:
    """``text`` with characters [start, end) replaced, and the replacement's span in the new text."""
    start, end = span
    if not 0 <= start < end <= len(text):
        raise ValueError(f"span {span} is not inside a text of length {len(text)}")
    return text[:start] + replacement + text[end:], (start, start + len(replacement))


# ------------------------------------------------------------------------------------------ swaps
@dataclass(frozen=True)
class Swap:
    key: str  # vector-store key, see swap_key
    instance_id: str
    sentence_id: str
    pos: str  # part of speech of the replacement (= of the candidates it serves)
    draw: int
    match: str
    original: str  # the target as written in the sentence
    replacement: str  # as written in the swapped sentence (capitalisation matched)
    text: str  # the swapped sentence
    span: tuple[int, int]  # character span of the replacement in `text`


def swap_key(instance_id: str, pos: str, draw: int, match: str = "pos") -> str:
    """Store key of one swapped sentence, e.g. 'd000.s000.t000~swap:pos:a:0'."""
    return f"{instance_id}~swap:{match}:{pos}:{draw}"


def make_swap(ds, pool: ReplacementPool, instance_id: str, pos: str, draw: int, seed: int = 0,
              match: str = "pos") -> Swap:
    """The swapped sentence for one target, part of speech and draw (deterministic, see the module docstring)."""
    inst = ds.instances[instance_id]
    text = ds.sentences[inst["sentence_id"]]["text"]
    span = ds.target_span(instance_id)
    original = text[span[0]: span[1]]
    rng = random.Random(f"{seed}|{instance_id}|{pos}|{draw}|{match}")
    # inflection matching only makes sense in the target's own part of speech
    mode = match if pos == inst["pos"] else "pos"
    word = draw_replacement(pool, inst["word"], inst["lemma"], pos, rng, mode)
    replacement = match_case(word, original)
    # clitic targets ("'s" in "It's", "was" in "wasn't") touch a neighbouring word: keep the replacement apart
    left = int(span[0] > 0 and text[span[0] - 1].isalnum())
    right = int(span[1] < len(text) and text[span[1]].isalnum())
    new_text, (a, b) = replace_span(text, span, " " * left + replacement + " " * right)
    new_span = (a + left, b - right)
    return Swap(swap_key(instance_id, pos, draw, match), instance_id, inst["sentence_id"], pos, draw, match,
                original, replacement, new_text, new_span)


def plan_swaps(ds, instance_ids: Iterable[str], draws: Sequence[int] = (0,), seed: int = 0, match: str = "pos",
               pool: ReplacementPool | None = None) -> list[Swap]:
    """Every swap needed to score ``instance_ids``: one per (instance, candidate POS, draw), in that order."""
    pool = pool or build_pool(ds.instances.values())
    return [make_swap(ds, pool, iid, pos, j, seed, match)
            for iid in instance_ids for pos in needed_pos(ds.instances[iid]["candidates"]) for j in draws]


def swap_inputs(tok, swaps: Iterable[Swap]) -> list[SentenceInput]:
    """Echo inputs (Processing/prompts.py) for swapped sentences; each has one target, keyed by the swap key."""
    return [build_sentence_input(tok, sw.key, sw.text, {sw.key: sw.span}) for sw in swaps]


def save_plan(path: str | Path, swaps: Iterable[Swap]) -> None:
    """Write swaps as JSON, merged with any already in the file (keyed by swap key)."""
    path = Path(path)
    merged = {sw.key: sw for sw in load_plan(path)} if path.exists() else {}
    merged.update({sw.key: sw for sw in swaps})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps([asdict(sw) for sw in merged.values()], indent=0))
    tmp.replace(path)


def load_plan(path: str | Path) -> list[Swap]:
    return [Swap(**{**d, "span": tuple(d["span"])}) for d in json.loads(Path(path).read_text())]
