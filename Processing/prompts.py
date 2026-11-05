"""Prompt construction and target alignment for the v2 pipeline.

Fixes from Report/REPORT.md, by construction:
  B1   targets are located by character offsets (tiktoken `decode_with_offsets`), never by word index
  B6   every sequence starts with <|begin_of_text|>
  B12  sequences are right-padded and masked by length (see Processing/extract.py), never by token value
  B13  nothing here can raise on unusual words: no colon check, no NaN-prone CSV columns
  B15  sentences come from Processing/semcor_data.detokenize, which never rewrites a token's characters

Sentence inputs use "echo" (Springer et al. 2024): ``[BOS] text + " " + text``. With causal attention a
token only sees what precedes it, so for each target:
  * ``plain`` = its sub-tokens in the first copy (left context only; identical to a plain pass), and
  * ``echo``  = its sub-tokens in the second copy (the model has already read the whole sentence).
One forward pass per sentence serves every target in it.

Gloss inputs are ``[BOS] "{lemma}: {gloss}"``; the gloss representation is pooled over the gloss tokens.
The target word must not be read at position 0 of a definition prompt: under causal attention that
position cannot see the definition (and without BOS it becomes the attention sink).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from LLM.llama_jax.model import BOS_ID


@dataclass
class SentenceInput:
    sentence_id: str
    token_ids: list[int]
    # instance_id -> {"plain": (first, last), "echo": (first, last)}; inclusive token indices into token_ids
    targets: dict[str, dict[str, tuple[int, int]]] = field(default_factory=dict)


@dataclass
class GlossInput:
    key: str  # gloss_key(lemma, synset)
    token_ids: list[int]
    span: tuple[int, int]  # [start, end) token indices of the gloss text in token_ids


def encode_with_offsets(tok, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Token ids for `text` (no BOS) and each token's [start, end) character span in `text`."""
    ids = tok.encode(text, bos=False, eos=False)
    decoded, starts = tok.model.decode_with_offsets(ids)
    if decoded != text:
        raise ValueError(f"tokenizer round trip failed for {text!r}")
    ends = list(starts[1:]) + [len(text)]
    return ids, list(zip(starts, ends))


def token_range(spans: list[tuple[int, int]], start: int, end: int) -> tuple[int, int]:
    """Inclusive (first, last) indices of the tokens overlapping characters [start, end)."""
    hits = [i for i, (s, e) in enumerate(spans) if s < end and e > start]
    if not hits:
        raise ValueError(f"no token overlaps characters [{start}, {end})")
    return hits[0], hits[-1]


def build_sentence_input(tok, sentence_id: str, text: str, targets: dict[str, tuple[int, int]]) -> SentenceInput:
    """Echo input for one sentence. `targets` maps instance_id -> character span of the word in `text`."""
    echo_text = f"{text} {text}"
    ids, spans = encode_with_offsets(tok, echo_text)
    second = len(text) + 1
    out = SentenceInput(sentence_id, [BOS_ID] + ids)
    for instance_id, (s, e) in targets.items():
        p_first, p_last = token_range(spans, s, e)
        e_first, e_last = token_range(spans, second + s, second + e)
        if spans[p_last][1] > second or spans[e_first][0] < len(text):
            raise ValueError(f"{instance_id}: target tokens straddle the two copies")
        out.targets[instance_id] = {"plain": (p_first + 1, p_last + 1), "echo": (e_first + 1, e_last + 1)}
    return out


def gloss_text(lemma: str, gloss: str) -> tuple[str, int]:
    """The definition prompt and the character offset where the gloss starts."""
    prefix = f"{lemma.replace('_', ' ')}:"
    return f"{prefix} {gloss}", len(prefix)


def build_gloss_input(tok, lemma: str, synset: str, gloss: str) -> GlossInput:
    """Definition prompt for one candidate sense of `lemma` (WordNet lemma name, underscores allowed)."""
    text, gloss_start = gloss_text(lemma, gloss)
    ids, spans = encode_with_offsets(tok, text)
    first, last = token_range(spans, gloss_start, len(text))
    return GlossInput(gloss_key(lemma, synset), [BOS_ID] + ids, (first + 1, last + 2))


def gloss_key(lemma: str, synset: str) -> str:
    """Gloss vectors depend on the lemma shown in the prompt, so they are cached per (lemma, synset)."""
    return f"{lemma}|{synset}"
