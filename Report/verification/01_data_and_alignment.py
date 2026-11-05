"""Verify the data/prompt side of the pipeline.

Checks performed (each prints a summary and is saved to results/01_*.json):

1. Focus-token alignment. `Processing/script.py::prep_row` walks the first
   `word_loc` *LLM tokens* to locate the focus word, but `word_loc` is an index
   into the SemCor *word* list. We compute the true token index from character
   offsets and compare it with what the pipeline actually used.
2. Token id 0 ("!") is used as the padding id, and the attention mask is
   derived from `token != 0`, so genuine "!" tokens are masked out.
3. `fix_whitespace` rewrites decimals/abbreviations ("3.5" -> "3. 5").
4. Clipping statistics (sentences / definitions longer than 32 tokens).
5. Dataset properties that affect the metric: gold definition present in the
   candidate list, duplicate glosses, candidates spanning several parts of
   speech, and the most-frequent-sense / list-order baseline.
"""

from __future__ import annotations

import random
import re
from collections import Counter

import pandas as pd

from common import (
    ANALYSED_CHUNKS,
    REPO,
    add_project_to_path,
    load_tokenizer,
    save_json,
    true_focus_token_index,
)

add_project_to_path()
from Processing import script as pipeline  # noqa: E402  (the real pipeline code)


def split_candidates(definitions: str) -> list[str]:
    """Uniform2 data was written with ' | ' separators; everything else uses '|'."""
    return definitions.split(" | ") if " | " in definitions else definitions.split("|")


def safe_prep(row, tok):
    """prep_row raises on some words (colon absorbed into the word token)."""
    try:
        return pipeline.prep_row(row.copy(), tok), None
    except Exception as e:  # noqa: BLE001 - we want to count exactly what the pipeline would hit
        return None, str(e)


def alignment_report(df: pd.DataFrame, tok, examples: int = 8) -> dict:
    n = len(df)
    status = Counter()
    mismatches = []
    script_first_tok_ok = 0
    true_first_tok_ok = 0
    evaluated = 0
    crashed_words = Counter()
    for _, row in df.iterrows():
        prepped, err = safe_prep(row, tok)
        if prepped is None:
            crashed_words[row["word"]] += 1
            status["prep_row_raised"] += 1
            continue
        script_idx = int(prepped["word_tok_index"])
        toks = prepped["sentence_toks"]
        words = row["sentence"].split("|")
        gt_idx, st = true_focus_token_index(words, int(row["word_loc"]), toks, tok)
        status[st] += 1
        if gt_idx is None:
            continue
        evaluated += 1
        first_word_tok = tok.encode(" " + row["word"], bos=False, eos=False)[0]
        script_first_tok_ok += int(script_idx < len(toks) and toks[script_idx] == first_word_tok)
        true_first_tok_ok += int(toks[gt_idx] == first_word_tok)
        if script_idx != gt_idx:
            mismatches.append(
                {
                    "word": row["word"],
                    "sentence": " ".join(words),
                    "word_loc": int(row["word_loc"]),
                    "script_token_index": script_idx,
                    "script_token": tok.decode([toks[script_idx]]) if script_idx < len(toks) else None,
                    "true_token_index": gt_idx,
                    "true_token": tok.decode([toks[gt_idx]]),
                }
            )
    return {
        "rows": n,
        "rows_evaluated": evaluated,
        "status_counts": dict(status),
        "misaligned_rows": len(mismatches),
        "misaligned_fraction": len(mismatches) / max(evaluated, 1),
        "script_token_equals_first_subtoken_of_word": script_first_tok_ok / max(evaluated, 1),
        "true_token_equals_first_subtoken_of_word": true_first_tok_ok / max(evaluated, 1),
        "mean_abs_offset_when_misaligned": (
            sum(abs(m["script_token_index"] - m["true_token_index"]) for m in mismatches) / len(mismatches)
            if mismatches
            else 0.0
        ),
        "script_token_before_true_token_fraction": (
            sum(m["script_token_index"] < m["true_token_index"] for m in mismatches) / len(mismatches)
            if mismatches
            else 0.0
        ),
        "rows_where_prep_row_raises": sum(crashed_words.values()),
        "words_that_make_prep_row_raise": dict(crashed_words.most_common(20)),
        "examples": random.Random(0).sample(mismatches, min(examples, len(mismatches))),
    }


def prompt_side_effects(df: pd.DataFrame, tok, max_len: int = 32) -> dict:
    """Bang-token masking, whitespace normalisation damage, clipping."""
    bang_id = tok.encode("!", bos=False, eos=False)
    assert bang_id == [0], bang_id
    sentences_with_bang = 0
    defs_with_bang = 0
    n_defs = 0
    sentences_clipped = 0
    defs_clipped = 0
    focus_word_lost_by_clip = 0
    decimal_or_abbrev_damaged = 0
    word_multi_token = 0
    stray_space_defs = 0
    for _, row in df.iterrows():
        prepped, _ = safe_prep(row, tok)
        if prepped is None:
            continue
        toks = prepped["sentence_toks"]
        sentences_with_bang += int(0 in toks)
        # definition prompts with a doubled space after the colon or a trailing space
        stray_space_defs += sum(p.endswith(" ") or ":  " in p for p in prepped["definition_prompts"])
        sentences_clipped += int(len(toks) > max_len)
        raw = " ".join(row["sentence"].split("|"))
        # fix_whitespace inserts a space after . ; : ! ? when followed by a non-space char
        decimal_or_abbrev_damaged += int(bool(re.search(r"[?.!;:](?=[^\s])", raw)))
        word_multi_token += int(len(tok.encode(" " + row["word"], bos=False, eos=False)) > 1)
        for d in prepped["definition_toks"]:
            n_defs += 1
            defs_with_bang += int(0 in d)
            defs_clipped += int(len(d) > max_len)
    n = len(df)
    return {
        "rows": n,
        "sentences_containing_token0_bang_masked_as_padding": sentences_with_bang / n,
        "definition_prompts_containing_token0": defs_with_bang / max(n_defs, 1),
        "sentences_longer_than_32_tokens_clipped": sentences_clipped / n,
        "definition_prompts_longer_than_32_tokens_truncated": defs_clipped / max(n_defs, 1),
        "sentences_altered_by_fix_whitespace_insertion_rule": decimal_or_abbrev_damaged / n,
        "focus_words_spanning_multiple_tokens": word_multi_token / n,
        "definition_prompts_with_stray_whitespace": stray_space_defs / max(n_defs, 1),
        "fix_whitespace_examples": {
            s: pipeline.fix_whitespace(s) for s in ["It costs 3.5 million .", "The U.S. army", "At 10:30 p.m. today"]
        },
    }


def dataset_properties(df: pd.DataFrame, check_pos: bool = True) -> dict:
    """Properties of the candidate lists and the list-order (MFS-like) baseline."""
    from nltk.corpus import wordnet as wn

    n = len(df)
    gold_in = 0
    gold_in_exact_pipe_split = 0
    gold_dupe = 0
    n_cands = []
    list_order_norm_rank = []
    first_is_gold = 0
    mixed_pos = 0
    gold_pos_differs_from_some_candidate = 0
    for _, row in df.iterrows():
        gold_in_exact_pipe_split += int(row["definition"] in row["definitions"].split("|"))
        cands = split_candidates(row["definitions"])
        n_cands.append(len(cands))
        if row["definition"] in cands:
            gold_in += 1
            idx = cands.index(row["definition"])
            gold_dupe += int(cands.count(row["definition"]) > 1)
            first_is_gold += int(idx == 0)
            if len(cands) > 1:
                list_order_norm_rank.append(1 - idx / (len(cands) - 1))
        if check_pos:
            key = row["wordnet"].split(" ")[0]
            lemma = key.split("%")[0]
            def_pos: dict[str, set] = {}
            for s in wn.synsets(lemma):
                def_pos.setdefault(s.definition(), set()).add(s.pos().replace("s", "a"))
            cand_pos = set().union(*(def_pos.get(c, set()) for c in cands)) if cands else set()
            mixed_pos += int(len(cand_pos) > 1)
            gold_pos = wn.lemma_from_key(key).synset().pos().replace("s", "a")
            gold_pos_differs_from_some_candidate += int(bool(cand_pos - {gold_pos}))
    s = pd.Series(n_cands)
    return {
        "rows": n,
        "gold_definition_in_candidates_with_pipeline_split('|')": gold_in_exact_pipe_split / n,
        "gold_definition_in_candidates": gold_in / n,
        "gold_definition_duplicated_in_candidates": gold_dupe / n,
        "candidates_mean": float(s.mean()),
        "candidates_median": float(s.median()),
        "rows_with_single_candidate": float((s == 1).mean()),
        "gold_is_first_listed_candidate": first_is_gold / n,
        "list_order_baseline_mean_normalised_rank": (
            sum(list_order_norm_rank) / len(list_order_norm_rank) if list_order_norm_rank else None
        ),
        "rows_whose_candidates_span_multiple_POS": (mixed_pos / n) if check_pos else None,
        "rows_with_a_candidate_of_a_different_POS_to_gold": (
            gold_pos_differs_from_some_candidate / n if check_pos else None
        ),
    }


def chunks_that_would_crash(full: pd.DataFrame, tok, chunk_size: int = 100) -> dict:
    """script.py raises on the 'colon absorbed' check, killing the whole chunk job.

    pandas also parses the SemCor word "null" as NaN, which makes prep_row raise a TypeError.
    """
    encode = lambda t: tok.encode(t, bos=False, eos=False)  # noqa: E731
    words = full["word"]
    nan_row = words.isna().to_numpy()
    bad_words = {w for w in words.dropna().unique() if len(encode(" " + w)) == len(encode(" " + w + ":"))}
    colon_row = words.isin(bad_words).to_numpy()
    bad_row = nan_row | colon_row
    n_chunks = (len(full) + chunk_size - 1) // chunk_size
    crashing = sum(bad_row[i * chunk_size : (i + 1) * chunk_size].any() for i in range(n_chunks))
    return {
        "distinct_words_triggering_exception": sorted(bad_words)[:50],
        "n_distinct_words_triggering_exception": len(bad_words),
        "rows_triggering_colon_exception": int(colon_row.sum()),
        "rows_with_word_parsed_as_NaN": int(nan_row.sum()),
        "rows_triggering_exception": int(bad_row.sum()),
        "chunks_of_100_that_would_crash": int(crashing),
        "total_chunks_of_100": int(n_chunks),
    }


def main() -> None:
    tok = load_tokenizer()
    out = {}

    for name, paths in ANALYSED_CHUNKS.items():
        df = pd.read_csv(paths["input"])
        print(f"\n=== {name}: {len(df)} rows")
        rep = {
            "alignment": alignment_report(df, tok),
            "prompt_side_effects": prompt_side_effects(df, tok),
            "dataset": dataset_properties(df),
        }
        a = rep["alignment"]
        print(
            f"misaligned focus token: {a['misaligned_rows']}/{a['rows_evaluated']} "
            f"({a['misaligned_fraction']:.1%}); status={a['status_counts']}"
        )
        print({k: v for k, v in rep["prompt_side_effects"].items() if k != "fix_whitespace_examples"})
        print(rep["dataset"])
        out[name] = rep

    # Corpus-wide estimate on a random sample of the full processed SemCor data.
    full = pd.read_csv(REPO / "Data/Processed/SemCoreProcessed.csv")
    sample = full.sample(n=20000, random_state=0)
    print(f"\n=== full SemCor: {len(full)} rows, sampling {len(sample)}")
    rep = {
        "alignment": alignment_report(sample, tok, examples=20),
        "prompt_side_effects": prompt_side_effects(sample, tok),
        "dataset": dataset_properties(sample),
    }
    a = rep["alignment"]
    print(
        f"misaligned focus token: {a['misaligned_rows']}/{a['rows_evaluated']} "
        f"({a['misaligned_fraction']:.1%}); status={a['status_counts']}"
    )
    print({k: v for k, v in rep["prompt_side_effects"].items() if k != "fix_whitespace_examples"})
    print(rep["dataset"])
    out["full_semcor_sample20k"] = rep
    out["full_semcor_rows"] = len(full)
    out["full_semcor_crash_check"] = chunks_that_would_crash(full, tok)
    print(out["full_semcor_crash_check"])
    for name in ("SemCoreProcessedUniform.csv", "SemCoreProcessedUniform2.csv"):
        d = pd.read_csv(REPO / "Data/Processed" / name)
        out[f"{name}_crash_check"] = chunks_that_would_crash(d, tok)
        out[f"{name}_rows"] = len(d)
        print(name, len(d), out[f"{name}_crash_check"]["chunks_of_100_that_would_crash"])

    path = save_json("01_data_and_alignment.json", out)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
