"""Shared paths and helpers for the verification scripts.

All scripts are meant to be run from the repository root with the project's
own virtualenv (the one used to produce the original results), e.g. from WSL:

    .venv/bin/python Report/verification/01_data_and_alignment.py

Paths to the Llama checkpoints default to the locations used in the original
notebooks and can be overridden with environment variables.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "Report" / "verification" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

CKPT_ROOT = Path(os.environ.get("LLAMA_CKPT_ROOT", "/home/matt/.llama/checkpoints"))
TOKENIZER_PATH = Path(os.environ.get("LLAMA_TOKENIZER", CKPT_ROOT / "Llama3.2-1B-hf-tok" / "tokenizer.model"))
HF_DIR = Path(os.environ.get("LLAMA_HF_DIR", CKPT_ROOT / "Llama3.2-1B-hf"))
META_DIR = Path(os.environ.get("LLAMA_META_DIR", CKPT_ROOT / "Llama3.2-1B"))
JAX_WEIGHTS = REPO / "Data" / "ModelWeights" / "llama_jax_weights.pkl"

# The chunks that the analysis notebooks actually used.
ANALYSED_CHUNKS = {
    "original_chunk0": {
        "input": REPO / "Data/Processed/SemCoreChunks/chunk_0.csv",
        "processed": REPO / "Data/Processed/ProcessedSemCoreChunks/chunk_0.csv",
        "store": REPO / "Data/Processed/jax_store/chunk_0.zip",
    },
    "uniform_chunk0": {
        "input": REPO / "Data/Processed/SemCoreChunksUniform/chunk_0.csv",
        "processed": REPO / "Data/Processed/ProcessedSemCoreChunksUniform/chunk_0.csv",
        "store": REPO / "Data/Processed/jax_store/uniform_chunk_0.zip",
    },
    "uniform_chunk1": {
        "input": REPO / "Data/Processed/SemCoreChunksUniform/chunk_1.csv",
        "processed": REPO / "Data/Processed/ProcessedSemCoreChunksUniform/chunk_1.csv",
        "store": REPO / "Data/Processed/jax_store/uniform_chunk_1.zip",
    },
    "uniform2_chunk0": {
        "input": REPO / "Data/Processed/SemCoreChunksUniform2/chunk_0.csv",
        "processed": REPO / "Data/Processed/ProcessedSemCoreChunksUniform2/chunk_0.csv",
        "store": REPO / "Data/Processed/jax_store/uniform2_chunk_0.zip",
    },
}


def add_project_to_path() -> None:
    """Make `Processing.script`, `llama.*` and `llama_jax.*` importable."""
    for p in (REPO, REPO / "LLM"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def load_tokenizer():
    add_project_to_path()
    from llama.tokenizer import Tokenizer

    return Tokenizer(str(TOKENIZER_PATH))


L_MARK, R_MARK = "", ""  # private-use chars: untouched by fix_whitespace's regexes


def true_focus_token_index(words: list[str], word_loc: int, toks: list[int], tok):
    """Token index containing the first character of SemCor word `word_loc`.

    Builds the sentence exactly as `Processing/script.py::prep_row` does
    (' ' + fix_whitespace(' '.join(words))) and locates the focus word through
    character offsets rather than by assuming one token per word.

    Returns (index or None, status string).
    """
    add_project_to_path()
    from Processing.script import fix_whitespace

    plain = " " + fix_whitespace(" ".join(words))
    marked_words = list(words)
    marked_words[word_loc] = L_MARK + words[word_loc] + R_MARK
    marked = " " + fix_whitespace(" ".join(marked_words))
    if marked.replace(L_MARK, "").replace(R_MARK, "") != plain:
        return None, "marker_changed_normalisation"
    char_start = marked.index(L_MARK)

    pieces = [tok.decode([t]) for t in toks]
    if "".join(pieces) != plain:
        return None, "per_token_decode_not_lossless"
    pos = 0
    for i, piece in enumerate(pieces):
        if pos <= char_start < pos + len(piece):
            return i, "ok"
        pos += len(piece)
    return None, "not_found"


def save_json(name: str, obj) -> Path:
    path = RESULTS / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)
    return path
