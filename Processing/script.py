#!/usr/bin/env python3
"""
Script to process a single chunk of data with LLaMA JAX,
generating (and storing) the hidden activations around each
focus word/definition.
"""

import re
import csv
import uuid
import pickle
import shutil
import zipfile
import hashlib
import argparse
from typing import List
import pandas as pd
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

import sys
import os

project_path = os.path.abspath(os.path.join(os.getcwd(), "../LLM"))

if project_path not in sys.path:
    sys.path.append(project_path)

# ------------
# LLaMA import
# ------------
# Make sure your "LLM/llama_jax" directory is in your Python path if needed
# or adjust accordingly:
from llama.tokenizer import Tokenizer
from llama_jax.model import reporting_transformer


# -----------------------------------------------------------------------
# 1. TEXT NORMALIZATION & TOKEN CLIPPING
# -----------------------------------------------------------------------

def fix_whitespace(text: str) -> str:
    """
    Remove redundant spaces around punctuation and add missing spaces after punctuation.
    """
    # Remove spaces before punctuation
    text = re.sub(r'\s+([?.!,;:])', r'\1', text)
    # Ensure space after punctuation if followed by a word (except for e.g. commas in numbers)
    text = re.sub(r'([?.!;:])(?=[^\s])', r'\1 ', text)
    return text


def prep_row(row: pd.Series, tok: Tokenizer) -> pd.Series:
    """
    Pre-process a single DataFrame row:
      - create 'all_defs', 'sentence_str', 'sentence_toks'
      - compute 'word_start_index' for the focus word
      - produce 'definition_prompts' and 'definition_toks'
    """
    row["all_defs"] = str.split(row.definitions, '|')
    # Normalized sentence
    row["sentence_str"] = ' ' + fix_whitespace(' '.join(str.split(row.sentence, '|')))
    row["sentence_toks"] = tok.encode(row["sentence_str"], bos=False, eos=False)

    remaining = row["sentence_str"]
    word_start_index = 0
    for i in range(row["word_loc"]):
        word_piece = tok.decode([row["sentence_toks"][i]])
        word_start_index += remaining.find(word_piece)
        word_start_index += len(word_piece)
        remaining = row["sentence_str"][word_start_index:]

    row["word_start_index"] = word_start_index

    # figure out which token in 'sentence_toks' corresponds to the focus word
    word_tok_index = 0
    length_sum = 0
    while True:
        piece = tok.decode([row["sentence_toks"][word_tok_index]])
        length_sum += len(piece)
        if length_sum > word_start_index:
            break
        word_tok_index += 1

    row["word_tok_index"] = word_tok_index

    # definition tokens
    encode = lambda t: tok.encode(t, bos=False, eos=False)
    # quick check to ensure colon doesn't collapse tokens
    if len(encode(" " + row["word"])) == len(encode(" " + row["word"] + ":")):
        raise Exception("Colon was absorbed - changing token embedding! Check your tokenization logic.")

    defs = row["definitions"].split('|')
    row["definition_prompts"] = [f" {row['word']}: {d}" for d in defs]
    row["definition_toks"] = [encode(def_prompt) for def_prompt in row["definition_prompts"]]

    return row


def clip_definition_token_list(def_toks: List[List[int]], max_tok_len=32) -> List[List[int]]:
    """
    Clip each definition to a max length, truncating from the end.
    """
    ret = []
    for lst in def_toks:
        if len(lst) <= max_tok_len:
            ret.append(lst)
        else:
            ret.append(lst[:max_tok_len])
    return ret


def clip_sentence_token_list(row: pd.Series, max_tok_len=32) -> pd.Series:
    """
    Clip the sentence tokens so that the final list has length <= max_tok_len,
    trying to center on the focus word.
    """
    sentence_toks = row["sentence_toks"]
    word_idx = row["word_tok_index"]

    if len(sentence_toks) <= max_tok_len:
        # No need to clip
        row["clipped_sentence_toks"] = sentence_toks
        row["clipped_word_tok_index"] = word_idx
        return row

    half = max_tok_len // 2
    extra = max_tok_len % 2

    # If focus word is near the start
    if word_idx < half:
        row["clipped_sentence_toks"] = sentence_toks[:max_tok_len]
        row["clipped_word_tok_index"] = word_idx
        return row

    # If focus word is near the end
    n_remaining = len(sentence_toks) - word_idx
    if n_remaining <= half:
        # We need last max_tok_len
        start = len(sentence_toks) - max_tok_len
        row["clipped_sentence_toks"] = sentence_toks[start:]
        # shift word index
        row["clipped_word_tok_index"] = word_idx - start
        return row

    # Otherwise, the focus word is in the middle somewhere
    start = word_idx - half
    end = word_idx + half + extra
    row["clipped_sentence_toks"] = sentence_toks[start:end]
    row["clipped_word_tok_index"] = word_idx - start
    return row


# -----------------------------------------------------------------------
# 2. JAX MODEL + ACTIVATION HELPERS
# -----------------------------------------------------------------------

def pad_toks(toks: List[int], max_len=32) -> (jnp.ndarray, jnp.ndarray):
    """
    Given a list of tokens, pad (or truncate) to length `max_len`.
    Return the padded token array (1, max_len) and attention mask (1, max_len).
    The mask uses 0 for real tokens, -inf for padding tokens.
    """
    to_pad = max_len - len(toks)
    if to_pad < 0:
        # Truncate
        ret = toks[:max_len]
    else:
        ret = toks + [0] * to_pad

    # 0 => real token, -inf => padding
    mask = [0 if x != 0 else -jnp.inf for x in ret]
    return jnp.array(ret)[None, :], jnp.array(mask)[None, :]


def slice_activations_body(activations: jax.Array, indices: jax.Array, width: int) -> jax.Array:
    """
    JITted subroutine to slice a window of 9 tokens (word ± 4) out of the
    [batch, layer, seq_len, hidden_dim] activations. Indices are start positions,
    each shape (batch,).
    """
    bsz, n_layers, seq_len, hidden = activations.shape
    # We'll pad so that we can safely slice negative or out-of-bounds
    # The slice shape will always be (n_layers, 9, hidden).
    # We pad width on both ends of the sequence dimension:
    padded = jnp.pad(activations, ((0, 0), (0, 0), (width, width), (0, 0)), mode='constant')
    # shift indices by `width` to accommodate the left pad
    adj_indices = indices + width

    def extract_slice_for_item(acts, start_pos):
        # dynamic_slice expects [start_layer, start_seq, start_hidden], with shape [L, 9, H]
        return jax.lax.dynamic_slice(acts, (0, start_pos, 0), (n_layers, 9, hidden))

    vmap_extract = jax.vmap(extract_slice_for_item, in_axes=(0, 0), out_axes=0)
    return vmap_extract(padded, jnp.ravel(adj_indices))


slice_activations_updated = jax.jit(slice_activations_body, static_argnames=['width'])


def slice_activations(activations: jax.Array, indices_df: pd.DataFrame, width: int = 4) -> jax.Array:
    """
    Python wrapper to convert a column of positions (DataFrame) into a jnp.array,
    and pass them to `slice_activations_updated` to get the [word-4..word+4] window.
    """
    jax_indices = jnp.array(indices_df.to_numpy())  # shape [batch, 1]
    return slice_activations_updated(activations, jax_indices, width=width)


# -----------------------------------------------------------------------
# 3. MAIN PIPELINE
# -----------------------------------------------------------------------

def transform_dataframe(df: pd.DataFrame, activation_width=4) -> pd.DataFrame:
    """
    For each row in the original df, we produce "records" for:
      - one entry for the sentence
      - one entry per definition
    We store:
      - 'map_index' (original row index in df)
      - the tokens
      - whether it's 'sentence' or 'definition'
      - which definition index
      - a unique hash for these tokens
      - the clipped focus word index
      - 'istart' = clipped_word_tok_index - activation_width (the start of the slice)
    """
    records = []
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Transform DataFrame"):
        sent_toks = row['clipped_sentence_toks']
        tok_index = row['clipped_word_tok_index']
        # Sentence
        rec_sen = (
            idx,               # map_index
            sent_toks,         # toks
            'sentence',        # column
            -1,                # def_index
            hash_tokens(sent_toks, tok_index),
            tok_index,
            tok_index - activation_width,
        )
        records.append(rec_sen)

        # Each definition
        for d_i, d_toks in enumerate(row['clipped_definition_toks']):
            # definitions: focus is at position 0 => center
            # but we store "istart" as (0 - activation_width)
            rec_def = (
                idx,
                d_toks,
                'definition',
                d_i,
                hash_tokens(d_toks, 0),
                0,
                0 - activation_width,
            )
            records.append(rec_def)

    columns = [
        "map_index", "toks", "column", "def_index",
        "hash", "clipped_word_tok_index", "istart"
    ]
    return pd.DataFrame(records, columns=columns)


def hash_tokens(tokens: List[int], index: int) -> str:
    """
    Create a unique hash from the token sequence plus the focus index.
    """
    token_str = ','.join(map(str, tokens)) + str(index)
    return hashlib.sha256(token_str.encode()).hexdigest()


def process_batch(batch_df: pd.DataFrame,
                  params,
                  jitted_transformer,
                  n_heads: int,
                  n_kv_heads: int,
                  max_tok_len: int = 32,
                  slice_width: int = 4) -> pd.DataFrame:
    """
    Runs the forward pass on each item in 'batch_df'.
    Then slices out the [word-4..word+4] region from the final hidden states.
    """
    # Combine the padded_toks and mask into large batch arrays
    batch_toks = jnp.concatenate(batch_df['padded_toks'].tolist(), axis=0)
    batch_mask = jnp.concatenate(batch_df['mask'].tolist(), axis=0)

    # Forward pass
    activations = jitted_transformer(batch_toks, params, batch_mask, n_heads, n_kv_heads)
    # Now slice the relevant region
    sub_acts = slice_activations(activations, batch_df[['istart']], width=slice_width)
    batch_df["model_output"] = list(sub_acts)
    return batch_df


# -----------------------------------------------------------------------
# 4. I/O + STORING ACTIVATIONS
# -----------------------------------------------------------------------

def save_jax_array(array, store_path: str) -> str:
    """
    Save a single JAX array to a pickle file and return a unique key (UUID).
    """
    unique_id = str(uuid.uuid4())
    file_path = os.path.join(store_path, f"{unique_id}.pkl")

    with open(file_path, "wb") as f:
        pickle.dump(array, f)

    return unique_id


def load_jax_array(key: str, store_path: str):
    """
    Load a JAX array from a pickle file using the key.
    """
    file_path = os.path.join(store_path, f"{key}.pkl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JAX array file not found: {file_path}")
    with open(file_path, "rb") as f:
        return pickle.load(f)


def write_results_to_disk(processed_batch_dfs: List[pd.DataFrame],
                          out_csv_path: str,
                          parent_df: pd.DataFrame,
                          store_path: str):
    """
    Write each row from the processed DataFrames to a CSV, storing
    the actual JAX arrays in a separate folder. The CSV references
    them by a unique key.
    """
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    file_exists = os.path.isfile(out_csv_path)

    # CSV fieldnames
    fieldnames = [
        "def_or_sentence", "def_index", "model_output_key", "word",
        "sentence", "word_loc", "wordnet", "definition", "definitions"
    ]

    with open(out_csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for proc_df in tqdm(processed_batch_dfs, desc="Writing to CSV"):
            for _, row in proc_df.iterrows():
                original = parent_df.iloc[row["map_index"]]
                # Save array, get key
                key = save_jax_array(row["model_output"], store_path)

                out_row = {
                    "def_or_sentence": row["column"],
                    "def_index": row["def_index"],
                    "model_output_key": key,
                    "word": original["word"],
                    "sentence": original["sentence"],
                    "word_loc": original["word_loc"],
                    "wordnet": original["wordnet"],
                    "definition": original["definition"],
                    "definitions": original["definitions"]
                }
                writer.writerow(out_row)

        f.flush()


def zip_and_clean_jax_store(store_path: str):
    """
    Compress the store_path directory into a .zip and delete the original files.
    """
    archive_name = store_path.rstrip("/")
    shutil.make_archive(archive_name, 'zip', store_path)
    for filename in os.listdir(store_path):
        file_path = os.path.join(store_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    # Remove empty dir if now empty
    if os.path.exists(store_path) and not os.listdir(store_path):
        os.rmdir(store_path)
    print(f"✅ JAX store compressed to {archive_name}.zip and cleaned up.")


# -----------------------------------------------------------------------
# 5. MAIN FUNCTION
# -----------------------------------------------------------------------

def main():

    print("Processing chunk")

    parser = argparse.ArgumentParser(description="Process a chunk of data with LLaMA JAX.")
    parser.add_argument("--input-csv", required=True, help="Path to input CSV (e.g. chunk_0.csv).")
    parser.add_argument("--output-csv", required=True, help="Where to write the final CSV with references to JAX arrays.")
    parser.add_argument("--chunk-store", required=True, help="Directory where the JAX pickle files will be stored.")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.model")
    parser.add_argument("--model-weights", required=True, help="Pickled LLaMA weights (params_jax_loaded.pkl).")
    parser.add_argument("--max-tok-len", type=int, default=32, help="Maximum token length for sentence/definitions.")
    parser.add_argument("--clip-width", type=int, default=4, help="Width of slice on either side of focus token.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for inference.")
    parser.add_argument("--no-zip", action="store_true", help="If set, do not zip & clean the store directory.")
    args = parser.parse_args()

    # 1. Setup
    os.makedirs(args.chunk_store, exist_ok=True)
    print(f"Loading tokenizer from {args.tokenizer}")
    tok = Tokenizer(args.tokenizer)

    print(f"Loading LLaMA weights from {args.model_weights}")
    with open(args.model_weights, "rb") as f:
        params_jax_loaded = pickle.load(f)

    # For demonstration, these might match your model
    n_heads = 32
    n_kv_heads = 8

    # Make sure the "freqs_cis" is also clipped
    # so it matches the maximum sequence length used
    max_len = args.max_tok_len
    params_jax_loaded["freqs_cis"] = params_jax_loaded["freqs_cis"][:max_len, :]

    # 2. Load chunk CSV
    print(f"Reading input CSV {args.input_csv}")
    df = pd.read_csv(args.input_csv)

    # 3. Preprocess (fix whitespace, tokenize, clip)
    print("Preprocessing rows...")
    tqdm.pandas() 
    df = df.progress_apply(lambda row: prep_row(row, tok), axis=1)
    df["clipped_definition_toks"] = df["definition_toks"].apply(lambda dt: clip_definition_token_list(dt, max_len))
    df = df.apply(lambda r: clip_sentence_token_list(r, max_len), axis=1)

    # 4. Convert to token-based records (one row per sentence or definition)
    print("Transforming DataFrame into token-based records...")
    toks_df = transform_dataframe(df, activation_width=args.clip_width)

    # 5. Deduplicate the token-lists, so we only run the model once per unique set
    unique_toks_df = toks_df.drop_duplicates(subset=["hash"]).reset_index(drop=True)

    # 6. Pad each row’s tokens
    print("Padding tokens for each row (mask included)...")
    unique_toks_df[["padded_toks", "mask"]] = unique_toks_df["toks"].progress_apply(
        lambda t: pad_toks(t, max_len)
    ).apply(pd.Series)

    # 7. Split into batches
    batch_size = args.batch_size
    unique_toks_df["batch_id"] = unique_toks_df.index // batch_size
    grouped = list(unique_toks_df.groupby("batch_id"))

    # 8. Prepare a JIT’d transformer
    print("Compiling JIT’d LLaMA model...")
    jitted_transformer = jax.jit(reporting_transformer, static_argnames=["n_heads", "n_kv_heads"])

    # 9. Process each batch
    processed_batch_dfs = []
    for batch_id, batch_df in tqdm(grouped, desc="Batches"):
        # Run the model, slice the relevant region around the focus word
        out_df = process_batch(
            batch_df.copy(),
            params_jax_loaded,
            jitted_transformer,
            n_heads,
            n_kv_heads,
            max_tok_len=max_len,
            slice_width=args.clip_width
        )
        processed_batch_dfs.append(out_df)

    # 10. Merge results back
    # We now have unique token-lists + hidden states in processed_batch_dfs.
    # We re-map them onto the original `toks_df` (which has duplicates).
    print("Merging batch results back onto the full token-based DataFrame...")
    # Create a small map: hash -> model_output
    big_map = {}
    for proc_df in processed_batch_dfs:
        for idx, row in proc_df.iterrows():
            big_map[(row["hash"], row["clipped_word_tok_index"], row["istart"])] = row["model_output"]

    # Attach model_output to all rows in toks_df
    toks_df["model_output"] = toks_df.apply(
        lambda r: big_map[(r["hash"], r["clipped_word_tok_index"], r["istart"])],
        axis=1
    )

    # 11. Group by batch_id again for final “write to CSV” step
    # (You could do it in one pass, but we do it in small slices if you prefer)
    # For memory reasons, you might want to write out as soon as each batch finishes.
    print(f"Writing out final results to CSV: {args.output_csv}")
    # We can re-group by the same batch_id or simply do single pass:
    final_batch_df_list = [toks_df.iloc[i:i+batch_size] for i in range(0, toks_df.shape[0], batch_size)]
    write_results_to_disk(final_batch_df_list, args.output_csv, df, args.chunk_store)

    # 12. Zip & clean (optional)
    if not args.no_zip:
        print("Zipping & cleaning up the JAX store directory...")
        zip_and_clean_jax_store(args.chunk_store)
    else:
        print("Skipping zipping of JAX store. Done!")


if __name__ == "__main__":
    main()
