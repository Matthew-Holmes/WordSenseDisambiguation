"""What does the missing <|begin_of_text|> token do inside the *real* model?

Uses HF transformers (float32, causal) only, so it isolates the BOS choice from the JAX bugs.

1. Definition prompts " {word}: {gloss}": the pipeline reads the target word at position 0.
   Without BOS, position 0 becomes the attention sink and carries "massive activations"
   (Sun et al. 2024). We report the ratio of the target token's hidden-state norm to the
   median norm of the other tokens, without BOS (target at position 0) and with BOS
   (target at position 1).
2. Sentences: how similar is the focus-token state with vs without BOS (centred cosine)?

Results: results/02b_bos_effect.json
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from common import ANALYSED_CHUNKS, HF_DIR, add_project_to_path, load_tokenizer, save_json, true_focus_token_index

add_project_to_path()
from Processing import script as pipeline  # noqa: E402

BOS = 128000
N = 24
LAYERS_REPORTED = (0, 2, 5, 8, 12, 15)


def main() -> None:
    from transformers import LlamaForCausalLM

    tok = load_tokenizer()
    model = LlamaForCausalLM.from_pretrained(str(HF_DIR), torch_dtype=torch.float32).model
    model.eval()
    captured = {}
    for k, layer in enumerate(model.layers):
        layer.register_forward_hook(lambda m, i, o, k=k: captured.__setitem__(k, o[0].detach()))

    def run(ids):
        with torch.no_grad():
            model(input_ids=torch.tensor([ids]))
        return np.stack([captured[k][0].numpy() for k in range(16)])  # (16, T, D)

    df = pd.read_csv(ANALYSED_CHUNKS["uniform2_chunk0"]["input"]).head(N)

    # ---- 1. definition prompts: norm of the target-word token relative to the other tokens
    ratios_nobos, ratios_bos = [], []
    for _, row in df.iterrows():
        prepped = pipeline.prep_row(row.copy(), tok)
        for d in prepped["definition_toks"][:3]:
            h0 = run(d)  # target at position 0
            h1 = run([BOS] + d)  # target at position 1
            n0 = np.linalg.norm(h0, axis=-1)  # (16, T)
            n1 = np.linalg.norm(h1, axis=-1)
            ratios_nobos.append(n0[:, 0] / np.median(n0[:, 1:], axis=1))
            ratios_bos.append(n1[:, 1] / np.median(n1[:, 2:], axis=1))
    rn, rb = np.stack(ratios_nobos), np.stack(ratios_bos)

    # ---- 2. sentences: focus-token state with vs without BOS
    v_nb, v_b = [], []
    pool_nb, pool_b = [], []
    for _, row in df.iterrows():
        r = pipeline.prep_row(row.copy(), tok)
        r = pipeline.clip_sentence_token_list(r, 32)
        toks = list(r["clipped_sentence_toks"])
        gt, _ = true_focus_token_index(row["sentence"].split("|"), int(row["word_loc"]), r["sentence_toks"], tok)
        shift = int(r["word_tok_index"]) - int(r["clipped_word_tok_index"])
        f = gt - shift if gt is not None and 0 <= gt - shift < len(toks) else int(r["clipped_word_tok_index"])
        a, b = run(toks), run([BOS] + toks)
        v_nb.append(a[:, f])
        v_b.append(b[:, f + 1])
        pool_nb.append(a[:, 1:])  # exclude position 0 (sink) from the centring statistics
        pool_b.append(b[:, 2:])
    mu_nb = np.concatenate(pool_nb, axis=1).mean(1, keepdims=True)[:, 0]
    mu_b = np.concatenate(pool_b, axis=1).mean(1, keepdims=True)[:, 0]
    A = np.stack(v_nb, 1) - mu_nb[:, None]
    B = np.stack(v_b, 1) - mu_b[:, None]
    ccos = ((A * B).sum(-1) / (np.linalg.norm(A, axis=-1) * np.linalg.norm(B, axis=-1))).mean(1)

    # ---- 3. the stored (JAX, no BOS, bidirectional) definition records: is the target a sink too?
    import pickle
    import zipfile

    proc = pd.read_csv(ANALYSED_CHUNKS["uniform2_chunk0"]["processed"])
    keys = set(proc[proc.def_or_sentence == "definition"].model_output_key.head(200))
    stored_ratio, sent_ratio = [], []
    sent_keys = set(proc[proc.def_or_sentence == "sentence"].model_output_key.head(100))
    with zipfile.ZipFile(ANALYSED_CHUNKS["uniform2_chunk0"]["store"]) as z:
        for name in z.namelist():
            key = name.split("/")[-1][:-4]
            if key not in keys and key not in sent_keys:
                continue
            with z.open(name) as fh:
                arr = np.asarray(pickle.load(fh), dtype=np.float32)  # (16, 9, 2048); window position 4 = focus
            norms = np.linalg.norm(arr, axis=-1)  # (16, 9)
            others = norms[:, 5:9]  # the 4 tokens after the focus (definition text / right context)
            ok = others.min(axis=1) > 0
            if not ok.all():
                continue
            (stored_ratio if key in keys else sent_ratio).append(norms[:, 4] / np.median(others, axis=1))
    sr = np.stack(stored_ratio)
    se = np.stack(sent_ratio)

    res = {
        "stored_activations_definition_target_norm_over_next_4_tokens": {
            f"L{l}": float(np.median(sr[:, l])) for l in LAYERS_REPORTED
        },
        "stored_activations_sentence_focus_norm_over_next_4_tokens": {
            f"L{l}": float(np.median(se[:, l])) for l in LAYERS_REPORTED
        },
        "n_stored_definition_records": len(sr),
        "n_stored_sentence_records": len(se),
        "n_definition_prompts": len(rn),
        "target_norm_over_median_other_tokens": {
            "without_bos_target_at_position_0": {f"L{l}": float(np.median(rn[:, l])) for l in LAYERS_REPORTED},
            "with_bos_target_at_position_1": {f"L{l}": float(np.median(rb[:, l])) for l in LAYERS_REPORTED},
        },
        "n_sentences": N,
        "sentence_focus_centred_cos_with_vs_without_bos": {f"L{l}": float(ccos[l]) for l in LAYERS_REPORTED},
    }
    print(res)
    save_json("02b_bos_effect.json", res)


if __name__ == "__main__":
    main()
