"""Parity of the JAX Llama against HuggingFace transformers (slow: loads both models, ~8 GB RAM).

Run with:  WSD_SLOW_TESTS=1 .venv/bin/python -m tests.run_tests model_parity

Thresholds: centred cosine >= 0.99 at every layer (the original pipeline scored 0.06 at layer 8),
next-token top-1 agreement >= 0.95 (bfloat16 vs float32), plus padding invariance and causality.
"""

from __future__ import annotations

import os
import unittest
from functools import lru_cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np

SLOW = True

CKPT = Path(os.environ.get("LLAMA_CKPT_ROOT", "/home/matt/.llama/checkpoints"))
HF_DIR = Path(os.environ.get("LLAMA_HF_DIR", CKPT / "Llama3.2-1B-hf"))
TOKENIZER = Path(os.environ.get("LLAMA_TOKENIZER", CKPT / "Llama3.2-1B-hf-tok" / "tokenizer.model"))
WEIGHTS = Path(os.environ.get("WSD_JAX_WEIGHTS", Path(__file__).resolve().parents[1] / "Data/ModelWeights/llama_jax_weights.pkl"))

SENTENCES = [
    "The capital of France is",
    "He sat on the bank of the river and watched the water.",
    "She deposited the cheque at the bank before it closed!",
    "How long has it been since you reviewed the objectives of your benefit and service program?",
    "Oh, yeah, I said.",
    "The model quite plainly thought Michelangelo could do no wrong, and said so.",
]


@lru_cache(maxsize=1)
def _setup():
    for p in (HF_DIR, TOKENIZER, WEIGHTS):
        if not p.exists():
            raise unittest.SkipTest(f"missing {p}")
    import torch
    from transformers import LlamaForCausalLM

    from LLM.llama.tokenizer import Tokenizer
    from LLM.llama_jax import model as m
    from LLM.llama_jax.weights import load_params

    tok = Tokenizer(str(TOKENIZER))
    ids = [tok.encode(s, bos=True, eos=False) for s in SENTENCES]

    hf = LlamaForCausalLM.from_pretrained(str(HF_DIR), torch_dtype=torch.float32)
    hf.eval()
    captured = {}
    hooks = [
        layer.register_forward_hook(lambda mod, inp, out, k=k: captured.__setitem__(k, out[0].detach()))
        for k, layer in enumerate(hf.model.layers)
    ]
    ref_states, ref_top1 = [], []
    with torch.no_grad():
        for t in ids:
            out = hf(torch.tensor([t]), output_hidden_states=True)
            emb = out.hidden_states[0][0].numpy()
            blocks = [captured[k][0].numpy() for k in range(len(hf.model.layers))]  # pre-final-norm
            ref_states.append(np.stack([emb] + blocks))  # (17, T, D)
            ref_top1.append(out.logits[0].argmax(-1).numpy())
    for h in hooks:
        h.remove()
    del hf

    params, args = load_params(WEIGHTS)
    T = max(len(t) for t in ids)
    tokens = np.full((len(ids), T), m.PAD_ID, dtype=np.int32)
    for i, t in enumerate(ids):
        tokens[i, : len(t)] = t
    lengths = jnp.array([len(t) for t in ids])
    mask = m.padding_mask(lengths, T)
    states = np.asarray(
        m.hidden_states_jit(jnp.asarray(tokens), params, mask, args.n_heads, args.n_kv_heads), dtype=np.float32
    )
    logits = np.asarray(m.transformer_jit(jnp.asarray(tokens), params, mask, args.n_heads, args.n_kv_heads))
    return dict(m=m, params=params, args=args, ids=ids, tokens=tokens, ref_states=ref_states, ref_top1=ref_top1,
                states=states, logits=logits)


def _cos(a, b):
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)


def test_hidden_states_match_hf_every_layer():
    s = _setup()
    # pool real positions after BOS (position 0 carries the attention-sink massive activations)
    ours = np.concatenate([s["states"][i, :, 1 : len(t)] for i, t in enumerate(s["ids"])], axis=1)
    ref = np.concatenate([r[:, 1:] for r in s["ref_states"]], axis=1)
    ours_c = ours - ours.mean(axis=1, keepdims=True)
    ref_c = ref - ref.mean(axis=1, keepdims=True)
    centred = _cos(ours_c, ref_c).mean(axis=1)
    raw = _cos(ours, ref).mean(axis=1)
    print("centred cosine per layer:", np.round(centred, 4))
    assert centred.min() >= 0.99, centred
    assert raw.min() >= 0.99, raw
    bos = _cos(s["states"][:, :, 0], np.stack([r[:, 0] for r in s["ref_states"]]))
    assert bos.min() >= 0.99, bos


def test_next_token_predictions_match_hf():
    s = _setup()
    agree = total = 0
    for i, t in enumerate(s["ids"]):
        agree += int((s["logits"][i, : len(t)].argmax(-1) == s["ref_top1"][i]).sum())
        total += len(t)
    print(f"top-1 agreement {agree}/{total}")
    assert agree / total >= 0.95
    # the notebook's sanity prompt: " Paris" must be the top prediction after "The capital of France is"
    paris = 12366
    assert int(s["logits"][0, len(s["ids"][0]) - 1].argmax()) == paris


def test_right_padding_does_not_change_real_positions():
    s = _setup()
    m, params, args = s["m"], s["params"], s["args"]
    t = s["ids"][1]
    alone = np.asarray(
        m.hidden_states_jit(jnp.asarray([t]), params, m.padding_mask(jnp.array([len(t)]), len(t)),
                            args.n_heads, args.n_kv_heads), dtype=np.float32)[0]
    batched = s["states"][1, :, : len(t)]
    assert _cos(alone, batched).min() >= 0.999


def test_attention_is_causal():
    s = _setup()
    m, params, args = s["m"], s["params"], s["args"]
    t = s["ids"][3]
    k = 7
    prefix = np.asarray(
        m.hidden_states_jit(jnp.asarray([t[:k]]), params, m.padding_mask(jnp.array([k]), k),
                            args.n_heads, args.n_kv_heads), dtype=np.float32)[0]
    full = s["states"][3, :, :k]
    assert _cos(prefix, full).min() >= 0.999, "states before position k must not depend on later tokens"
