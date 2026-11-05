"""Is the JAX model that produced the activations actually Llama-3.2-1B?

Part A - weight provenance.
    `Processing/001pickle_jax_weights.ipynb` builds `llama_jax_weights.pkl` from
    the HuggingFace safetensors. `convert_llama_to_hf.py` permutes wq/wk so that
    HF's `rotate_half` RoPE works; the JAX code applies Meta's *interleaved*
    RoPE. We check the pickle == HF weights == permute(Meta weights).

Part B - forward-pass fidelity.
    Hidden states from the project's `reporting_transformer` (called exactly as
    `Processing/script.py` calls it) are compared against HuggingFace
    transformers (float32) on real SemCor prompts. Bugs are fixed one at a time
    to attribute the error:

        J0  as used by the pipeline (HF-permuted q/k, no causal mask, buggy
            RoPE scaling, eps=1e-6, no BOS, pad id 0)
        J1  J0 + q/k un-permuted (correct RoPE pairing)
        J2  J1 + causal mask
        J3  J2 + correct llama3 RoPE scaling + eps=1e-5   (should match HF)
        J4  J3 + <|begin_of_text|>                         (compared to HF+BOS)
        Jc  J0 + causal mask only (q/k still permuted)

Part C - the "capital of France" check from LLM/llama_to_jax.ipynb, showing
    that the notebook compared logits at a padding position.

Results: results/02_model_fidelity.json
"""

from __future__ import annotations

import gc
import math
import pickle
import time

import numpy as np
import pandas as pd

from common import (
    ANALYSED_CHUNKS,
    HF_DIR,
    JAX_WEIGHTS,
    META_DIR,
    add_project_to_path,
    load_tokenizer,
    save_json,
    true_focus_token_index,
)

add_project_to_path()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import torch  # noqa: E402

from llama_jax import model as jm  # noqa: E402  (the project's JAX model)
from Processing import script as pipeline  # noqa: E402

N_HEADS, N_KV_HEADS, HEAD_DIM = 32, 8, 64
BOS = 128000
N_TEST = 24


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def hf_permute(w: np.ndarray, n_heads: int) -> np.ndarray:
    """The permutation applied by convert_llama_to_hf.py (Meta -> HF layout)."""
    d1, d2 = w.shape
    return w.reshape(n_heads, d1 // n_heads // 2, 2, d2).swapaxes(1, 2).reshape(d1, d2)


def hf_unpermute(w, n_heads: int):
    """Inverse of hf_permute (HF -> Meta layout). Works for numpy and jax arrays."""
    d1, d2 = w.shape
    return w.reshape(n_heads, 2, d1 // n_heads // 2, d2).swapaxes(1, 2).reshape(d1, d2)


def llama3_inv_freq(dim=HEAD_DIM, theta=500000.0, factor=32.0, low=1.0, high=4.0, old_ctx=8192):
    """Reference llama3 RoPE scaling (matches Meta's reference and HF's 'llama3' rope_type)."""
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: dim // 2].astype(np.float64) / dim))
    low_wl, high_wl = old_ctx / low, old_ctx / high
    out = []
    for f in freqs:
        wl = 2 * math.pi / f
        if wl < high_wl:
            out.append(f)
        elif wl > low_wl:
            out.append(f / factor)
        else:
            s = (old_ctx / wl - low) / (high - low)
            out.append((1 - s) * f / factor + s * f)
    return np.array(out, dtype=np.float32)


def freqs_cis_from_inv(inv_freq: np.ndarray, length: int):
    t = np.arange(length, dtype=np.float32)
    return jnp.asarray(np.exp(1j * np.outer(t, inv_freq)).astype(np.complex64))


def attention_4d(x, mask4d, freqs_cis, wq, wk, wv, wo, n_heads, n_kv_heads):
    """The project's attention_block, verbatim except that it takes a 4-D additive mask."""
    bs, seqlen, model_dim = x.shape
    head_dim = model_dim // n_heads
    n_rep = n_heads // n_kv_heads
    xq = jnp.dot(x, wq.T).reshape(bs, seqlen, n_heads, head_dim)
    xk = jnp.dot(x, wk.T).reshape(bs, seqlen, n_kv_heads, head_dim)
    xv = jnp.dot(x, wv.T).reshape(bs, seqlen, n_kv_heads, head_dim)
    xq, xk = jm.apply_rotary_emb(xq, xk, freqs_cis)
    keys = jm.repeat_kv(xk, n_rep)
    values = jm.repeat_kv(xv, n_rep)
    xq = jnp.transpose(xq, (0, 2, 1, 3))
    keys = jnp.transpose(keys, (0, 2, 1, 3))
    values = jnp.transpose(values, (0, 2, 1, 3))
    scores = jnp.matmul(xq, jnp.transpose(keys, (0, 1, 3, 2))) / math.sqrt(head_dim)
    scores = scores + mask4d
    scores = jax.nn.softmax(scores, axis=-1)
    out = jnp.matmul(scores, values)
    out = jnp.transpose(out, (0, 2, 1, 3)).reshape(bs, seqlen, -1)
    return jnp.dot(out, wo.T).astype(jnp.bfloat16)


def forward_variant(tokens, layers, emb, freqs_cis, mask4d, eps):
    """The project's reporting_transformer with a 4-D mask and configurable eps."""
    h = emb[tokens]
    acts = []
    for lp in layers:
        a, ff, nm = lp["attention"], lp["feed_forward"], lp["norms"]
        x = jm.RMSNorm(h, nm["pre_attention_rms"], eps)
        h = h + attention_4d(x, mask4d, freqs_cis, a["wq"], a["wk"], a["wv"], a["wo"], N_HEADS, N_KV_HEADS)
        x = jm.RMSNorm(h, nm["post_attention_rms"], eps)
        h = h + jm.feed_forward(x, ff["gate"], ff["down"], ff["up"])
        acts.append(h)
    return jnp.stack(acts, axis=1)


forward_variant_jit = jax.jit(forward_variant, static_argnames=["eps"])


def build_masks(token_lists, pad_to, causal: bool):
    """Key-padding mask from true lengths (not from token id) plus optional causal mask."""
    B = len(token_lists)
    key = np.zeros((B, pad_to), dtype=np.float32)
    for i, t in enumerate(token_lists):
        key[i, len(t):] = -np.inf
    mask = key[:, None, None, :]
    if causal:
        causal_m = np.triu(np.full((pad_to, pad_to), -np.inf, dtype=np.float32), k=1)[None, None]
        mask = mask + causal_m
    return jnp.asarray(mask)


def pad(token_lists, pad_to):
    arr = np.zeros((len(token_lists), pad_to), dtype=np.int32)
    for i, t in enumerate(token_lists):
        arr[i, : len(t)] = t
    return jnp.asarray(arr)


def cos(a, b, axis=-1):
    num = (a * b).sum(axis)
    den = np.linalg.norm(a, axis=axis) * np.linalg.norm(b, axis=axis) + 1e-12
    return num / den


def compare(variant_acts, ref_acts, token_lists, focus_idx, offset=0):
    """Per-layer agreement between a variant and the reference.

    variant_acts: (B, L, T_pad, D) ; ref_acts: list of (L, T_i, D)
    offset: number of leading positions (e.g. BOS) excluded from the metrics.
    """
    L = ref_acts[0].shape[0]
    v_tok, r_tok, v_foc, r_foc = [], [], [], []
    for i, toks in enumerate(token_lists):
        n = len(toks)
        v = np.asarray(variant_acts[i, :, offset:n, :], dtype=np.float32)
        r = ref_acts[i][:, offset:n, :]
        v_tok.append(v)
        r_tok.append(r)
        v_foc.append(np.asarray(variant_acts[i, :, focus_idx[i], :], dtype=np.float32))
        r_foc.append(ref_acts[i][:, focus_idx[i], :])
    V = np.concatenate(v_tok, axis=1)  # (L, N_tokens, D)
    R = np.concatenate(r_tok, axis=1)
    Vf = np.stack(v_foc, axis=1)  # (L, B, D)
    Rf = np.stack(r_foc, axis=1)
    out = {}
    out["cos_all_tokens"] = cos(V, R).mean(axis=1).tolist()
    Vc = V - V.mean(axis=1, keepdims=True)
    Rc = R - R.mean(axis=1, keepdims=True)
    out["centred_cos_all_tokens"] = cos(Vc, Rc).mean(axis=1).tolist()
    out["rel_l2_all_tokens"] = (np.linalg.norm(V - R, axis=-1) / np.linalg.norm(R, axis=-1)).mean(axis=1).tolist()
    out["cos_focus_token"] = cos(Vf, Rf).mean(axis=1).tolist()
    Vfc = Vf - V.mean(axis=1, keepdims=True)
    Rfc = Rf - R.mean(axis=1, keepdims=True)
    out["centred_cos_focus_token"] = cos(Vfc, Rfc).mean(axis=1).tolist()
    # Can a focus-token vector be matched to its own sentence's reference vector?
    hits = []
    for layer in range(L):
        sims = cos(Vfc[layer][:, None, :], Rfc[layer][None, :, :])  # (B, B)
        hits.append(float((sims.argmax(axis=1) == np.arange(sims.shape[0])).mean()))
    out["focus_retrieval_top1"] = hits
    return out


def logits_from_hidden(h_last, head):
    """Final norm + tied output projection, in float32. head = (norm_w, W_out) as float32."""
    norm_w, w_out = head
    h = np.asarray(h_last, dtype=np.float32)
    h = h / np.sqrt((h**2).mean(-1, keepdims=True) + 1e-5) * norm_w
    return h @ w_out.T


# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    results: dict = {}
    tok = load_tokenizer()

    # -------------------------------------------------------------- Part A
    from safetensors import safe_open

    print("Part A: weight provenance")
    meta = torch.load(META_DIR / "consolidated.00.pth", map_location="cpu", mmap=True, weights_only=True)
    partA = {}
    with safe_open(str(HF_DIR / "model.safetensors"), framework="pt") as f:
        for layer in (0, 7, 15):
            for hf_name, meta_name, nh in (
                ("q_proj", "wq", N_HEADS),
                ("k_proj", "wk", N_KV_HEADS),
                ("v_proj", "wv", None),
                ("o_proj", "wo", None),
            ):
                hf_w = f.get_tensor(f"model.layers.{layer}.self_attn.{hf_name}.weight").float().numpy()
                me_w = meta[f"layers.{layer}.attention.{meta_name}.weight"].float().numpy()
                rec = {"hf_equals_meta": bool(np.array_equal(hf_w, me_w))}
                if nh:
                    rec["hf_equals_permute(meta)"] = bool(np.array_equal(hf_w, hf_permute(me_w, nh)))
                    rec["unpermute(hf)_equals_meta"] = bool(np.array_equal(hf_unpermute(hf_w, nh), me_w))
                partA[f"layer{layer}.{meta_name}"] = rec
    del meta
    gc.collect()

    with open(JAX_WEIGHTS, "rb") as fh:
        params = pickle.load(fh)
    with safe_open(str(HF_DIR / "model.safetensors"), framework="pt") as f:
        hf_q0 = f.get_tensor("model.layers.0.self_attn.q_proj.weight").float().numpy()
    pk_q0 = np.asarray(params["layers"][0]["attention"]["wq"], dtype=np.float32)
    partA["pickle_layer0.wq_equals_hf_q_proj"] = bool(np.array_equal(pk_q0, hf_q0))
    results["partA_weight_provenance"] = partA
    for k, v in partA.items():
        print(f"  {k}: {v}")

    # RoPE frequencies: project vs reference vs HF
    pickled_fc = np.asarray(params["freqs_cis"])
    project_inv = np.angle(pickled_fc[1]).astype(np.float64)  # position 1 => angle == inv_freq
    ref_inv = llama3_inv_freq()

    # -------------------------------------------------------------- test prompts
    df = pd.read_csv(ANALYSED_CHUNKS["uniform2_chunk0"]["input"]).head(N_TEST)
    token_lists, focus_idx, focus_words = [], [], []
    for _, row in df.iterrows():
        r = pipeline.prep_row(row.copy(), tok)
        r = pipeline.clip_sentence_token_list(r, 32)
        toks = list(r["clipped_sentence_toks"])
        # true focus index (re-located inside the clipped window)
        words = row["sentence"].split("|")
        gt, _ = true_focus_token_index(words, int(row["word_loc"]), r["sentence_toks"], tok)
        shift = int(r["word_tok_index"]) - int(r["clipped_word_tok_index"])
        gt_clipped = gt - shift if gt is not None else int(r["clipped_word_tok_index"])
        if not 0 <= gt_clipped < len(toks):
            gt_clipped = int(r["clipped_word_tok_index"])
        token_lists.append(toks)
        focus_idx.append(gt_clipped)
        focus_words.append(row["word"])
    bos_lists = [[BOS] + t for t in token_lists]
    bos_focus = [i + 1 for i in focus_idx]

    # -------------------------------------------------------------- HF reference
    print("Part B: HF reference forward passes")
    from transformers import LlamaForCausalLM

    hf = LlamaForCausalLM.from_pretrained(str(HF_DIR), torch_dtype=torch.float32, low_cpu_mem_usage=True)
    hf.eval()
    hf_inv = hf.model.rotary_emb.inv_freq.detach().numpy().astype(np.float64)
    results["rope_inv_freq"] = {
        "max_abs_diff_reference_vs_hf": float(np.abs(ref_inv - hf_inv).max()),
        "project_vs_hf_by_index": {
            int(i): {"project": float(project_inv[i]), "hf": float(hf_inv[i])}
            for i in range(32)
            if not np.isclose(project_inv[i], hf_inv[i], rtol=1e-3)
        },
    }
    print("  RoPE indices where project != HF:", list(results["rope_inv_freq"]["project_vs_hf_by_index"]))

    captured: dict = {}
    hooks = [
        layer.register_forward_hook(lambda m, i, o, k=k: captured.__setitem__(k, o[0].detach()))
        for k, layer in enumerate(hf.model.layers)
    ]

    def hf_run(lists):
        acts, top1 = [], []
        with torch.no_grad():
            for t in lists:
                out = hf(torch.tensor([t]))
                acts.append(np.stack([captured[k][0].numpy() for k in range(16)]))
                top1.append(out.logits[0].argmax(-1).numpy())
        return acts, top1

    ref_nobos, ref_nobos_logits = hf_run(token_lists)
    ref_bos, ref_bos_logits = hf_run(bos_lists)

    france = tok.encode("The capital of France is", bos=True, eos=False)
    with torch.no_grad():
        fr_logits = hf(torch.tensor([france])).logits[0, -1].numpy()
    for h in hooks:
        h.remove()
    del hf
    gc.collect()

    # -------------------------------------------------------------- JAX variants
    print("Part B: JAX variants")
    emb = params["tok_embeddings"]
    layers_perm = params["layers"]
    layers_unperm = [
        {
            **lp,
            "attention": {
                **lp["attention"],
                "wq": hf_unpermute(lp["attention"]["wq"], N_HEADS),
                "wk": hf_unpermute(lp["attention"]["wk"], N_KV_HEADS),
            },
        }
        for lp in layers_perm
    ]
    fc_proj32 = jnp.asarray(pickled_fc[:32])
    fc_ref32 = freqs_cis_from_inv(ref_inv, 32)
    fc_ref33 = freqs_cis_from_inv(ref_inv, 33)

    # J0 exactly as Processing/script.py does it
    padded, masks = zip(*[pipeline.pad_toks(t, 32) for t in token_lists])
    p32 = dict(params)
    p32["freqs_cis"] = fc_proj32
    rt = jax.jit(jm.reporting_transformer, static_argnames=["n_heads", "n_kv_heads"])
    t = time.time()
    J0 = np.asarray(rt(jnp.concatenate(padded), p32, jnp.concatenate(masks), N_HEADS, N_KV_HEADS), dtype=np.float32)
    print(f"  J0 done ({time.time() - t:.0f}s)")

    toks32 = pad(token_lists, 32)
    variants = {
        "J0_as_used": (J0, ref_nobos, token_lists, focus_idx, 0),
    }
    # consistency: our re-implementation with J0's settings must reproduce J0
    J0_re = np.asarray(
        forward_variant_jit(toks32, layers_perm, emb, fc_proj32, build_masks(token_lists, 32, False), 1e-6),
        dtype=np.float32,
    )
    real = np.zeros_like(J0, dtype=bool)
    for i, tl in enumerate(token_lists):
        real[i, :, : len(tl), :] = True
    results["reimplementation_check_max_abs_diff_vs_J0_on_real_tokens"] = float(np.abs(J0_re - J0)[real].max())
    print("  re-implementation vs J0 max abs diff:", results["reimplementation_check_max_abs_diff_vs_J0_on_real_tokens"])

    specs = {
        "Jc_causal_only": (layers_perm, fc_proj32, False, True, 1e-6),
        "J1_unpermuted_qk": (layers_unperm, fc_proj32, False, False, 1e-6),
        "J2_unpermuted_plus_causal": (layers_unperm, fc_proj32, False, True, 1e-6),
        "J3_all_fixes_no_bos": (layers_unperm, fc_ref32, False, True, 1e-5),
        "J4_all_fixes_with_bos": (layers_unperm, fc_ref33, True, True, 1e-5),
    }
    for name, (lyr, fc, use_bos, causal, eps) in specs.items():
        t = time.time()
        lists = bos_lists if use_bos else token_lists
        L = 33 if use_bos else 32
        acts = np.asarray(
            forward_variant_jit(pad(lists, L), lyr, emb, fc, build_masks(lists, L, causal), eps), dtype=np.float32
        )
        if use_bos:
            variants[name] = (acts, ref_bos, bos_lists, bos_focus, 1)
        else:
            variants[name] = (acts, ref_nobos, token_lists, focus_idx, 0)
        print(f"  {name} done ({time.time() - t:.0f}s)")

    # J0 vs the BOS reference: how far are the stored activations from what the
    # model computes for the same sentence when run normally (BOS + causal)?
    J0_shift = np.zeros((J0.shape[0], J0.shape[1], 33, J0.shape[3]), dtype=np.float32)
    J0_shift[:, :, 1:, :] = J0[:, :, :32, :]
    variants["J0_as_used_vs_HF_with_BOS"] = (J0_shift, ref_bos, bos_lists, bos_focus, 1)

    head = (
        np.asarray(params["norm_scale"], dtype=np.float32),
        np.asarray(params["output_weight"], dtype=np.float32),
    )
    partB = {}
    for name, (acts, ref, lists, fidx, off) in variants.items():
        partB[name] = compare(acts, ref, lists, fidx, off)
        # next-token top-1 agreement with HF over real positions
        ref_top1 = ref_bos_logits if lists is bos_lists else ref_nobos_logits
        agree, total = 0, 0
        for i, tl in enumerate(lists):
            lg = logits_from_hidden(acts[i, -1, off : len(tl), :], head)
            agree += int((lg.argmax(-1) == ref_top1[i][off : len(tl)]).sum())
            total += len(tl) - off
        partB[name]["next_token_top1_agreement_with_hf"] = agree / total
        s = partB[name]
        print(
            f"  {name:28s} focus cos L1/L8/L16: "
            f"{s['cos_focus_token'][0]:.3f}/{s['cos_focus_token'][7]:.3f}/{s['cos_focus_token'][15]:.3f}  "
            f"centred: {s['centred_cos_focus_token'][0]:.3f}/{s['centred_cos_focus_token'][7]:.3f}/"
            f"{s['centred_cos_focus_token'][15]:.3f}  retrieval L8/L16: "
            f"{s['focus_retrieval_top1'][7]:.2f}/{s['focus_retrieval_top1'][15]:.2f}  "
            f"top1-agree: {s['next_token_top1_agreement_with_hf']:.2f}"
        )
    results["partB_forward_fidelity"] = partB
    results["partB_test_set"] = {
        "source": "uniform2_chunk0 first rows",
        "n_sentences": len(token_lists),
        "focus_words": focus_words,
    }

    # -------------------------------------------------------------- Part C
    print("Part C: capital-of-France check")

    def top5(logits):
        idx = np.argsort(-logits)[:5]
        lp = logits - (np.log(np.exp(logits - logits.max()).sum()) + logits.max())
        return [(tok.decode([int(i)]), round(float(lp[i]), 3)) for i in idx]

    fr_pad = france + [0] * (30 - len(france))
    fr_mask = [0 if x != 0 else -jnp.inf for x in fr_pad]
    p30 = dict(params)
    p30["freqs_cis"] = jnp.asarray(pickled_fc[:30])
    out_nb = np.asarray(
        jm.transformer(jnp.array(fr_pad)[None, :], p30, jnp.array(fr_mask)[None, :], N_HEADS, N_KV_HEADS),
        dtype=np.float32,
    )[0]
    fixed = np.asarray(
        forward_variant_jit(pad([france], 8), layers_unperm, emb, freqs_cis_from_inv(ref_inv, 8),
                            build_masks([france], 8, True), 1e-5),
        dtype=np.float32,
    )[0, -1, len(france) - 1]
    results["partC_capital_of_france"] = {
        "hf_reference_last_real_token": top5(fr_logits),
        "notebook_jax_at_position_-1_(a_pad_token)": top5(out_nb[-1]),
        "notebook_jax_at_last_real_token": top5(out_nb[len(france) - 1]),
        "fixed_jax_at_last_real_token": top5(logits_from_hidden(fixed[None], head)[0]),
    }
    for k, v in results["partC_capital_of_france"].items():
        print(f"  {k}: {v}")

    results["runtime_seconds"] = time.time() - t0
    print("saved", save_json("02_model_fidelity.json", results))


if __name__ == "__main__":
    main()
