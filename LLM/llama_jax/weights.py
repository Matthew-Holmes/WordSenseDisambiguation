"""Load, convert and save Llama weights for `llama_jax.model`.

`llama_jax.model` applies Meta-style (interleaved) RoPE, so `wq`/`wk` must be in Meta's layout.
The HuggingFace safetensors permute those two matrices for HF's rotate-half RoPE
(`convert_llama_to_hf.py::permute`); this module undoes that.

Formats
-------
* **v2 (current)**: a pickle of ``{"format_version": 2, "layout": "meta", "args": {...},
  "params": {...}}``, written by `save_params` / the CLI below.
* **legacy (March 2025)**: the bare params pickle written by
  `Processing/001pickle_jax_weights.ipynb`. It holds HF-layout `wq`/`wk` and RoPE frequencies
  computed with the wrong scaling constants. `load_params` repairs both on the fly and warns.

CLI (run from the repo root)::

    python -m LLM.llama_jax.weights upgrade --legacy Data/ModelWeights/llama_jax_weights.pkl \
        --out Data/ModelWeights/llama_jax_weights_v2.pkl
    python -m LLM.llama_jax.weights from-hf --hf-dir ~/.llama/checkpoints/Llama3.2-1B-hf \
        --out Data/ModelWeights/llama_jax_weights_v2.pkl
    python -m LLM.llama_jax.weights from-meta --meta-dir ~/.llama/checkpoints/Llama3.2-1B \
        --out Data/ModelWeights/llama_jax_weights_v2.pkl
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, Tuple

import jax.numpy as jnp
import numpy as np

from .model import ModelArgs, freqs_cis_for

FORMAT_VERSION = 2
LLAMA_3_2_1B = ModelArgs()  # defaults are Llama-3.2-1B


# ---------------------------------------------------------------------------
# q/k layout
# ---------------------------------------------------------------------------
def hf_permute(w, n_heads: int):
    """Meta layout -> HF layout (the permutation in convert_llama_to_hf.py). numpy or jax arrays."""
    d1, d2 = w.shape
    return w.reshape(n_heads, d1 // n_heads // 2, 2, d2).swapaxes(1, 2).reshape(d1, d2)


def hf_unpermute(w, n_heads: int):
    """HF layout -> Meta layout (inverse of hf_permute). numpy or jax arrays."""
    d1, d2 = w.shape
    return w.reshape(n_heads, 2, d1 // n_heads // 2, d2).swapaxes(1, 2).reshape(d1, d2)


# ---------------------------------------------------------------------------
# building params
# ---------------------------------------------------------------------------
def _layer(get, i: int, args: ModelArgs, from_hf: bool, to_jax) -> Dict:
    """One block's pytree. `get(name)` returns a numpy float32 array in the source naming."""
    if from_hf:
        p = f"model.layers.{i}."
        wq = hf_unpermute(get(p + "self_attn.q_proj.weight"), args.n_heads)
        wk = hf_unpermute(get(p + "self_attn.k_proj.weight"), args.n_kv_heads)
        names = {
            "wv": p + "self_attn.v_proj.weight", "wo": p + "self_attn.o_proj.weight",
            "gate": p + "mlp.gate_proj.weight", "up": p + "mlp.up_proj.weight",
            "down": p + "mlp.down_proj.weight", "pre": p + "input_layernorm.weight",
            "post": p + "post_attention_layernorm.weight",
        }
    else:
        p = f"layers.{i}."
        wq, wk = get(p + "attention.wq.weight"), get(p + "attention.wk.weight")
        names = {
            "wv": p + "attention.wv.weight", "wo": p + "attention.wo.weight",
            "gate": p + "feed_forward.w1.weight", "up": p + "feed_forward.w3.weight",
            "down": p + "feed_forward.w2.weight", "pre": p + "attention_norm.weight",
            "post": p + "ffn_norm.weight",
        }
    return {
        "attention": {"wq": to_jax(wq), "wk": to_jax(wk), "wv": to_jax(get(names["wv"])), "wo": to_jax(get(names["wo"]))},
        "feed_forward": {"gate": to_jax(get(names["gate"])), "up": to_jax(get(names["up"])), "down": to_jax(get(names["down"]))},
        "norms": {"pre_attention_rms": to_jax(get(names["pre"])), "post_attention_rms": to_jax(get(names["post"]))},
    }


def _assemble(layers, embed, norm, args: ModelArgs) -> Dict:
    return {
        "tok_embeddings": embed,
        "freqs_cis": freqs_cis_for(args),
        "layers": layers,
        "norm_scale": norm,
        "output_weight": embed,  # tied embeddings: same array, no extra memory
    }


def params_from_hf(hf_dir: str | Path, dtype=jnp.bfloat16, max_seq_len: int = 2048) -> Tuple[Dict, ModelArgs]:
    """Build Meta-layout params from a HuggingFace checkpoint directory (config.json + model.safetensors)."""
    from safetensors import safe_open

    hf_dir = Path(hf_dir).expanduser()
    args = ModelArgs.from_hf_config(json.loads((hf_dir / "config.json").read_text()), max_seq_len)
    to_jax = lambda a: jnp.asarray(a, dtype=dtype)  # noqa: E731
    with safe_open(str(hf_dir / "model.safetensors"), framework="pt") as f:
        get = lambda name: f.get_tensor(name).float().numpy()  # noqa: E731
        layers = [_layer(get, i, args, True, to_jax) for i in range(args.n_layers)]
        embed, norm = to_jax(get("model.embed_tokens.weight")), to_jax(get("model.norm.weight"))
    return _assemble(layers, embed, norm, args), args


def params_from_meta(meta_dir: str | Path, dtype=jnp.bfloat16, max_seq_len: int = 2048) -> Tuple[Dict, ModelArgs]:
    """Build params from Meta's checkpoint directory (params.json + consolidated.00.pth). Needs torch."""
    import torch

    meta_dir = Path(meta_dir).expanduser()
    p = json.loads((meta_dir / "params.json").read_text())
    args = ModelArgs(
        dim=p["dim"], n_layers=p["n_layers"], n_heads=p["n_heads"], n_kv_heads=p.get("n_kv_heads", p["n_heads"]),
        vocab_size=p["vocab_size"], norm_eps=p.get("norm_eps", 1e-5), rope_theta=p.get("rope_theta", 500000.0),
        use_scaled_rope=p.get("use_scaled_rope", True), max_seq_len=max_seq_len,
    )
    state = torch.load(meta_dir / "consolidated.00.pth", map_location="cpu", mmap=True, weights_only=True)
    to_jax = lambda a: jnp.asarray(a, dtype=dtype)  # noqa: E731
    get = lambda name: state[name].float().numpy()  # noqa: E731
    layers = [_layer(get, i, args, False, to_jax) for i in range(args.n_layers)]
    return _assemble(layers, to_jax(get("tok_embeddings.weight")), to_jax(get("norm.weight")), args), args


def upgrade_legacy(params: Dict, args: ModelArgs = LLAMA_3_2_1B) -> Dict:
    """Repair a March-2025 pickle: un-permute wq/wk, recompute RoPE frequencies, re-tie the output weight."""
    layers = []
    for lp in params["layers"]:
        attn = dict(lp["attention"])
        attn["wq"] = hf_unpermute(attn["wq"], args.n_heads)
        attn["wk"] = hf_unpermute(attn["wk"], args.n_kv_heads)
        layers.append({**lp, "attention": attn})
    return _assemble(layers, params["tok_embeddings"], params["norm_scale"], args)


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------
def save_params(path: str | Path, params: Dict, args: ModelArgs) -> None:
    payload = {"format_version": FORMAT_VERSION, "layout": "meta", "args": dataclasses.asdict(args), "params": params}
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_params(path: str | Path, dtype=None, max_seq_len: int | None = None) -> Tuple[Dict, ModelArgs]:
    """Load weights for llama_jax.model, repairing legacy pickles. Returns (params, args).

    dtype: optionally cast every weight (e.g. jnp.float32, which is faster than bfloat16 on CPUs
    without native bfloat16 support). max_seq_len: recompute RoPE tables for this length.
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and payload.get("format_version") == FORMAT_VERSION:
        args = ModelArgs(**payload["args"])
        params = payload["params"]
    elif isinstance(payload, dict) and {"tok_embeddings", "layers", "freqs_cis"} <= set(payload):
        warnings.warn(
            f"{path} is a legacy (March 2025) weights pickle with HF-layout wq/wk and incorrect RoPE "
            "scaling; repairing it in memory. Run `python -m LLM.llama_jax.weights upgrade` to save a v2 file.",
            stacklevel=2,
        )
        args = LLAMA_3_2_1B
        params = upgrade_legacy(payload, args)
    else:
        raise ValueError(f"{path}: unrecognised weights file")
    if max_seq_len is not None and max_seq_len != args.max_seq_len:
        args = dataclasses.replace(args, max_seq_len=max_seq_len)
        params = {**params, "freqs_cis": freqs_cis_for(args)}
    if dtype is not None:
        params = cast_params(params, dtype)
    return params, args


def cast_params(params: Dict, dtype) -> Dict:
    """Cast every real-valued weight to `dtype` (RoPE tables stay complex64); keeps embeddings tied."""
    embed = jnp.asarray(params["tok_embeddings"], dtype=dtype)

    def cast(tree):
        if isinstance(tree, dict):
            return {k: cast(v) for k, v in tree.items()}
        if isinstance(tree, list):
            return [cast(v) for v in tree]
        return jnp.asarray(tree, dtype=dtype)

    return {
        "tok_embeddings": embed,
        "freqs_cis": params["freqs_cis"],
        "layers": cast(params["layers"]),
        "norm_scale": jnp.asarray(params["norm_scale"], dtype=dtype),
        "output_weight": embed,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("upgrade", help="repair a legacy pickle and save it in v2 format")
    up.add_argument("--legacy", required=True)
    up.add_argument("--out", required=True)
    hf = sub.add_parser("from-hf", help="build v2 weights from a HuggingFace checkpoint directory")
    hf.add_argument("--hf-dir", required=True)
    hf.add_argument("--out", required=True)
    me = sub.add_parser("from-meta", help="build v2 weights from Meta's consolidated.00.pth")
    me.add_argument("--meta-dir", required=True)
    me.add_argument("--out", required=True)
    for p in (up, hf, me):
        p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    a = parser.parse_args()
    dtype = jnp.bfloat16 if a.dtype == "bfloat16" else jnp.float32
    if a.cmd == "upgrade":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params, args = load_params(a.legacy, dtype=dtype)
    elif a.cmd == "from-hf":
        params, args = params_from_hf(a.hf_dir, dtype)
    else:
        params, args = params_from_meta(a.meta_dir, dtype)
    save_params(a.out, params, args)
    print(f"saved {a.out} ({a.dtype}, Meta layout, format v{FORMAT_VERSION})")


if __name__ == "__main__":
    main()
