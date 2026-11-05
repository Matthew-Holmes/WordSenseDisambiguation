"""Functional Llama 3 in JAX.

Validated against HuggingFace `transformers` (see tests/test_model_parity.py): with the same
checkpoint and input tokens, every layer's hidden states agree to a centred cosine of >= 0.999
and next-token predictions agree on >= 98% of positions in bfloat16.

Two things callers must get right (both handled by `llama_jax.weights.load_params`):

* **Weight layout.** `apply_rotary_emb` rotates *adjacent* dimension pairs (2i, 2i+1), as in
  Meta's reference code, so `wq`/`wk` must be in Meta's layout (`consolidated.00.pth`). The
  HuggingFace safetensors permute `wq`/`wk` for HF's rotate-half RoPE; feeding them in directly
  scrambles the positional encoding. `load_params` un-permutes them.
* **Inputs.** Start every sequence with `<|begin_of_text|>` (id 128000), right-pad, and pass a
  key-padding mask built from the true lengths (`padding_mask`), never from token values
  (id 0 is the token "!").

Attention is causal by default, as in training. `hidden_states` returns the embeddings plus the
residual stream after every block, in HuggingFace's convention (index 0 = embeddings,
index k = output of block k).

History: the March 2025 version had no causal mask, ran RMSNorm/softmax in bfloat16 with
eps=1e-6, and used high_freq_factor=1 in the RoPE scaling. See Report/REPORT.md, bugs B2-B6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import jax
import jax.numpy as jnp

BOS_ID = 128000
PAD_ID = 128004  # <|finetune_right_pad_id|>; any id works because padding is masked by length


@dataclass(frozen=True)
class ModelArgs:
    """Architecture hyper-parameters (defaults: Llama-3.2-1B)."""

    dim: int = 2048
    n_layers: int = 16
    n_heads: int = 32
    n_kv_heads: int = 8
    vocab_size: int = 128256
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    use_scaled_rope: bool = True
    rope_scale_factor: float = 32.0
    rope_low_freq_factor: float = 1.0
    rope_high_freq_factor: float = 4.0
    rope_original_context: int = 8192
    max_seq_len: int = 2048

    def __post_init__(self):
        assert self.n_heads % self.n_kv_heads == 0
        assert self.dim % self.n_heads == 0

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @classmethod
    def from_hf_config(cls, config: dict, max_seq_len: int = 2048) -> "ModelArgs":
        """Build from a HuggingFace `config.json` dict."""
        scaling = config.get("rope_scaling") or {}
        return cls(
            dim=config["hidden_size"],
            n_layers=config["num_hidden_layers"],
            n_heads=config["num_attention_heads"],
            n_kv_heads=config.get("num_key_value_heads", config["num_attention_heads"]),
            vocab_size=config["vocab_size"],
            norm_eps=config.get("rms_norm_eps", 1e-5),
            rope_theta=config.get("rope_theta", 500000.0),
            use_scaled_rope=scaling.get("rope_type") == "llama3",
            rope_scale_factor=scaling.get("factor", 32.0),
            rope_low_freq_factor=scaling.get("low_freq_factor", 1.0),
            rope_high_freq_factor=scaling.get("high_freq_factor", 4.0),
            rope_original_context=scaling.get("original_max_position_embeddings", 8192),
            max_seq_len=max_seq_len,
        )


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------
def RMSNorm(x: jnp.ndarray, w: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    """RMSNorm computed in float32 and cast back to x.dtype before the weight, as in Meta's code."""
    xf = x.astype(jnp.float32)
    normed = xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return normed.astype(x.dtype) * w


# ---------------------------------------------------------------------------
# rotary position embeddings
# ---------------------------------------------------------------------------
def apply_scaling(
    freqs: jnp.ndarray,
    scale_factor: float = 32.0,
    original: int = 8192,
    low_freq_factor: float = 1.0,
    high_freq_factor: float = 4.0,
) -> jnp.ndarray:
    """Llama 3 RoPE frequency scaling ("llama3" rope_type).

    Wavelengths shorter than original/high_freq_factor are kept, longer than
    original/low_freq_factor are divided by scale_factor, and the band in between is
    interpolated smoothly.
    """
    low_freq_wavelen = original / low_freq_factor
    high_freq_wavelen = original / high_freq_factor
    wavelen = 2 * jnp.pi / freqs
    smooth = (original / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed = (1 - smooth) * freqs / scale_factor + smooth * freqs
    return jnp.where(
        wavelen < high_freq_wavelen,
        freqs,
        jnp.where(wavelen > low_freq_wavelen, freqs / scale_factor, smoothed),
    )


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 500000.0,
    use_scaled: bool = True,
    scale_factor: float = 32.0,
    original: int = 8192,
    low_freq_factor: float = 1.0,
    high_freq_factor: float = 4.0,
) -> jnp.ndarray:
    """Complex rotations exp(i * t * freq) for positions t < end; shape (end, dim // 2), complex64."""
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(jnp.float32) / dim))
    if use_scaled:
        freqs = apply_scaling(freqs, scale_factor, original, low_freq_factor, high_freq_factor)
    t = jnp.arange(end, dtype=jnp.float32)
    return jnp.exp(1j * jnp.outer(t, freqs)).astype(jnp.complex64)


def freqs_cis_for(args: ModelArgs, end: Optional[int] = None) -> jnp.ndarray:
    return precompute_freqs_cis(
        args.head_dim,
        end or args.max_seq_len,
        args.rope_theta,
        args.use_scaled_rope,
        args.rope_scale_factor,
        args.rope_original_context,
        args.rope_low_freq_factor,
        args.rope_high_freq_factor,
    )


def reshape_for_broadcast(freqs_cis: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (freqs_cis.shape, x.shape)
    shape = [x.shape[i] if i in (1, x.ndim - 1) else 1 for i in range(x.ndim)]
    return jnp.reshape(freqs_cis, shape)


def apply_rotary_emb(xq: jnp.ndarray, xk: jnp.ndarray, freqs_cis: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Meta-style RoPE: rotate adjacent pairs (2i, 2i+1). Expects Meta-layout wq/wk (see module doc)."""

    def rotate(x):
        xf = x.astype(jnp.float32).reshape(x.shape[:-1] + (x.shape[-1] // 2, 2))
        xc = jax.lax.complex(xf[..., 0], xf[..., 1])
        xr = xc * reshape_for_broadcast(freqs_cis, xc)
        out = jnp.stack([jnp.real(xr), jnp.imag(xr)], axis=-1).reshape(x.shape)
        return out.astype(x.dtype)

    return rotate(xq), rotate(xk)


def repeat_kv(x: jnp.ndarray, n_rep: int) -> jnp.ndarray:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return jnp.broadcast_to(x[:, :, :, None, :], (bs, slen, n_kv_heads, n_rep, head_dim)).reshape(
        bs, slen, n_kv_heads * n_rep, head_dim
    )


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------
def padding_mask(lengths: jnp.ndarray, seqlen: int) -> jnp.ndarray:
    """Additive key-padding mask (bs, seqlen): 0 for real tokens, -inf for right-padding."""
    positions = jnp.arange(seqlen)[None, :]
    return jnp.where(positions < lengths[:, None], 0.0, -jnp.inf).astype(jnp.float32)


def causal_mask(seqlen: int) -> jnp.ndarray:
    """Additive causal mask (seqlen, seqlen): -inf above the diagonal."""
    return jnp.triu(jnp.full((seqlen, seqlen), -jnp.inf, dtype=jnp.float32), k=1)


def _attention_bias(mask: Optional[jnp.ndarray], seqlen: int, causal: bool) -> Optional[jnp.ndarray]:
    """Combine a key-padding mask (bs, T) or a full additive mask (bs|1, 1, T, T) with the causal mask."""
    bias = None
    if mask is not None:
        mask = mask.astype(jnp.float32)
        bias = mask[:, None, None, :] if mask.ndim == 2 else mask
    if causal:
        c = causal_mask(seqlen)[None, None]
        bias = c if bias is None else bias + c
    return bias


# ---------------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------------
def attention_block(
    x: jnp.ndarray,
    mask: Optional[jnp.ndarray],
    freqs_cis: jnp.ndarray,
    wq: jnp.ndarray,
    wk: jnp.ndarray,
    wv: jnp.ndarray,
    wo: jnp.ndarray,
    n_heads: int,
    n_kv_heads: int,
    causal: bool = True,
) -> jnp.ndarray:
    """Grouped-query self-attention.

    `mask` may be None, an additive key-padding mask of shape (bs, seqlen) as returned by
    `padding_mask`, or a full additive mask of shape (bs or 1, 1, seqlen, seqlen). A causal
    mask is added unless causal=False. Scores and softmax are computed in float32.
    """
    bs, seqlen, model_dim = x.shape
    head_dim = model_dim // n_heads
    n_rep = n_heads // n_kv_heads

    xq = jnp.dot(x, wq.T).reshape(bs, seqlen, n_heads, head_dim)
    xk = jnp.dot(x, wk.T).reshape(bs, seqlen, n_kv_heads, head_dim)
    xv = jnp.dot(x, wv.T).reshape(bs, seqlen, n_kv_heads, head_dim)
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis[:seqlen])

    keys = repeat_kv(xk, n_rep)
    values = repeat_kv(xv, n_rep)
    xq = jnp.transpose(xq, (0, 2, 1, 3))
    keys = jnp.transpose(keys, (0, 2, 1, 3))
    values = jnp.transpose(values, (0, 2, 1, 3))

    scores = jnp.matmul(xq, jnp.transpose(keys, (0, 1, 3, 2)), preferred_element_type=jnp.float32)
    scores = scores / math.sqrt(head_dim)
    bias = _attention_bias(mask, seqlen, causal)
    if bias is not None:
        scores = scores + bias
    probs = jax.nn.softmax(scores, axis=-1).astype(x.dtype)

    output = jnp.matmul(probs, values)
    output = jnp.transpose(output, (0, 2, 1, 3)).reshape(bs, seqlen, -1)
    return jnp.dot(output, wo.T).astype(x.dtype)


def feed_forward(x: jnp.ndarray, gate: jnp.ndarray, down: jnp.ndarray, up: jnp.ndarray) -> jnp.ndarray:
    """SwiGLU MLP: down(silu(gate(x)) * up(x))."""
    return jnp.dot(jax.nn.silu(jnp.dot(x, gate.T)) * jnp.dot(x, up.T), down.T)


def transformer_block(
    x: jnp.ndarray,
    params: Dict,
    mask: Optional[jnp.ndarray],
    freqs_cis: jnp.ndarray,
    n_heads: int,
    n_kv_heads: int,
    causal: bool = True,
    eps: float = 1e-5,
) -> jnp.ndarray:
    attn, ff, norms = params["attention"], params["feed_forward"], params["norms"]
    h = x + attention_block(
        RMSNorm(x, norms["pre_attention_rms"], eps), mask, freqs_cis,
        attn["wq"], attn["wk"], attn["wv"], attn["wo"], n_heads, n_kv_heads, causal,
    )
    return h + feed_forward(RMSNorm(h, norms["post_attention_rms"], eps), ff["gate"], ff["down"], ff["up"])


# ---------------------------------------------------------------------------
# whole model
# ---------------------------------------------------------------------------
def hidden_states(
    tokens: jnp.ndarray,
    params: Dict,
    mask: Optional[jnp.ndarray],
    n_heads: int,
    n_kv_heads: int,
    causal: bool = True,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Residual stream, HuggingFace convention: (bs, n_layers + 1, seqlen, dim), index 0 = embeddings."""
    seqlen = tokens.shape[1]
    freqs_cis = params["freqs_cis"][:seqlen]
    h = params["tok_embeddings"][tokens]
    states = [h]
    for layer_params in params["layers"]:
        h = transformer_block(h, layer_params, mask, freqs_cis, n_heads, n_kv_heads, causal, eps)
        states.append(h)
    return jnp.stack(states, axis=1)


def reporting_transformer(
    tokens: jnp.ndarray,
    params: Dict,
    mask: Optional[jnp.ndarray],
    n_heads: int,
    n_kv_heads: int,
    causal: bool = True,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Residual stream after each block: (bs, n_layers, seqlen, dim). Kept for the original API."""
    return hidden_states(tokens, params, mask, n_heads, n_kv_heads, causal, eps)[:, 1:]


def transformer(
    tokens: jnp.ndarray,
    params: Dict,
    mask: Optional[jnp.ndarray],
    n_heads: int,
    n_kv_heads: int,
    causal: bool = True,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Next-token logits (bs, seqlen, vocab) in float32."""
    seqlen = tokens.shape[1]
    freqs_cis = params["freqs_cis"][:seqlen]
    h = params["tok_embeddings"][tokens]
    for layer_params in params["layers"]:
        h = transformer_block(h, layer_params, mask, freqs_cis, n_heads, n_kv_heads, causal, eps)
    h = RMSNorm(h, params["norm_scale"], eps)
    return jnp.dot(h, params["output_weight"].T, preferred_element_type=jnp.float32)


# jit-ready versions (the integer and flag arguments must be static)
STATIC_ARGS = ("n_heads", "n_kv_heads", "causal", "eps")
hidden_states_jit = jax.jit(hidden_states, static_argnames=STATIC_ARGS)
reporting_transformer_jit = jax.jit(reporting_transformer, static_argnames=STATIC_ARGS)
transformer_jit = jax.jit(transformer, static_argnames=STATIC_ARGS)
