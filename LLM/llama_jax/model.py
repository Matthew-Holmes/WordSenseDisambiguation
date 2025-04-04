from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import math

import jax.numpy as jnp
import jax

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    use_scaled_rope: bool = True
    rope_scale_factor: float = 32.0

    max_batch_size: int = 32
    original_rotary_embed_len: int = 2048
    cache_len: int = 2048


    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.n_kv_heads <= self.n_heads
        assert self.n_heads % self.n_kv_heads == 0
        assert self.dim % self.n_heads == 0

def RMSNorm(x: jnp.array, w: jnp.array, eps: float = 1e-6) -> jnp.array:

    assert w.dtype == jnp.bfloat16, f"Expected weight to be bfloat16, but got {w.dtype}"
    assert x.dtype == jnp.bfloat16, f"Expected x to be bfloat16, but got {x.dtype}"

    rms = jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + eps)
    return (x * rms * w).astype(jnp.bfloat16)



def apply_scaling(freqs: jnp.ndarray, scale_factor: float, original: int) -> jnp.ndarray:
    low_freq_factor = 1.0
    high_freq_factor = 1.0
    old_context_len = float(original)

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * jnp.pi / freqs
    new_freqs = jnp.where(wavelen > low_freq_wavelen, freqs / scale_factor, freqs)
    smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    return jnp.where(
        (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen),
        (1 - smooth) * new_freqs / scale_factor + smooth * new_freqs,
        new_freqs
    )

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0,
                             use_scaled: bool = False, scale_factor: float = 32.0, original: int = 8192):
    dtype = jnp.float32
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
    t = jnp.arange(end, dtype=dtype)
    
    if use_scaled:
        freqs = apply_scaling(freqs, scale_factor, original)

    freqs = jnp.outer(t, freqs)
    freq_cis = jnp.exp(1j * freqs)
    return freq_cis


def reshape_for_broadcast(freqs_cis: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    ndim = x.ndim
    assert 0 <= 1 < ndim, f"Expected at least 2D x, got shape {x.shape}"
    shape = [
        (x.shape[i] if (i == 1 or i == ndim - 1) else 1)
        for i in range(ndim)
    ]
    return jnp.reshape(freqs_cis, shape)

def apply_rotary_emb(
    xq: jnp.ndarray,
    xk: jnp.ndarray,
    freqs_cis: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    dtype = jnp.float32

    xq_ = xq.astype(dtype)
    hidden_q = xq_.shape[-1]
    xq_reshaped = jnp.reshape(xq_, xq_.shape[:-1] + (hidden_q // 2, 2))

    xk_ = xk.astype(dtype)
    hidden_k = xk_.shape[-1]
    xk_reshaped = jnp.reshape(xk_, xk_.shape[:-1] + (hidden_k // 2, 2))

    # complex from (real, imag)
    xq_real = xq_reshaped[..., 0]
    xq_imag = xq_reshaped[..., 1]
    xk_real = xk_reshaped[..., 0]
    xk_imag = xk_reshaped[..., 1]

    xq_complex = xq_real + 1j * xq_imag
    xk_complex = xk_real + 1j * xk_imag

    freqs_cis_brd = reshape_for_broadcast(freqs_cis, xq_complex)

    xq_mul = xq_complex * freqs_cis_brd
    xk_mul = xk_complex * freqs_cis_brd

    def view_as_real(z: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack([jnp.real(z), jnp.imag(z)], axis=-1)

    xq_realimag = view_as_real(xq_mul)  # shape (..., 2)
    xk_realimag = view_as_real(xk_mul)  # shape (..., 2)

    # Flatten(3) => flatten the last two dimensions into one, 
    # e.g. if shape is [b, t, h, 2], flatten from dim=3 => [b, t, h * 2]
    # We'll do that by '... + (-1,)', effectively combining the last 2 dims.
    xq_out = jnp.reshape(xq_realimag, xq_realimag.shape[:3] + (-1,))
    xk_out = jnp.reshape(xk_realimag, xk_realimag.shape[:3] + (-1,))

    # back to original dtype
    xq_out = xq_out.astype(xq.dtype)
    xk_out = xk_out.astype(xk.dtype)

    return xq_out, xk_out


def repeat_kv(x: jnp.array, n_rep: int) -> jnp.array:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return jnp.reshape(
        jnp.tile(x[:, :, :, None, :], (1, 1, 1, n_rep, 1)), 
        (bs, slen, n_kv_heads * n_rep, head_dim)
    )


def attention_block(x: jnp.array, mask: Optional[jnp.array], freqs_cis: jnp.array,
                    wq: jnp.array, wk: jnp.array, wv: jnp.array,
                    wo: jnp.array,
                    n_heads: int, n_kv_heads: int) -> jnp.array:
    
    assert x.dtype == jnp.bfloat16, f"Expected x to be bfloat16, but got {x.dtype}"
    
    bs, seqlen, model_dim = x.shape

    head_dim         = model_dim // n_heads
    n_local_heads    = n_heads
    n_local_kv_heads = n_kv_heads
    n_rep            = n_local_heads // n_local_kv_heads

    xq = jnp.dot(x, wq.T).reshape(bs, seqlen, n_local_heads,    head_dim)
    xk = jnp.dot(x, wk.T).reshape(bs, seqlen, n_local_kv_heads, head_dim)
    xv = jnp.dot(x, wv.T).reshape(bs, seqlen, n_local_kv_heads, head_dim)

    xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

    keys   = repeat_kv(xk, n_rep)
    values = repeat_kv(xv, n_rep)

    xq     = jnp.transpose(xq,     (0, 2, 1, 3))  # (bs, n_local_heads, seqlen, head_dim)
    keys   = jnp.transpose(keys,   (0, 2, 1, 3))  # (bs, n_local_heads, seqlen, head_dim)
    values = jnp.transpose(values, (0, 2, 1, 3))  # (bs, n_local_heads, seqlen, head_dim)

    # here is where we compute the cross-attention
    scores = jnp.matmul(xq, jnp.transpose(keys, (0, 1, 3, 2))) / math.sqrt(head_dim) #(bs, n_local_heads, seqlen, seqlen)

    if mask is not None:
        # scores_unmasked = scores 
        scores += mask[:, None, None, :]   # mask should contain -inf for positions to ignore

        # print("Attention Scores Before Masking (Batch 0, Head 0):", scores_unmasked[0, 0])
        # print("Mask (Batch 0):", mask[0])  # Verify actual mask values
        # print("Attention Scores After Masking (Batch 0, Head 0):", scores[0, 0])

    scores = jax.nn.softmax(scores, axis=-1)
    # print("Softmaxed Attention Scores (Batch 0, Head 0):", scores[0, 0])

    # sum_probs = scores.sum(axis=-1)
    # print("Sum of softmax probabilities (should be 1 for unmasked tokens):", sum_probs[0])

    output = jnp.matmul(scores, values)
    output = jnp.transpose(output, (0, 2, 1, 3)).reshape(bs, seqlen, -1)
    output = jnp.dot(output, wo.T)

    return output.astype(jnp.bfloat16)


def feed_forward(x: jnp.array,
                 gate: jnp.array, down: jnp.array, up: jnp.array) -> jnp.array:
        
    assert x.dtype == jnp.bfloat16, f"Expected x to be bfloat16, but got {x.dtype}"

    return jnp.dot(jax.nn.silu(jnp.dot(x, gate.T)) * jnp.dot(x, up.T), down.T)
                
    
def transformer_block(x: jnp.ndarray,
                      params: Dict, 
                      mask: Optional[jnp.ndarray], 
                      freqs_cis: jnp.ndarray,
                      n_heads: int, n_kv_heads: int):
    
    assert x.dtype == jnp.bfloat16, f"Expected x to be bfloat16, but got {x.dtype}"

    # expects a params pytree
    attn_params = params["attention"]
    ff_params   = params["feed_forward"]
    norms       = params["norms"]

    x_norm = RMSNorm(x, norms["pre_attention_rms"])

    attn_out = attention_block(
        x_norm, mask, freqs_cis,
        attn_params["wq"], attn_params["wk"], attn_params["wv"], attn_params["wo"],
        n_heads, n_kv_heads
    )
    x = x + attn_out # residual
    x_norm = RMSNorm(x, norms["post_attention_rms"])

    ff_out = feed_forward(x_norm,
        ff_params["gate"], ff_params["down"], ff_params["up"])
    x = x + ff_out   # another residual

    return x


def transformer(tokens: jnp.ndarray,
                params: Dict,
                mask: Optional[jnp.ndarray], 
                n_heads: int, n_kv_heads: int):

    # bsz, seqlen = tokens.shape
    # use tokens as indices in the embedding matrix
    h = params["tok_embeddings"][tokens]  # (bsz, seqlen, dim)

    assert h.dtype == jnp.bfloat16, f"Expected h to be bfloat16, but got {h.dtype}"

    freqs_cis = params["freqs_cis"]

    for layer_params in params["layers"]:
        h = transformer_block(h, layer_params, mask, freqs_cis, n_heads, n_kv_heads)

    h = RMSNorm(h, params["norm_scale"])

    # project to vocabulary logits
    output = jnp.dot(h, params["output_weight"].T)

    return output


def reporting_transformer(tokens: jnp.ndarray,
                          params: Dict,
                          mask: Optional[jnp.ndarray], 
                          n_heads: int, n_kv_heads: int):
                          

    # bsz, seqlen = tokens.shape
    # use tokens as indices in the embedding matrix
    h = params["tok_embeddings"][tokens]  # (bsz, seqlen, dim)

    assert h.dtype == jnp.bfloat16, f"Expected h to be bfloat16, but got {h.dtype}"

    freqs_cis = params["freqs_cis"]

    acts = []
    for layer_params in params["layers"]:
        h = transformer_block(h, layer_params, mask, freqs_cis, n_heads, n_kv_heads)
        acts.append(h)

    return jnp.stack(acts, axis=1)  # (batch, layers, seq_len, hidden_dim)

