from dataclasses import dataclass
from typing import Optional, Tuple

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
    # TODO - what dtype should x be

    rms = jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + eps)
    return x * rms * w
