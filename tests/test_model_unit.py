"""Fast unit tests for the JAX Llama implementation (no model weights needed)."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from LLM.llama_jax import model as m
from LLM.llama_jax import weights as w

HF_DIR = Path(os.environ.get("LLAMA_HF_DIR", "/home/matt/.llama/checkpoints/Llama3.2-1B-hf"))


def test_rope_frequencies_match_huggingface():
    """B5: the scaled RoPE frequencies must equal transformers' "llama3" initialisation."""
    if not (HF_DIR / "config.json").exists():
        raise unittest.SkipTest(f"no HF config at {HF_DIR}")
    from transformers import LlamaConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    config = LlamaConfig.from_pretrained(str(HF_DIR))
    hf_inv_freq, _ = ROPE_INIT_FUNCTIONS["llama3"](config, "cpu")
    args = m.ModelArgs.from_hf_config(config.to_dict())
    fc = np.asarray(m.freqs_cis_for(args, end=2))
    ours = np.angle(fc[1])  # position 1 -> angle == frequency
    np.testing.assert_allclose(ours, hf_inv_freq.numpy(), rtol=1e-5, atol=1e-9)


def test_apply_scaling_bands():
    freqs = 1.0 / (500000.0 ** (jnp.arange(0, 64, 2).astype(jnp.float32) / 64))
    scaled = np.asarray(m.apply_scaling(freqs))
    wavelen = 2 * np.pi / np.asarray(freqs)
    high = wavelen < 8192 / 4
    low = wavelen > 8192
    mid = ~high & ~low
    np.testing.assert_allclose(scaled[high], np.asarray(freqs)[high])
    np.testing.assert_allclose(scaled[low], np.asarray(freqs)[low] / 32, rtol=1e-6)
    assert mid.any(), "Llama-3.2 has frequencies in the smoothing band"
    assert np.all(scaled[mid] < np.asarray(freqs)[mid]) and np.all(scaled[mid] > np.asarray(freqs)[mid] / 32)
    assert np.all(np.isfinite(scaled))


def test_hf_permutation_round_trip():
    rng = np.random.default_rng(0)
    for n_heads, rows in ((32, 2048), (8, 512)):
        x = rng.standard_normal((rows, 16)).astype(np.float32)
        np.testing.assert_array_equal(w.hf_unpermute(w.hf_permute(x, n_heads), n_heads), x)
        assert not np.array_equal(w.hf_permute(x, n_heads), x)


def test_hf_permute_maps_meta_pairs_to_half_split():
    """Meta row 2i (real part of pair i) goes to HF row i; row 2i+1 (imag) goes to HF row i + head_dim/2."""
    head_dim = 8
    x = np.arange(head_dim, dtype=np.float32)[:, None]  # one head, row r holds the value r
    hf = w.hf_permute(x, n_heads=1)[:, 0]
    np.testing.assert_array_equal(hf, [0, 2, 4, 6, 1, 3, 5, 7])


def test_masks():
    pm = np.asarray(m.padding_mask(jnp.array([2, 4]), 4))
    assert np.isneginf(pm[0, 2:]).all() and (pm[0, :2] == 0).all() and (pm[1] == 0).all()
    cm = np.asarray(m.causal_mask(3))
    assert np.isneginf(cm[np.triu_indices(3, 1)]).all() and (cm[np.tril_indices(3)] == 0).all()
    bias = np.asarray(m._attention_bias(m.padding_mask(jnp.array([2]), 3), 3, causal=True))
    assert bias.shape == (1, 1, 3, 3)
    # row 2 (a padded query) can still see real keys 0 and 1, so no row is all -inf
    assert np.isfinite(bias[0, 0]).any(axis=-1).all()


def test_rmsnorm_float32_internals():
    x = jnp.array([[1e-3, 2e-3, 3e-3, 4e-3]], dtype=jnp.bfloat16)
    out = np.asarray(m.RMSNorm(x, jnp.ones(4, dtype=jnp.bfloat16)), dtype=np.float32)
    xf = np.asarray(x, dtype=np.float32)
    ref = xf / np.sqrt((xf**2).mean(-1, keepdims=True) + 1e-5)
    np.testing.assert_allclose(out, ref, rtol=1e-2)
