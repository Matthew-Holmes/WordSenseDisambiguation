# Verification scripts

These scripts back every number quoted in [`../REPORT.md`](../REPORT.md). They use the
project's own virtualenv, data files and checkpoints, and they import the pipeline code
(`Processing/script.py`, `LLM/llama_jax/model.py`) directly. That way they test the code
that actually produced the results rather than a re-implementation.

Run from WSL, from this directory:

```bash
cd /mnt/c/Users/Matt/Projects/WordSenseDisambiguation/Report/verification
PY=../../.venv/bin/python
$PY 01_data_and_alignment.py              # ~12 min, CPU only, tokenizer + WordNet
$PY 02_model_fidelity.py                  # ~16 min, peaks at ~9 GB RAM (JAX pickle + HF model)
$PY 02b_bos_effect.py                     # ~4 min, HF model only
$PY 03_reanalysis.py                      # ~5 min, reads the stored activations from the zips
$PY 03b_reproduce_normalised_notebooks.py # ~2 min
$PY 04_pilot_fixed_pipeline.py            # ~20 min on 4 CPU cores, HF model in float32
$PY 05_figures.py                         # renders ../figures/*.png from the JSON results
```

Times are from the original 4-core, 16 GB machine. Run 02 and 04 one at a time, because each loads the full
model.

Checkpoint locations default to `/home/matt/.llama/checkpoints/...`. You can override them
with `LLAMA_CKPT_ROOT`, `LLAMA_TOKENIZER`, `LLAMA_HF_DIR` or `LLAMA_META_DIR` (see
`common.py`).

| Script | Question it answers | Output |
|---|---|---|
| `01_data_and_alignment.py` | Does `prep_row` pick the right token for the focus word? How often do the padding/`!` bug, whitespace damage, clipping and the colon crash occur? What do the candidate lists look like, and how strong is the list-order (MFS) baseline? | `results/01_data_and_alignment.json` |
| `02_model_fidelity.py` | Are the pickled weights HF-permuted? How close are the JAX activations to the real Llama-3.2-1B, with each bug fixed in turn? Was the notebook's "capital of France" comparison taken at a padding position? | `results/02_model_fidelity.json` |
| `02b_bos_effect.py` | In the real model, what does the missing BOS do? Does the target word at position 0 of a definition prompt become an attention sink? Did that happen in the stored activations? | `results/02b_bos_effect.json` |
| `03_reanalysis.py` | Can the notebook numbers be reproduced? How do they compare with baselines, confidence intervals and a permutation null for the layer-grid search? Is there a difference between aligned and misaligned rows? | `results/03_reanalysis.json` |
| `03b_reproduce_normalised_notebooks.py` | Which scoring setting (raw dot product or cosine) produced the chunk001 and Uniform2 notebook numbers? | `results/03b_reproduce_normalised_notebooks.json` |
| `04_pilot_fixed_pipeline.py` | With a correct forward pass, is there word-sense signal on the same 100 instances? | `results/04_pilot_fixed_pipeline.json` |
| `05_figures.py` | Renders the report's figures from the JSON results. | `../figures/*.png` |
| `common.py` | Shared paths, project imports and the character-offset focus-token locator (`true_focus_token_index`). | — |

The `results/*_console.txt` files hold the console output of the runs quoted in the report. They are not named
`*.log` because the repository's `.gitignore` ignores that extension. To capture a new run, use
`$PY -u <script> | tee results/<script>_console.txt`.
