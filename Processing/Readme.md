# Processing

## v2 pipeline (November 2026)

The March 2025 pipeline (notebooks 001–003 and the original `script.py`) produced activations that did not
match Llama-3.2-1B. See `Report/REPORT.md`, bugs B1–B16. The v2 pipeline replaces it; every step is
covered by tests in `tests/`.

| File | Role |
|---|---|
| `semcor_data.py`, `build_dataset.py` | Builds the v2 dataset in `Data/Processed/v2/`. It includes clean detokenised text with character spans, POS-restricted WordNet candidates, and a document-level dev/test split. It also provides the natural and sense-balanced samplers. |
| `prompts.py` | Builds the inputs. Sentences use the `[BOS] text text` "echo" form. Definitions use `[BOS] lemma: gloss`. Targets are located by character offsets. |
| `extract.py` | Runs batched JAX forward passes (float32 on CPU) and stores the pooled vectors as float16 arrays. The store is resumable. |
| `evaluation.py` | Metrics with random tie-breaking, sentence-clustered bootstrap CIs, baselines and dev-only selection. |
| `script.py` | Command-line interface with three commands: `sample`, `estimate` and `extract`. |
| `run_script.bash` | Runs the whole workflow end to end, including `Analysis/layer_study.py`. |

The model code is `LLM/llama_jax/model.py`, which is now fixed: it has a causal mask, the correct RoPE scaling,
and float32 norms and softmax. Load weights with `LLM/llama_jax/weights.py::load_params`. It accepts the
existing `llama_jax_weights.pkl` and repairs it in memory: it un-permutes q/k and recomputes the RoPE tables. To
save a repaired copy and avoid the warning, run:

```bash
.venv/bin/python -m LLM.llama_jax.weights upgrade --legacy Data/ModelWeights/llama_jax_weights.pkl \
    --out Data/ModelWeights/llama_jax_weights_v2.pkl
```

To run the tests from the repository root in WSL:

```bash
.venv/bin/python -m tests.run_tests                     # fast tests (~4 min, mostly imports)
WSD_SLOW_TESTS=1 .venv/bin/python -m tests.run_tests    # adds JAX-vs-HuggingFace parity (~6 min, ~8 GB RAM)
```

Throughput on the 4-core development laptop is about 32 tokens/s in float32. Extracting the four study samples
(1,208 instances) takes about an hour. A whole-corpus pass would take about a day on this CPU, or minutes on a
GPU. The minikube plan below was the March 2025 answer to the old pipeline's cost of about 3 weeks and 0.6 TB.

---

## Original plan (March 2025)

In this directly we'll collect all the scripts and steps to setup a minikube cluster, and iteratively process the dataset.

## Step 1 - Setting up minikube

### 1. install Docker

If you do not already have Docker installed, then set that up. (if you are running in WSL, then install Docker using Docker desktop, and enable WSL 2 - don't install natively from Ubuntu command line)

#### WSL

If using WSL, then first check your version

```powershell
wsl -l -v
```

You want to be running version 2, if not then run:

```powershell
wsl --set-version Ubuntu 2
```

To get this to work I had to:
* update some kernels (from a link in the failed attempt message)
* enable the Virtual Machine Platform (from: turn windows features on or off)

Once you've configured Docker, run:

```bash
docker --version
```
To check that is has been installed correctly. You can test it further with:

```bash
docker run hello-world
```

### 2. install minikube

Follow the instructions on their website, depends on CPU architecture, for me it was:

```bash
curl -LO https://github.com/kubernetes/minikube/releases/latest/download/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube && rm minikube-linux-amd64
```

(on WSL)

## Step 2 - Using Llama Jax to process the dataset

Currently we have a Python module to play around with the Llama model on toy prompts. In this notebook we develop this into functionality that we can containerise and use to process the dataset build from SemCore.

To get this operational the following are required:
* quick weight loading into ram (no weird conversions from PyTorch format) - `001pickle_jax_weights.ipynb`
* reusable JIT compilation information
* prompt batching and normalisation
* options to return intermediate activations, not just the final results








