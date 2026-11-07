#!/usr/bin/env bash
# v2 pipeline, run from the repository root inside WSL (see Processing/Readme.md).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

# 0. one-off: build the v2 dataset (POS-restricted candidates, document splits, clean text)
[ -f Data/Processed/v2/instances.jsonl ] || $PY -m Processing.build_dataset

# 1. samples (document-level dev/test split)
$PY -m Processing.script sample --split dev  --kind natural  --n 80
$PY -m Processing.script sample --split test --kind natural  --n 80
$PY -m Processing.script sample --split dev  --kind balanced --n 200
$PY -m Processing.script sample --split test --kind balanced --n 200

# 2. cost estimate, then extraction (resumable; cached in Data/Processed/v2/activations)
SAMPLES="dev_natural_80 test_natural_80 dev_balanced_200 test_balanced_200"
$PY -m Processing.script estimate --samples $SAMPLES
$PY -u -m Processing.script extract --samples $SAMPLES

# 3. layer study: choose on dev, validate on test, then draw the figures
$PY Analysis/layer_study.py
$PY Analysis/layer_study_figures.py

# 4. swap study: re-run each sentence with the target replaced by a random word of the same POS
#    (pos:0 = the design as specified; pos+form:0 = inflection-matched; pos:1 = a second random draw)
$PY -m Processing.script estimate --samples $SAMPLES --swaps pos:0 pos+form:0 pos:1
$PY -u -m Processing.script extract --samples $SAMPLES --swaps pos:0 pos+form:0 pos:1
$PY Analysis/swap_study.py
$PY Analysis/swap_study_figures.py

# 5. gloss-mean study: remove the mean of each target's candidate gloss vectors (no new extraction)
$PY Analysis/gloss_mean_study.py
$PY Analysis/gloss_mean_study_figures.py

# 6. swap-draws study: up to 5 POS-matched swaps per target, averaged (sense-balanced samples only)
$PY -u -m Processing.script extract --samples dev_balanced_200 test_balanced_200 --swaps pos:2-4
$PY Analysis/swap_draws_study.py
$PY Analysis/swap_draws_figures.py

# 7. metrics study: other similarity metrics and cheap post-processing, judged on sense-balanced data
$PY -u Analysis/metrics_study.py
$PY Analysis/metrics_study_figures.py
