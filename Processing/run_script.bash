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
