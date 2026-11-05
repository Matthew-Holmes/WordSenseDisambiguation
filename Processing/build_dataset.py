"""Build the v2 SemCor dataset (see Processing/semcor_data.py).

Run from the repository root (WSL, project venv):

    .venv/bin/python -m Processing.build_dataset
    .venv/bin/python -m Processing.build_dataset --out Data/Processed/v2 --seed 0

Writes sentences.jsonl, instances.jsonl, synsets.json, splits.json and build_summary.json.
Takes about a minute on CPU; needs NLTK's WordNet data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from Processing.semcor_data import DEFAULT_ROOT, RAW_KEYS, RAW_XML, build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-xml", type=Path, default=RAW_XML, help="SemCor XML (Raganato et al. 2017 format).")
    parser.add_argument("--raw-keys", type=Path, default=RAW_KEYS, help="SemCor gold key file.")
    parser.add_argument("--out", type=Path, default=DEFAULT_ROOT, help="Output directory.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the document-level dev/test split.")
    args = parser.parse_args()

    summary = build(args.raw_xml, args.raw_keys, args.out, seed=args.seed)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
