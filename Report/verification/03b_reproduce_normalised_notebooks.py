"""Reproduce the (2,12) mixture numbers of the chunk001 / Uniform2 notebooks.

Those two notebooks un-comment the L2-normalisation lines inside
compute_layerwise_dot_product_matrix / compute_layerwise_centred_dot_product_matrix,
so they score with cosine similarity (normalise first, then centre + abs for
the centred variant), whereas analysisUniform_chunk000.ipynb and
analysis.ipynb use raw dot products. This script tries both settings for each
expert to find which one the saved outputs correspond to.

Results: results/03b_reproduce_normalised_notebooks.json
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from common import ANALYSED_CHUNKS, save_json

spec = importlib.util.spec_from_file_location("reanalysis", Path(__file__).with_name("03_reanalysis.py"))
ra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ra)

NOTEBOOK_VALUES = {
    "uniform_chunk1": {"median": 0.6, "mean": 0.5661049684129406},
    "uniform2_chunk0": {"median": 0.5857142857142856, "mean": 0.5479928989139515},
}


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def lists_for(df, store, norm_raw: bool, norm_centred: bool):
    sep = " | " if df.definitions.str.contains(" | ", regex=False).any() else "|"
    raw_l, cen_l, gold = [], [], []
    for _, grp in df.groupby("sentence"):
        for _, sub in grp.groupby("word"):
            X = np.stack([store[k] for k in sub.model_output_key])  # (rows, 16, 2048)
            Xr = unit(X) if norm_raw else X
            raw_l.append(np.einsum("nle,le->nl", Xr, Xr[0])[1:])
            Xc = unit(X) if norm_centred else X
            Xc = Xc - Xc.mean(0, keepdims=True)
            cen_l.append(np.abs(np.einsum("nle,le->nl", Xc, Xc[0]))[1:])
            gold.append(sub.iloc[0].definitions.split(sep).index(sub.iloc[0].definition))
    return raw_l, cen_l, gold


def main() -> None:
    out = {}
    for name, target in NOTEBOOK_VALUES.items():
        paths = ANALYSED_CHUNKS[name]
        df = pd.read_csv(paths["processed"])
        store = ra.load_store(paths["store"])
        out[name] = {"notebook": target}
        for nr in (False, True):
            for nc in (False, True):
                raw_l, cen_l, gold = lists_for(df, store, nr, nc)
                med = ra.nb_mixture(raw_l, cen_l, 2, 12, gold, "median")[0]
                mean = ra.nb_mixture(raw_l, cen_l, 2, 12, gold, "mean")[0]
                key = f"normalise_uncentred={nr}, normalise_centred={nc}"
                match = bool(np.isclose(med, target["median"]) and np.isclose(mean, target["mean"]))
                out[name][key] = {"median": med, "mean": mean, "matches_notebook": match}
                print(f"{name} {key}: median {med:.4f} mean {mean:.4f} {'<== MATCH' if match else ''}")
    save_json("03b_reproduce_normalised_notebooks.json", out)


if __name__ == "__main__":
    main()
