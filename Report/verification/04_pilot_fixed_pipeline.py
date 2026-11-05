"""Pilot: does a *correct* forward pass recover word-sense signal?

Evaluated on the exact 100 instances of Data/Processed/SemCoreChunksUniform2/chunk_0.csv,
where the original pipeline scored ~0.55 mean normalised rank (chance = 0.5).

Model: Llama-3.2-1B via HuggingFace transformers (float32, causal, with
<|begin_of_text|>), i.e. a reference implementation rather than the JAX code.

Note on the original method under a correct model: the definition prompt puts
the focus word at position 0, and with causal attention position 0 cannot see
the definition, so all candidate vectors would be identical. The original
method only "worked" because the JAX model had no causal mask. The pilot
therefore uses causal-friendly representations:

Sentence side (vector for the focus word in context, last sub-token):
    S_plain  left context only (sequence truncated after the word)
    S_echo   sentence window repeated twice; vector taken from the 2nd copy, so
             it has seen the whole sentence (Springer et al. 2024, "echo embeddings")
Gloss side:
    G_mean   mean-pool over the definition tokens of " {word}: {definition}"
    G_echo   last sub-token of the word in " {word}: {definition}. {word}"
Supervised:
    kNN      centroid of S_plain vectors from up to K other SemCor sentences with
             the same lemma + sense (excluding the test sentence itself)

Scores: cosine after centring each side; normalised rank of the gold (1 = top,
notebook convention) and accuracy@1. Layer 8 (of 16) is the pre-specified
headline layer; full curves are saved too. The S_echo x G_echo pairing was fixed
as the headline design (in 05_figures.py) before this script was first run; it
did not beat chance, and the report says so alongside the variants that did.

Results: results/04_pilot_fixed_pipeline.json
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from common import HF_DIR, L_MARK, R_MARK, REPO, add_project_to_path, load_tokenizer, save_json

add_project_to_path()
from Processing.script import fix_whitespace  # noqa: E402

BOS = 128000
PAD = 128004  # <|finetune_right_pad_id|>
K_EXAMPLES = 3
WINDOW_LEFT, WINDOW_RIGHT = 40, 16
HEADLINE_LAYER = 8  # index into the 16 decoder-layer outputs (0-based)
BATCH = 16
RNG = np.random.default_rng(0)

torch.set_num_threads(4)


def candidates_of(definitions: str) -> list[str]:
    return definitions.split(" | ") if " | " in definitions else definitions.split("|")


def target_token_span(words, word_loc, tok):
    """(tokens, first_idx, last_idx) for the focus word, located via character offsets."""
    plain = " " + fix_whitespace(" ".join(words))
    mw = list(words)
    mw[word_loc] = L_MARK + words[word_loc] + R_MARK
    marked = " " + fix_whitespace(" ".join(mw))
    if marked.replace(L_MARK, "").replace(R_MARK, "") != plain:
        return None
    a = marked.index(L_MARK)
    b = marked.index(R_MARK) - 1  # exclusive end in the plain string
    toks = tok.encode(plain, bos=False, eos=False)
    pieces = [tok.decode([t]) for t in toks]
    if "".join(pieces) != plain:
        return None
    pos, first, last = 0, None, None
    for i, p in enumerate(pieces):
        s, e = pos, pos + len(p)
        if e > a and s < b:
            first = i if first is None else first
            last = i
        pos = e
    if first is None:
        return None
    return toks, first, last


class Embedder:
    def __init__(self):
        from transformers import LlamaForCausalLM

        self.model = LlamaForCausalLM.from_pretrained(str(HF_DIR), torch_dtype=torch.float32).model
        self.model.eval()
        self.captured = {}
        for k, layer in enumerate(self.model.layers):
            layer.register_forward_hook(lambda m, i, o, k=k: self.captured.__setitem__(k, o[0].detach()))
        self.tokens_processed = 0

    def run(self, seqs: list[list[int]], spans: list[tuple[int, int]]) -> np.ndarray:
        """Mean of hidden states over span [a, b) of each sequence -> (N, 16, 2048)."""
        out = np.zeros((len(seqs), 16, 2048), dtype=np.float32)
        order = np.argsort([len(s) for s in seqs])
        for start in range(0, len(seqs), BATCH):
            idx = order[start : start + BATCH]
            L = max(len(seqs[i]) for i in idx)
            ids = torch.full((len(idx), L), PAD, dtype=torch.long)
            att = torch.zeros((len(idx), L), dtype=torch.long)
            for r, i in enumerate(idx):
                ids[r, : len(seqs[i])] = torch.tensor(seqs[i])
                att[r, : len(seqs[i])] = 1
            with torch.no_grad():
                self.model(input_ids=ids, attention_mask=att)
            self.tokens_processed += int(att.sum())
            hs = torch.stack([self.captured[k] for k in range(16)], dim=1)  # (B, 16, L, D)
            for r, i in enumerate(idx):
                a, b = spans[i]
                out[i] = hs[r, :, a:b, :].mean(dim=1).numpy()
        return out


def norm_rank(scores: np.ndarray, gold: int) -> np.ndarray:
    n = scores.shape[0]
    ranks = np.argsort(np.argsort(-scores, axis=0), axis=0)
    return 1 - ranks[gold] / (n - 1)


def centred_cos_scores(sent_vecs, gloss_vecs_per_inst):
    """Per-layer centred cosine. sent_vecs (N, 16, D); gloss list of (n_i, 16, D)."""
    mu_s = sent_vecs.mean(0, keepdims=True)
    all_g = np.concatenate(gloss_vecs_per_inst)
    mu_g = all_g.mean(0, keepdims=True)
    out = []
    for s, g in zip(sent_vecs - mu_s[0], gloss_vecs_per_inst):
        gc = g - mu_g
        num = np.einsum("nld,ld->nl", gc, s)
        den = np.linalg.norm(gc, axis=-1) * np.linalg.norm(s, axis=-1)[None] + 1e-12
        out.append(num / den)
    return out


def summarise(score_list, golds, n_cands) -> dict:
    nr = np.stack([norm_rank(s, g) for s, g in zip(score_list, golds)])
    acc = np.stack([(s.argmax(0) == g) for s, g in zip(score_list, golds)]).astype(float)
    idx = RNG.integers(0, len(nr), size=(2000, len(nr)))
    boot = nr[:, HEADLINE_LAYER][idx].mean(1)
    return {
        "mean_norm_rank_by_layer": nr.mean(0).round(4).tolist(),
        "acc@1_by_layer": acc.mean(0).round(4).tolist(),
        f"layer{HEADLINE_LAYER}_mean_norm_rank": round(float(nr[:, HEADLINE_LAYER].mean()), 4),
        f"layer{HEADLINE_LAYER}_95ci": [round(float(np.percentile(boot, q)), 4) for q in (2.5, 97.5)],
        f"layer{HEADLINE_LAYER}_acc@1": round(float(acc[:, HEADLINE_LAYER].mean()), 4),
        "chance_acc@1": round(float(np.mean([1 / n for n in n_cands])), 4),
        "best_layer": int(nr.mean(0).argmax()),
        "best_layer_mean_norm_rank": round(float(nr.mean(0).max()), 4),
    }


def main() -> None:
    t0 = time.time()
    tok = load_tokenizer()
    test = pd.read_csv(REPO / "Data/Processed/SemCoreChunksUniform2/chunk_0.csv")
    full = pd.read_csv(REPO / "Data/Processed/SemCoreProcessed.csv")
    full["lemma"] = full.wordnet.str.split(" ").str[0].str.split("%").str[0]

    insts = []
    for _, r in test.iterrows():
        words = r.sentence.split("|")
        span = target_token_span(words, int(r.word_loc), tok)
        cands = candidates_of(r.definitions)
        if span is None or r.definition not in cands:
            continue
        toks, first, last = span
        insts.append(
            {"word": r.word, "words": words, "word_loc": int(r.word_loc), "toks": toks, "first": first,
             "last": last, "cands": cands, "gold": cands.index(r.definition),
             "lemma": r.wordnet.split(" ")[0].split("%")[0], "sentence": r.sentence}
        )
    print(f"{len(insts)} usable test instances")

    emb = Embedder()

    # ---- sentence side
    plain_seqs, plain_spans, echo_seqs, echo_spans = [], [], [], []
    for it in insts:
        lo = max(0, it["last"] - WINDOW_LEFT)
        left = it["toks"][lo : it["last"] + 1]
        plain_seqs.append([BOS] + left)
        plain_spans.append((len(left), len(left) + 1))  # last sub-token (BOS offset +1, then -1)
        win = it["toks"][lo : it["last"] + 1 + WINDOW_RIGHT]
        pos_in_win = it["last"] - lo
        echo_seqs.append([BOS] + win + win)
        p = 1 + len(win) + pos_in_win
        echo_spans.append((p, p + 1))
    S_plain = emb.run(plain_seqs, plain_spans)
    S_echo = emb.run(echo_seqs, echo_spans)
    print(f"sentences embedded ({time.time() - t0:.0f}s, {emb.tokens_processed} tokens)")

    # ---- gloss side
    g_mean_seqs, g_mean_spans, g_echo_seqs, g_echo_spans, owner = [], [], [], [], []
    for k, it in enumerate(insts):
        head = tok.encode(f" {it['word']}:", bos=False, eos=False)
        for c in it["cands"]:
            body = tok.encode(f" {it['word']}: {c}", bos=False, eos=False)
            g_mean_seqs.append([BOS] + body)
            g_mean_spans.append((1 + len(head), 1 + len(body)))
            ech = tok.encode(f" {it['word']}: {c}. {it['word']}", bos=False, eos=False)
            g_echo_seqs.append([BOS] + ech)
            g_echo_spans.append((len(ech), len(ech) + 1))
            owner.append(k)
    G_mean_flat = emb.run(g_mean_seqs, g_mean_spans)
    G_echo_flat = emb.run(g_echo_seqs, g_echo_spans)
    owner = np.array(owner)
    G_mean = [G_mean_flat[owner == k] for k in range(len(insts))]
    G_echo = [G_echo_flat[owner == k] for k in range(len(insts))]
    print(f"glosses embedded ({time.time() - t0:.0f}s, {emb.tokens_processed} tokens)")

    golds = [it["gold"] for it in insts]
    n_c = [len(it["cands"]) for it in insts]
    res = {"n_instances": len(insts), "headline_layer": HEADLINE_LAYER}
    for s_name, S in (("S_plain", S_plain), ("S_echo", S_echo)):
        for g_name, G in (("G_mean", G_mean), ("G_echo", G_echo)):
            res[f"gloss_match_{s_name}_x_{g_name}"] = summarise(centred_cos_scores(S, G), golds, n_c)
            r = res[f"gloss_match_{s_name}_x_{g_name}"]
            print(f"{s_name} x {g_name}: L{HEADLINE_LAYER} {r[f'layer{HEADLINE_LAYER}_mean_norm_rank']} "
                  f"{r[f'layer{HEADLINE_LAYER}_95ci']} acc@1 {r[f'layer{HEADLINE_LAYER}_acc@1']} "
                  f"(chance {r['chance_acc@1']}); best L{r['best_layer']} {r['best_layer_mean_norm_rank']}")

    # ---- supervised kNN centroids from other SemCor sentences
    ex_seqs, ex_spans, ex_owner = [], [], []
    coverage = []
    for k, it in enumerate(insts):
        same_lemma = full[full.lemma == it["lemma"]]
        cov = []
        for ci, c in enumerate(it["cands"]):
            pool = same_lemma[(same_lemma.definition == c) & ~(
                (same_lemma.sentence == it["sentence"]) & (same_lemma.word_loc == it["word_loc"]))]
            pool = pool.sample(n=min(K_EXAMPLES, len(pool)), random_state=0) if len(pool) else pool
            cov.append(len(pool))
            for _, ex in pool.iterrows():
                sp = target_token_span(ex.sentence.split("|"), int(ex.word_loc), tok)
                if sp is None:
                    continue
                toks, first, last = sp
                lo = max(0, last - WINDOW_LEFT)
                left = toks[lo : last + 1]
                ex_seqs.append([BOS] + left)
                ex_spans.append((len(left), len(left) + 1))
                ex_owner.append((k, ci))
        coverage.append(cov)
    print(f"embedding {len(ex_seqs)} kNN example sentences")
    EX = emb.run(ex_seqs, ex_spans)
    mu = np.concatenate([S_plain, EX]).mean(0)
    knn_scores, knn_full_cov = [], []
    for k, it in enumerate(insts):
        sc = np.full((len(it["cands"]), 16), -np.inf, dtype=np.float32)
        s = S_plain[k] - mu
        for ci in range(len(it["cands"])):
            rows = [j for j, o in enumerate(ex_owner) if o == (k, ci)]
            if rows:
                cen = EX[rows].mean(0) - mu
                sc[ci] = (cen * s).sum(-1) / (np.linalg.norm(cen, axis=-1) * np.linalg.norm(s, axis=-1) + 1e-12)
        knn_scores.append(sc)
        knn_full_cov.append(all(c > 0 for c in coverage[k]))
    res["knn_centroid_S_plain_all"] = summarise(knn_scores, golds, n_c)
    full_idx = [k for k, f in enumerate(knn_full_cov) if f]
    res["knn_centroid_S_plain_full_coverage_subset"] = summarise(
        [knn_scores[k] for k in full_idx], [golds[k] for k in full_idx], [n_c[k] for k in full_idx]
    )
    res["knn_instances_with_examples_for_every_candidate"] = len(full_idx)
    res["knn_examples_per_candidate_mean"] = float(np.mean([c for cov in coverage for c in cov]))
    for key in ("knn_centroid_S_plain_all", "knn_centroid_S_plain_full_coverage_subset"):
        r = res[key]
        print(f"{key}: L{HEADLINE_LAYER} {r[f'layer{HEADLINE_LAYER}_mean_norm_rank']} {r[f'layer{HEADLINE_LAYER}_95ci']} "
              f"acc@1 {r[f'layer{HEADLINE_LAYER}_acc@1']} (chance {r['chance_acc@1']}); "
              f"best L{r['best_layer']} {r['best_layer_mean_norm_rank']}")

    res["tokens_processed"] = emb.tokens_processed
    res["runtime_seconds"] = time.time() - t0
    print("saved", save_json("04_pilot_fixed_pipeline.json", res))


if __name__ == "__main__":
    main()
