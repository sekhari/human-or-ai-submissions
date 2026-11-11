"""Zero-shot AI-text scores from two pretrained causal LMs (Binoculars, Hans et al. 2024).

For each passage, an observer and a performer model (Falcon-7B and Falcon-7B-instruct, Apache 2.0)
read the same tokens. The Binoculars score divides the performer's log-perplexity by the
cross-perplexity between the two models; AI text scores lower. No labels are used to compute it,
so it does not depend on which genres the labeled data covers, and all 10,000 train.csv passages
are a fair evaluation set.

    python -m experiments.binoculars --sets train test        # score (resumable per set)
    python -m experiments.binoculars --report-only            # AUROC overall, on validation, per genre

Outputs:
    results/binoculars/<set>.csv   id, n_tokens, log_ppl_obs, log_ppl_perf, log_xppl, score (+ label for train)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.data import DATA_DIR, RESULTS_DIR, load_labeled, load_split, load_unlabeled
from src.genres import genre
from src.utils import get_device

BINO_DIR = RESULTS_DIR / "binoculars"
FEATURES = ("log_ppl_obs", "log_ppl_perf", "log_xppl", "score")


def binoculars_features(obs_logits: torch.Tensor, perf_logits: torch.Tensor, input_ids: torch.Tensor,
                        attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-passage log-perplexities and Binoculars score from two models' logits over the same tokens.

    Position t predicts token t + 1, so the last position and padding are excluded.
    log_xppl is the performer's cross-entropy against the observer's next-token distribution.
    """
    obs = obs_logits[:, :-1].float().log_softmax(-1)
    perf = perf_logits[:, :-1].float().log_softmax(-1)
    target = input_ids[:, 1:].unsqueeze(-1)
    mask = attention_mask[:, 1:].float()
    n = mask.sum(1).clamp(min=1)

    def mean(x: torch.Tensor) -> torch.Tensor:
        return (x * mask).sum(1) / n

    log_ppl_obs = mean(-obs.gather(-1, target).squeeze(-1))
    log_ppl_perf = mean(-perf.gather(-1, target).squeeze(-1))
    log_xppl = mean(-(obs.exp() * perf).sum(-1))
    return {"n_tokens": n, "log_ppl_obs": log_ppl_obs, "log_ppl_perf": log_ppl_perf, "log_xppl": log_xppl,
            "score": log_ppl_perf / log_xppl}


@torch.no_grad()
def score_texts(texts: list[str], observer: Path, performer: Path, max_length: int = 512, batch_size: int = 8,
                device: str | None = None) -> pd.DataFrame:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = get_device(device)
    tok = AutoTokenizer.from_pretrained(performer)
    if tok.get_vocab() != AutoTokenizer.from_pretrained(observer).get_vocab():
        raise SystemExit("observer and performer must share a tokenizer")
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "right"
    models = [AutoModelForCausalLM.from_pretrained(m, dtype=torch.bfloat16).to(dev).eval() for m in (observer, performer)]

    enc = tok(texts, truncation=True, max_length=max_length)["input_ids"]
    order = np.argsort([len(e) for e in enc])[::-1]  # longest first: any out-of-memory error shows up at once
    rows: list[dict] = [{}] * len(texts)
    t0 = time.perf_counter()
    for b, start in enumerate(range(0, len(order), batch_size), 1):
        idx = order[start: start + batch_size]
        batch = tok.pad({"input_ids": [enc[i] for i in idx]}, return_tensors="pt").to(dev)
        obs_logits, perf_logits = (m(**batch).logits for m in models)
        feats = binoculars_features(obs_logits, perf_logits, batch["input_ids"], batch["attention_mask"])
        for j, i in enumerate(idx):
            rows[i] = {k: float(v[j]) for k, v in feats.items()}
        if b % 50 == 0:
            print(f"{start + len(idx)}/{len(texts)} passages ({time.perf_counter() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows).astype({"n_tokens": int})


def report() -> None:
    """AUROC of each zero-shot feature (lower = AI, so scores are negated) overall, on validation, per genre."""
    data = load_labeled(DATA_DIR / "train.csv")
    df = pd.read_csv(BINO_DIR / "train.csv").merge(pd.DataFrame({"id": data.ids, "text": data.texts}), on="id")
    df["genre"] = df["text"].map(genre)
    val = df[df["id"].isin(load_split(data).val_ids)]

    def aucs(d: pd.DataFrame) -> dict[str, float]:
        return {f: round(roc_auc_score(d["label"], -d[f]), 5) for f in FEATURES}

    print("all train.csv", len(df), aucs(df))
    print("validation   ", len(val), aucs(val))
    for g, d in df.groupby("genre"):
        print(f"{g:<13}", len(d), f"ai={d['label'].mean():.2f}", aucs(d))

    test_path = BINO_DIR / "test.csv"
    if test_path.exists():
        ids, texts = load_unlabeled(DATA_DIR / "test.csv")
        test = pd.read_csv(test_path).merge(pd.DataFrame({"id": ids, "genre": [genre(t) for t in texts]}), on="id")
        print("median score by genre: train human / train AI / test")
        for g in sorted(df["genre"].unique()):
            tr, te = df[df["genre"] == g], test[test["genre"] == g]
            print(f"{g:<13} {tr[tr.label == 0]['score'].median():.3f} / {tr[tr.label == 1]['score'].median():.3f}"
                  f" / {te['score'].median():.3f} (n_test={len(te)})")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sets", nargs="+", choices=("train", "test", "unlabeled"), default=["train", "test"])
    p.add_argument("--observer", type=Path, default=Path("models/falcon-7b"))
    p.add_argument("--performer", type=Path, default=Path("models/falcon-7b-instruct"))
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=None, help="smoke test: first N passages per set (writes nothing)")
    p.add_argument("--report-only", action="store_true")
    a = p.parse_args(argv)

    if not a.report_only:
        BINO_DIR.mkdir(parents=True, exist_ok=True)
        for name in a.sets:
            out = BINO_DIR / f"{name}.csv"
            if out.exists() and a.limit is None:
                print(f"skip {name} ({out} exists)", flush=True)
                continue
            if name == "train":
                data = load_labeled(DATA_DIR / "train.csv")
                ids, texts, labels = data.ids, data.texts, data.labels
            else:
                ids, texts = load_unlabeled(DATA_DIR / f"{name}.csv")
                labels = None
            ids, texts = ids[: a.limit], texts[: a.limit]
            print(f"[{name}] scoring {len(texts)} passages", flush=True)
            feats = score_texts(texts, a.observer, a.performer, a.max_length, a.batch_size, a.device)
            feats.insert(0, "id", ids)
            if labels is not None:
                feats["label"] = labels[: a.limit]
            if a.limit is not None:
                print(feats.describe().round(3).to_string())
                if labels is not None and feats["label"].nunique() > 1:
                    print("smoke AUROC (-score):", round(roc_auc_score(feats["label"], -feats["score"]), 4))
                continue
            feats.to_csv(out, index=False)
            print(f"[{name}] wrote {out}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if (BINO_DIR / "train.csv").exists():
        report()


if __name__ == "__main__":
    main()
