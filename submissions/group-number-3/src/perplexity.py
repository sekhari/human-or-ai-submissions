"""Perplexity-style features from a small pretrained language model (default: gpt2,
124M parameters, MIT license), then a gradient-boosting classifier on them.

For every passage (newlines collapsed to spaces, first 256 tokens) the LM scores
each token. We summarise how "predictable" the passage is to the LM:

  mean/std/percentiles of token log-probability, mean token entropy,
  mean log-rank of the true token, share of tokens that were the LM's top-1 /
  top-10 guess, and the gap between surprise and entropy (a Binoculars-style
  ratio). AI text tends to be unusually predictable; these signals do not depend
  on topic words or formatting.

Features are cached in results/ppl_features.csv (train + test), so only the
first run is slow.

    python src/perplexity.py --mode val    # train fold -> fixed 2,000-row val set
    python src/perplexity.py --mode full   # all 10,000 rows -> sub_ppl_gbm.csv
"""
import argparse
import json
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from data import ROOT, get_split, load_splits, load_test, load_train

SEED = 42
CACHE = ROOT / "results" / "ppl_features.csv"
FEATURES = ["lp_mean", "lp_std", "lp_p10", "lp_p25", "lp_median", "lp_p90",
            "entropy_mean", "logrank_mean", "top1_frac", "top10_frac",
            "surprise_minus_entropy", "binoculars_ratio", "n_tokens"]


def pick_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def score_texts(texts, model_name, max_len, batch):
    # imported here so cached runs never load torch (avoids an OpenMP clash with sklearn on macOS)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_grad_enabled(False)
    device = pick_device()
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    print(f"Scoring {len(texts)} passages with {model_name} on {device}", flush=True)

    rows, start = [], time.time()
    for i in range(0, len(texts), batch):
        chunk = [re.sub(r"\s+", " ", t).strip() for t in texts[i:i + batch]]
        enc = tok(chunk, return_tensors="pt", truncation=True, max_length=max_len,
                  padding=True).to(device)
        logits = model(**enc).logits[:, :-1].float()
        target = enc["input_ids"][:, 1:]
        mask = enc["attention_mask"][:, 1:].bool()

        logp = torch.log_softmax(logits, dim=-1)
        tok_lp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        entropy = -(logp.exp() * logp).sum(-1)
        rank = (logp > tok_lp.unsqueeze(-1)).sum(-1) + 1  # 1 = LM's top guess

        for j in range(len(chunk)):
            m = mask[j]
            lp = tok_lp[j][m].cpu().numpy()
            if len(lp) < 2:
                lp = np.array([0.0, 0.0])
            ent = entropy[j][m].cpu().numpy()
            rk = rank[j][m].cpu().numpy()
            surprise = -lp.mean()
            rows.append({
                "lp_mean": lp.mean(), "lp_std": lp.std(),
                "lp_p10": np.percentile(lp, 10), "lp_p25": np.percentile(lp, 25),
                "lp_median": np.median(lp), "lp_p90": np.percentile(lp, 90),
                "entropy_mean": ent.mean() if len(ent) else 0.0,
                "logrank_mean": np.log(rk).mean() if len(rk) else 0.0,
                "top1_frac": (rk == 1).mean() if len(rk) else 0.0,
                "top10_frac": (rk <= 10).mean() if len(rk) else 0.0,
                "surprise_minus_entropy": surprise - (ent.mean() if len(ent) else 0.0),
                "binoculars_ratio": surprise / (ent.mean() + 1e-6) if len(ent) else 0.0,
                "n_tokens": int(m.sum()),
            })
        done = i + len(chunk)
        if (i // batch) % 25 == 0:
            rate = done / (time.time() - start)
            print(f"  {done}/{len(texts)}  ({rate:.0f} passages/s, "
                  f"~{(len(texts) - done) / rate / 60:.1f} min left)", flush=True)
    return pd.DataFrame(rows)


def get_features(model_name, max_len, batch):
    if CACHE.exists():
        df = pd.read_csv(CACHE)
        if df.attrs.get("model", model_name) == model_name:
            print(f"Loaded cached features from {CACHE.relative_to(ROOT)}")
            return df
    train, test = load_train(), load_test()
    texts = train["text"].tolist() + test["text"].tolist()
    feats = score_texts(texts, model_name, max_len, batch)
    feats.insert(0, "split", ["train"] * len(train) + ["test"] * len(test))
    feats.insert(0, "id", train["id"].tolist() + test["id"].tolist())
    feats.to_csv(CACHE, index=False)
    print(f"Saved {CACHE.relative_to(ROOT)}")
    return feats


def make_gbm():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["val", "full"], default="val")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()

    feats = get_features(args.model, args.max_len, args.batch)
    train = load_train()
    tr_feats = train[["id", "label"]].merge(feats[feats.split == "train"], on="id")

    if args.mode == "val":
        splits = load_splits()
        tr, va = get_split(train, splits, "random_fold")
        Xtr = tr[["id", "label"]].merge(tr_feats.drop(columns="label"), on="id")
        Xva = va[["id", "label"]].merge(tr_feats.drop(columns="label"), on="id")

        print("\nSingle-feature validation AUROC (0.5 = useless; far from 0.5 = strong):")
        for f in FEATURES:
            auc = roc_auc_score(Xva["label"], Xva[f])
            print(f"  {f:<24} {max(auc, 1 - auc):.4f}  ({'higher' if auc > 0.5 else 'lower'} = AI)")

        gbm = make_gbm().fit(Xtr[FEATURES], Xtr["label"])
        tr_auc = roc_auc_score(Xtr["label"], gbm.predict_proba(Xtr[FEATURES])[:, 1])
        va_p = gbm.predict_proba(Xva[FEATURES])[:, 1]
        va_auc = roc_auc_score(Xva["label"], va_p)
        print(f"\nPerplexity GBM: train AUROC = {tr_auc:.4f}  val AUROC = {va_auc:.4f}")

        pd.DataFrame({"id": Xva["id"], "label": Xva["label"], "p": va_p}).to_csv(
            ROOT / "results" / "val_probs_ppl_gbm.csv", index=False)
        with open(ROOT / "results" / "val_results.jsonl", "a") as f:
            f.write(json.dumps({"model": "ppl_gbm", "lm": args.model,
                                "timestamp": datetime.now().isoformat(timespec="seconds"),
                                "random_fold_auroc": round(va_auc, 4),
                                "train_auroc": round(tr_auc, 4)}) + "\n")
        print("Saved results/val_probs_ppl_gbm.csv and logged to results/val_results.jsonl")

    else:
        test = load_test()
        te = test[["id"]].merge(feats[feats.split == "test"], on="id")
        gbm = make_gbm().fit(tr_feats[FEATURES], tr_feats["label"])
        sub = pd.DataFrame({"id": te["id"], "label": gbm.predict_proba(te[FEATURES])[:, 1]})
        assert len(sub) == 3000 and sub["id"].is_unique and sub["label"].between(0, 1).all()
        sub.to_csv(ROOT / "sub_ppl_gbm.csv", index=False)
        print("Wrote sub_ppl_gbm.csv (3000 rows)")


if __name__ == "__main__":
    main()
