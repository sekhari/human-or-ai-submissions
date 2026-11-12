"""Stronger zero-shot features from TWO pretrained language models that share a
tokenizer: gpt2 (124M, "observer") and gpt2-medium (355M, "performer"), both MIT.

Per passage (newlines collapsed, first 256 tokens) we compute, for each LM:
  mean/std/percentiles of token log-prob, mean entropy, mean log-rank,
  top-1 / top-10 share
plus two-model Binoculars scores (Hans et al., 2024):
  binoculars = log-perplexity under the observer
               / cross-perplexity (performer's predicted distribution scored by the observer)
and the reverse direction. Binoculars normalises away how "surprising the topic is"
and is among the strongest published zero-shot detectors across unseen generators.

First run scores 13,000 passages (~30 min on an M4) and caches
results/ppl2_features.csv, then exits. Run the same command again to train.
(Torch and scikit-learn are never loaded in the same run: on macOS their OpenMP
libraries can clash and crash the process.)

    python src/perplexity2.py --mode val
    python src/perplexity2.py --mode full
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from data import ROOT, load_test, load_train

SEED = 42
CACHE = ROOT / "results" / "ppl2_features.csv"
OBSERVER, PERFORMER = "gpt2", "gpt2-medium"


def stats(prefix, lp, ent, rk):
    if len(lp) < 2:
        lp, ent, rk = np.zeros(2), np.zeros(2), np.ones(2)
    return {f"{prefix}_lp_mean": lp.mean(), f"{prefix}_lp_std": lp.std(),
            f"{prefix}_lp_p10": np.percentile(lp, 10), f"{prefix}_lp_p25": np.percentile(lp, 25),
            f"{prefix}_lp_median": np.median(lp), f"{prefix}_lp_p90": np.percentile(lp, 90),
            f"{prefix}_entropy": ent.mean(), f"{prefix}_logrank": np.log(rk).mean(),
            f"{prefix}_top1": (rk == 1).mean(), f"{prefix}_top10": (rk <= 10).mean()}


def compute_features(max_len, batch):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_grad_enabled(False)
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    tok = AutoTokenizer.from_pretrained(OBSERVER)
    tok.pad_token = tok.eos_token
    obs = AutoModelForCausalLM.from_pretrained(OBSERVER).to(device).eval()
    perf = AutoModelForCausalLM.from_pretrained(PERFORMER).to(device).eval()

    train, test = load_train(), load_test()
    ids = train["id"].tolist() + test["id"].tolist()
    texts = train["text"].tolist() + test["text"].tolist()
    print(f"Scoring {len(texts)} passages with {OBSERVER} + {PERFORMER} on {device}", flush=True)

    rows, start = [], time.time()
    for i in range(0, len(texts), batch):
        chunk = [re.sub(r"\s+", " ", t).strip() for t in texts[i:i + batch]]
        enc = tok(chunk, return_tensors="pt", truncation=True, max_length=max_len,
                  padding=True).to(device)
        target = enc["input_ids"][:, 1:]
        mask = enc["attention_mask"][:, 1:].bool()

        out = {}
        for name, model in [("obs", obs), ("perf", perf)]:
            logp = torch.log_softmax(model(**enc).logits[:, :-1].float(), dim=-1)
            tok_lp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            ent = -(logp.exp() * logp).sum(-1)
            rank = (logp > tok_lp.unsqueeze(-1)).sum(-1) + 1
            out[name] = (logp, tok_lp, ent, rank)

        logp_o, lp_o, ent_o, rk_o = out["obs"]
        logp_p, lp_p, ent_p, rk_p = out["perf"]
        # cross-entropies between the two models' next-token distributions
        x_obs = -(logp_p.exp() * logp_o).sum(-1)   # performer's dist scored by observer
        x_perf = -(logp_o.exp() * logp_p).sum(-1)  # observer's dist scored by performer
        del out, logp_o, logp_p

        for j in range(len(chunk)):
            m = mask[j]
            g = lambda t: t[j][m].cpu().numpy()
            a_lp, a_ent, a_rk = g(lp_o), g(ent_o), g(rk_o)
            b_lp, b_ent, b_rk = g(lp_p), g(ent_p), g(rk_p)
            xo, xp = g(x_obs), g(x_perf)
            row = {"id": ids[i + j], "split": "train" if i + j < len(train) else "test",
                   "n_tokens": int(m.sum())}
            row.update(stats("obs", a_lp, a_ent, a_rk))
            row.update(stats("perf", b_lp, b_ent, b_rk))
            ppl_o = -a_lp.mean() if len(a_lp) else 0.0
            ppl_p = -b_lp.mean() if len(b_lp) else 0.0
            row["binoculars"] = ppl_o / (xo.mean() + 1e-6) if len(xo) else 0.0
            row["binoculars_rev"] = ppl_p / (xp.mean() + 1e-6) if len(xp) else 0.0
            row["lp_gap_perf_minus_obs"] = (b_lp - a_lp).mean() if len(a_lp) else 0.0
            row["lp_gap_std"] = (b_lp - a_lp).std() if len(a_lp) > 1 else 0.0
            rows.append(row)

        done = i + len(chunk)
        if (i // batch) % 50 == 0:
            rate = done / (time.time() - start)
            print(f"  {done}/{len(texts)}  ({rate:.1f} passages/s, "
                  f"~{(len(texts) - done) / rate / 60:.1f} min left)", flush=True)

    pd.DataFrame(rows).to_csv(CACHE, index=False)
    print(f"Saved {CACHE.relative_to(ROOT)}.\n>>> Now run the same command again to train the model.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["val", "full"], default="val")
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args()

    if not CACHE.exists():
        compute_features(args.max_len, args.batch)
        sys.exit(0)

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from data import get_split, load_splits

    feats = pd.read_csv(CACHE)
    cols = [c for c in feats.columns if c not in ("id", "split")]
    train = load_train()
    trf = train[["id", "label"]].merge(feats[feats.split == "train"], on="id")
    gbm = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED)

    if args.mode == "val":
        tr, va = get_split(train, load_splits(), "random_fold")
        Xtr, Xva = trf[trf.id.isin(tr.id)], trf[trf.id.isin(va.id)]
        print("Strongest single features (validation AUROC):")
        single = sorted(((max(a, 1 - a), c, "higher" if a > 0.5 else "lower")
                         for c in cols for a in [roc_auc_score(Xva.label, Xva[c])]), reverse=True)
        for auc, c, d in single[:10]:
            print(f"  {c:<24} {auc:.4f}  ({d} = AI)")

        m = gbm().fit(Xtr[cols], Xtr.label)
        tr_auc = roc_auc_score(Xtr.label, m.predict_proba(Xtr[cols])[:, 1])
        va_p = m.predict_proba(Xva[cols])[:, 1]
        va_auc = roc_auc_score(Xva.label, va_p)
        print(f"\nPerplexity-2 GBM ({len(cols)} features): train AUROC = {tr_auc:.4f}  val AUROC = {va_auc:.4f}")
        pd.DataFrame({"id": Xva.id, "label": Xva.label, "p": va_p}).to_csv(
            ROOT / "results" / "val_probs_ppl2_gbm.csv", index=False)
        with open(ROOT / "results" / "val_results.jsonl", "a") as f:
            f.write(json.dumps({"model": "ppl2_gbm", "lms": [OBSERVER, PERFORMER],
                                "timestamp": datetime.now().isoformat(timespec="seconds"),
                                "random_fold_auroc": round(va_auc, 4), "train_auroc": round(tr_auc, 4)}) + "\n")
        print("Saved results/val_probs_ppl2_gbm.csv")
    else:
        test = load_test()
        te = test[["id"]].merge(feats[feats.split == "test"], on="id")
        m = gbm().fit(trf[cols], trf.label)
        sub = pd.DataFrame({"id": te.id, "label": m.predict_proba(te[cols])[:, 1]})
        assert len(sub) == 3000 and sub.id.is_unique
        sub.to_csv(ROOT / "sub_ppl2_gbm.csv", index=False)
        print("Wrote sub_ppl2_gbm.csv (3000 rows)")


if __name__ == "__main__":
    main()
