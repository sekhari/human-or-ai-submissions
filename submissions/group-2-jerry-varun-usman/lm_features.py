"""Zero-shot language-model features (Binoculars-style) and a small LR on top (report Q5).

Motivation: about a third of test.csv is poetry / fiction that train.csv barely covers,
and fine-tuned DeBERTa is close to guessing there. Perplexity-type scores from a frozen
LM do not depend on topic, so they should transfer to unseen genres.

Two frozen LMs that share a tokenizer:
  observer  = Qwen/Qwen2.5-1.5B,  performer = Qwen/Qwen2.5-1.5B-Instruct
Per passage (means over predicted tokens):
  nll_obs, nll_perf   token negative log-likelihood under each model (log-perplexity)
  xent                cross-perplexity: -sum softmax(observer) * log_softmax(performer)
  bino                nll_perf / xent  (Binoculars, Hans et al. 2024; lower = more AI-like)
  logrank_obs         mean log rank of the true token under the observer
  ent_obs             mean entropy of the observer's next-token distribution
  log_ntok            log number of tokens
A logistic regression on these 7 features is tuned by 3-fold CV on the training partition
(same protocol as baselines.py), scored once on the fixed validation split, then refit on
train+val to predict test. Writes results/preds/lm_main.npz and lm_final_test.npz, which
blend.py picks up as the variant `lm`. Features are cached per split (lm_feats_<split>.npz).

Usage (GPU):  python lm_features.py
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common import SEED, Data, auc, load_preds, save_preds, update_results

FEATS = ["nll_obs", "nll_perf", "xent", "bino", "logrank_obs", "ent_obs", "log_ntok"]


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Scorer:
    def __init__(self, observer: str, performer: str, max_len: int, batch_size: int):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = pick_device()
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(observer)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"
        load = lambda n: AutoModelForCausalLM.from_pretrained(n, torch_dtype=dtype).to(self.device).eval()
        self.obs, self.perf = load(observer), load(performer)
        self.max_len, self.bs = max_len, batch_size
        print(f"device={self.device} dtype={dtype} observer={observer} performer={performer}")

    @torch.no_grad()
    def features(self, texts) -> np.ndarray:
        texts = list(texts)
        order = np.argsort([len(t) for t in texts])  # length-sorted batches: less padding
        out = np.zeros((len(texts), len(FEATS)), dtype=np.float32)
        t0 = time.time()
        for s in range(0, len(texts), self.bs):
            idx = order[s:s + self.bs]
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                           truncation=True, max_length=self.max_len).to(self.device)
            lo = self.obs(**enc).logits[:, :-1]
            lp = self.perf(**enc).logits[:, :-1]
            tgt = enc["input_ids"][:, 1:]
            mask = enc["attention_mask"][:, 1:].bool()
            for j, i in enumerate(idx):
                m = mask[j]
                n = int(m.sum())
                if n == 0:
                    continue
                o = lo[j][m].float(); p = lp[j][m].float(); y = tgt[j][m]
                lso, lsp = torch.log_softmax(o, -1), torch.log_softmax(p, -1)
                nll_o = -lso.gather(1, y[:, None]).squeeze(1)
                nll_p = -lsp.gather(1, y[:, None]).squeeze(1)
                po = lso.exp()
                xent = -(po * lsp).sum(-1)
                ent = -(po * lso).sum(-1)
                rank = (o > o.gather(1, y[:, None])).sum(-1).float() + 1
                nll_p_m, xent_m = nll_p.mean().item(), xent.mean().item()
                out[i] = [nll_o.mean().item(), nll_p_m, xent_m, nll_p_m / max(xent_m, 1e-6),
                          torch.log(rank).mean().item(), ent.mean().item(), float(np.log(n))]
            if (s // self.bs) % 50 == 0:
                print(f"  {s + len(idx)}/{len(texts)} ({time.time() - t0:.0f}s)", flush=True)
        return out


def get_feats(name: str, texts, scorer_fn):
    cached = load_preds(f"lm_feats_{name}")
    if cached is not None:
        return cached["F"]
    F = scorer_fn().features(texts)
    save_preds(f"lm_feats_{name}", F=F)
    return F


def lr_pipe():
    return Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=5000))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--observer", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--performer", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    d = Data()
    holder = {}

    def scorer():
        if "s" not in holder:
            holder["s"] = Scorer(args.observer, args.performer, args.max_len, args.batch_size)
        return holder["s"]

    t0 = time.time()
    F_tr = get_feats("tr", d.X_tr, scorer)
    F_va = get_feats("va", d.X_va, scorer)
    F_te = get_feats("test", d.X_test, scorer)
    feat_time = time.time() - t0

    zero_shot = {f: auc(d.y_va, -F_va[:, k]) for k, f in enumerate(FEATS)}  # low score = AI
    gs = GridSearchCV(lr_pipe(), {"clf__C": [0.01, 0.1, 1, 10]}, scoring="roc_auc",
                      cv=StratifiedKFold(3, shuffle=True, random_state=SEED))
    gs.fit(F_tr, d.y_tr)
    p_tr, p_va = gs.predict_proba(F_tr)[:, 1], gs.predict_proba(F_va)[:, 1]
    save_preds("lm_main", p_tr=p_tr, p_va=p_va)

    final = lr_pipe().set_params(**gs.best_params_)
    t1 = time.time()
    final.fit(np.vstack([F_tr, F_va]), np.concatenate([d.y_tr, d.y_va]))
    save_preds("lm_final_test", p_test=final.predict_proba(F_te)[:, 1])

    out = dict(train_auc=auc(d.y_tr, p_tr), val_auc=auc(d.y_va, p_va), train_time_s=time.time() - t1,
               feature_time_s=feat_time, best_params=gs.best_params_,
               cv_table=[{"params": {k: str(v) for k, v in p.items()}, "cv_auc": float(s)}
                         for p, s in zip(gs.cv_results_["params"], gs.cv_results_["mean_test_score"])],
               zero_shot_val_auc_negated=zero_shot, observer=args.observer, performer=args.performer,
               max_len=args.max_len)
    update_results("model:lm_binoculars_lr", out)
    print(f"LM-feature LR: train AUROC {out['train_auc']:.4f} | val AUROC {out['val_auc']:.4f} | "
          f"best {gs.best_params_}")
    print("zero-shot val AUROC per feature (lower = AI):", {k: round(v, 4) for k, v in zero_shot.items()})


if __name__ == "__main__":
    main()
