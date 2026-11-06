"""Fine-tune DeBERTa-v3-small as a Human-vs-AI classifier.

Run on a GPU (Colab T4 is enough; Apple-Silicon MPS works but is slower).
Each mode caches predictions to results/preds/<name>.npz and skips work
that is already done, so a disconnected Colab can simply re-run.

Modes
  main    train on the training partition, predict train/val/unlabeled  (Q1, Q4)
  lc      learning curve at 500/1k/2k/5k/8k rows, same hyperparameters    (Q2)
  pseudo  add confident pseudo-labels from unlabeled.csv and retrain      (Q3)
  final   retrain on all 10,000 labeled rows, predict test.csv            (Kaggle)
  all     main -> lc -> pseudo -> final

Usage:  python train_deberta.py --mode all
        # stronger variant for the Q5 improvement / Kaggle (main + final only):
        python train_deberta.py --mode main --tag base512 --model-name microsoft/deberta-v3-base \
            --max-len 512 --batch-size 8 --grad-accum 2
        python train_deberta.py --mode final --tag base512 ...same flags...
"""
from __future__ import annotations

import argparse
import gc
import random
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.common import SEED, Data, auc, load_preds, save_preds, update_results

CFG = dict(model_name="microsoft/deberta-v3-small", max_len=384, batch_size=16, epochs=2,
           lr=3e-5, weight_decay=0.01, warmup_frac=0.1, pseudo_thresh=0.95,
           lc_sizes=[500, 1000, 2000, 5000, 8000], grad_accum=1, seed=SEED, tag="")


def pname(cfg: dict, what: str) -> str:
    """Cache name: deberta_<what> for the default run, <tag>_<what> for tagged variants."""
    return f"{cfg['tag'] or 'deberta'}_{what}"


def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Trainer:
    def __init__(self, cfg: dict, use_amp: bool = True):
        from transformers import AutoTokenizer, DataCollatorWithPadding
        self.cfg = cfg
        self.device = pick_device()
        self.amp = use_amp and self.device == "cuda"
        self.tok = AutoTokenizer.from_pretrained(cfg["model_name"])
        self.collate = DataCollatorWithPadding(self.tok)
        print(f"device={self.device} amp={self.amp}")

    def _ds(self, texts, labels=None):
        enc = self.tok(list(texts), truncation=True, max_length=self.cfg["max_len"])
        items = []
        for i in range(len(enc["input_ids"])):
            item = {"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i]}
            if labels is not None:
                item["labels"] = int(labels[i])
            items.append(item)
        return items

    def predict(self, model, texts, bs: int = 64) -> np.ndarray:
        dl = DataLoader(self._ds(texts), batch_size=bs, shuffle=False, collate_fn=self.collate)
        model.eval(); out = []
        with torch.no_grad():
            for b in dl:
                b = {k: v.to(self.device) for k, v in b.items()}
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                    logits = model(**b).logits
                out.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    def train(self, texts, labels, seed: int | None = None):
        from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
        c = self.cfg
        seed = c["seed"] if seed is None else seed
        set_seed(seed)
        model = AutoModelForSequenceClassification.from_pretrained(c["model_name"], num_labels=2)
        model = model.float().to(self.device)  # checkpoint is fp16; newer transformers keep it, GradScaler needs fp32 master weights
        g = torch.Generator(); g.manual_seed(seed)
        dl = DataLoader(self._ds(texts, labels), batch_size=c["batch_size"], shuffle=True,
                        collate_fn=self.collate, generator=g)
        opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
        acc = c["grad_accum"]
        total = -(-len(dl) // acc) * c["epochs"]
        sch = get_linear_schedule_with_warmup(opt, int(c["warmup_frac"] * total), total)
        scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        t0 = time.time()
        for ep in range(c["epochs"]):
            model.train(); tot = 0.0
            opt.zero_grad()
            for i, b in enumerate(dl):
                b = {k: v.to(self.device) for k, v in b.items()}
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                    loss = model(**b).loss
                scaler.scale(loss / acc).backward(); tot += loss.item()
                if (i + 1) % acc == 0 or i + 1 == len(dl):
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt); scaler.update(); sch.step(); opt.zero_grad()
            print(f"  epoch {ep + 1}/{c['epochs']} n={len(texts)} loss={tot / len(dl):.4f} ({time.time() - t0:.0f}s)")
        return model, time.time() - t0


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def mode_main(T: Trainer, d: Data):
    p = load_preds(pname(T.cfg, "main"))
    if p is None:
        model, tt = T.train(d.X_tr, d.y_tr)
        p = dict(p_tr=T.predict(model, d.X_tr), p_va=T.predict(model, d.X_va),
                 p_unl=T.predict(model, d.X_unl) if d.X_unl is not None else np.array([]),
                 train_time=np.array(tt))
        save_preds(pname(T.cfg, "main"), **p); del model; cleanup()
    out = dict(train_auc=auc(d.y_tr, p["p_tr"]), val_auc=auc(d.y_va, p["p_va"]),
               train_time_s=float(p["train_time"]), device=T.device, config=T.cfg)
    update_results(f"model:{T.cfg['tag'] or 'deberta_v3_small'}", out); print("deberta main:", out)
    return p


def mode_lc(T: Trainer, d: Data):
    main = mode_main(T, d)
    lc = {}
    for n in T.cfg["lc_sizes"]:
        n_eff = min(n, len(d.tr))
        if n_eff == len(d.tr):
            p, y_tr = main, d.y_tr
        else:
            p = load_preds(f"lc_{n}")
            if p is None:
                idx, _ = train_test_split(np.arange(len(d.tr)), train_size=n_eff,
                                          stratify=d.y_tr, random_state=SEED)
                Xs, ys = d.X_tr[idx], d.y_tr[idx]
                model, tt = T.train(Xs, ys)
                p = dict(p_tr=T.predict(model, Xs), y_tr=ys, p_va=T.predict(model, d.X_va),
                         train_time=np.array(tt))
                save_preds(f"lc_{n}", **p); del model; cleanup()
            y_tr = p["y_tr"]
        lc[str(n)] = dict(n_used=n_eff, train_auc=auc(y_tr, p["p_tr"]), val_auc=auc(d.y_va, p["p_va"]),
                          train_time_s=float(p["train_time"]))
        print(n, lc[str(n)])
    update_results("learning_curve", lc)


def mode_pseudo(T: Trainer, d: Data):
    main = mode_main(T, d)
    if d.X_unl is None or not len(main["p_unl"]):
        print("unlabeled.csv missing; skipping"); return
    p_unl, th = main["p_unl"], T.cfg["pseudo_thresh"]
    conf = (p_unl >= th) | (p_unl <= 1 - th)
    pl_y = (p_unl[conf] >= 0.5).astype(int)
    p = load_preds("deberta_pseudo")
    if p is None:
        model, tt = T.train(np.concatenate([d.X_tr, d.X_unl[conf]]), np.concatenate([d.y_tr, pl_y]))
        p = dict(p_va=T.predict(model, d.X_va), train_time=np.array(tt))
        save_preds("deberta_pseudo", **p); del model; cleanup()
    out = dict(threshold=th, n_pseudo=int(conf.sum()), n_unlabeled=int(len(p_unl)),
               pseudo_ai_frac=float(pl_y.mean()) if len(pl_y) else float("nan"),
               val_auc_supervised=auc(d.y_va, main["p_va"]), val_auc_with_unlabeled=auc(d.y_va, p["p_va"]),
               train_time_s=float(p["train_time"]))
    update_results("pseudo_label", out); print("pseudo:", out)


def mode_final(T: Trainer, d: Data):
    if d.X_test is None:
        raise FileNotFoundError("test.csv missing")
    p = load_preds(pname(T.cfg, "final_test"))
    if p is None:
        X, y = d.full[d.text].values, d.full[d.label].values
        model, tt = T.train(X, y)
        p = dict(p_test=T.predict(model, d.X_test), train_time=np.array(tt))
        save_preds(pname(T.cfg, "final_test"), **p); del model; cleanup()
    update_results(pname(T.cfg, "final"), dict(n_train=len(d.full), train_time_s=float(p["train_time"])))
    print("final test preds cached:", p["p_test"].shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["main", "lc", "pseudo", "final", "all"], default="all")
    ap.add_argument("--no-amp", action="store_true", help="disable fp16 (use if loss becomes nan)")
    ap.add_argument("--epochs", type=int, default=CFG["epochs"])
    ap.add_argument("--tag", default="", help="cache/results prefix for a variant (main/final modes only)")
    ap.add_argument("--model-name", default=CFG["model_name"])
    ap.add_argument("--max-len", type=int, default=CFG["max_len"])
    ap.add_argument("--batch-size", type=int, default=CFG["batch_size"])
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=CFG["lr"])
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    cfg = dict(CFG, epochs=args.epochs, tag=args.tag, model_name=args.model_name, max_len=args.max_len,
               batch_size=args.batch_size, grad_accum=args.grad_accum, lr=args.lr, seed=args.seed)
    modes = ["main", "lc", "pseudo", "final"] if args.mode == "all" else [args.mode]
    if args.tag and set(modes) - {"main", "final"}:
        raise SystemExit("--tag variants support only --mode main / final")
    d = Data(); T = Trainer(cfg, use_amp=not args.no_amp)
    for m in modes:
        print(f"===== {m} ====="); globals()[f"mode_{m}"](T, d)


if __name__ == "__main__":
    main()
