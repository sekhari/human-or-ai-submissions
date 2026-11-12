"""Fine-tune a pretrained transformer (default: distilroberta-base, Apache-2.0)
for human-vs-AI classification. Runs on Apple Silicon GPU (MPS), CUDA, or CPU.

Two modes:
  --mode val   train on the 8,000-row training fold, score the fixed 2,000-row
               validation set (same protocol as every other model), save val
               probabilities for ensembling, log to results/val_results.jsonl
  --mode full  train on all 10,000 labeled rows and write a Kaggle submission

By default newlines are replaced with spaces before tokenizing, so the model
cannot learn the blank-line formatting shortcut we found (which did not
transfer to the test set). Use --keep_newlines to disable this.

    python src/finetune_transformer.py --mode val
    python src/finetune_transformer.py --mode full
"""
import argparse
import json
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)

from data import ROOT, get_split, load_splits, load_test, load_train

SEED = 42


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def clean(texts, keep_newlines):
    texts = list(texts)
    return texts if keep_newlines else [re.sub(r"\s+", " ", t).strip() for t in texts]


def make_loader(tok, texts, labels, max_len, batch, shuffle):
    enc = tok(texts, truncation=True, max_length=max_len)
    items = [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i],
              **({"labels": int(labels[i])} if labels is not None else {})}
             for i in range(len(texts))]
    g = torch.Generator().manual_seed(SEED)
    return DataLoader(items, batch_size=batch, shuffle=shuffle, generator=g,
                      collate_fn=DataCollatorWithPadding(tok))


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    for b in loader:
        b = {k: v.to(device) for k, v in b.items() if k != "labels"}
        logits = model(**b).logits
        probs.append(torch.softmax(logits, dim=-1)[:, 1].float().cpu().numpy())
    return np.concatenate(probs)


def train(model, loader, device, epochs, lr, eval_fn=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = epochs * len(loader)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        start, running = time.time(), 0.0
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            loss = model(**b).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            running += loss.item()
            step += 1
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{total}  loss {running / 50:.4f}  "
                      f"({time.time() - start:.0f}s)", flush=True)
                running = 0.0
        msg = f"Epoch {epoch} done in {time.time() - start:.0f}s"
        if eval_fn:
            msg += f"  |  val AUROC = {eval_fn():.4f}"
        print(msg, flush=True)


def main():
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["val", "full"], default="val")
    p.add_argument("--model", default="distilroberta-base")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--keep_newlines", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    global SEED
    SEED = args.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = pick_device()
    tag = (args.model.rstrip("/").split("/")[-1] + ("_nl" if args.keep_newlines else "")
           + (f"_s{args.seed}" if args.seed != 42 else ""))
    print(f"Device: {device} | model: {args.model} | mode: {args.mode} | "
          f"epochs={args.epochs} lr={args.lr} batch={args.batch} max_len={args.max_len} "
          f"keep_newlines={args.keep_newlines}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)
    train_df, splits = load_train(), load_splits()

    if args.mode == "val":
        tr, va = get_split(train_df, splits, "random_fold")
        tr_texts, va_texts = clean(tr["text"], args.keep_newlines), clean(va["text"], args.keep_newlines)
        tr_loader = make_loader(tok, tr_texts, tr["label"].values, args.max_len, args.batch, True)
        va_loader = make_loader(tok, va_texts, None, args.max_len, args.batch * 2, False)

        start = time.time()
        train(model, tr_loader, device, args.epochs, args.lr,
              eval_fn=lambda: roc_auc_score(va["label"], predict(model, va_loader, device)))
        train_secs = time.time() - start

        va_p = predict(model, va_loader, device)
        tr_eval = make_loader(tok, tr_texts, None, args.max_len, args.batch * 2, False)
        tr_p = predict(model, tr_eval, device)
        va_auc, tr_auc = roc_auc_score(va["label"], va_p), roc_auc_score(tr["label"], tr_p)
        print(f"\nFinal: train AUROC = {tr_auc:.4f}  val AUROC = {va_auc:.4f}  "
              f"train time = {train_secs / 60:.1f} min")

        pd.DataFrame({"id": va["id"].values, "label": va["label"].values, "p": va_p}).to_csv(
            ROOT / "results" / f"val_probs_{tag}.csv", index=False)
        with open(ROOT / "results" / "val_results.jsonl", "a") as f:
            f.write(json.dumps({"model": tag, "timestamp": datetime.now().isoformat(timespec="seconds"),
                                "random_fold_auroc": round(va_auc, 4), "train_auroc": round(tr_auc, 4),
                                "train_time_s": round(train_secs), **vars(args)}) + "\n")
        print(f"Saved results/val_probs_{tag}.csv and logged to results/val_results.jsonl")

    else:
        test = load_test()
        tr_loader = make_loader(tok, clean(train_df["text"], args.keep_newlines),
                                train_df["label"].values, args.max_len, args.batch, True)
        te_loader = make_loader(tok, clean(test["text"], args.keep_newlines), None,
                                args.max_len, args.batch * 2, False)
        train(model, tr_loader, device, args.epochs, args.lr)
        sub = pd.DataFrame({"id": test["id"], "label": predict(model, te_loader, device)})
        assert len(sub) == 3000 and sub["id"].is_unique and sub["label"].between(0, 1).all()
        out = ROOT / f"sub_{tag}.csv"
        sub.to_csv(out, index=False)
        print(f"Wrote {out.relative_to(ROOT)} ({len(sub)} rows)")


if __name__ == "__main__":
    main()
