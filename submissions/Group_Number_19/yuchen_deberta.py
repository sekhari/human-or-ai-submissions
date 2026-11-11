"""Fine-tune a pretrained encoder (RoBERTa-base; DeBERTa-v3 hit NaN, see below) as a sequence classifier, with leave-one-topic-cluster-out (LCO) CV.

Unlike the frozen-LM features (yuchen_lm_features.py / yuchen_bino_features.py), here the
whole encoder is trained end-to-end on the 10,000 labeled texts. Newlines are replaced by
a visible marker because the DeBERTa tokenizer otherwise folds them into spaces, and
line/blank-line layout is a strong signal in this data.

For each of the 8 topic clusters k: train on clusters != k, predict cluster k (OOF) and
test. Then train on all labeled data and predict test + unlabeled.
Run on a bf16 GPU (A6000/A100):
  python yuchen_deberta.py --data data --clusters clusters.csv --out out/deberta
Note: with transformers 5.22, microsoft/deberta-v3-base produced a NaN loss at the second
optimizer step in both bf16 and fp32 (first loss ~0.70), so the runs use FacebookAI/roberta-base
(--model FacebookAI/roberta-base --nl-marker off).
Outputs: oof_lco.csv (id, cluster, label, pred), test_full.csv, test_lco_mean.csv,
unlabeled_full.csv, log of per-fold AUC.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

NL = " ¶ "


NL_MARKER = True   # set from --nl-marker; byte-level BPE models (RoBERTa) keep "\n" natively


def prep(texts: pd.Series) -> list[str]:
    texts = [t.replace("\r\n", "\n") for t in texts]
    return [t.replace("\n", NL) for t in texts] if NL_MARKER else texts


def batches_for_inference(enc_lens, bs):
    order = np.argsort(enc_lens)
    return [order[i:i + bs] for i in range(0, len(order), bs)]


@torch.no_grad()
def predict(model, tok, texts, args, device):
    model.eval()
    enc = tok(texts, truncation=True, max_length=args.max_len)
    ids = enc["input_ids"]
    out = np.zeros(len(texts))
    for idx in batches_for_inference([len(x) for x in ids], args.bs * 2):
        b = tok.pad({"input_ids": [ids[i] for i in idx]}, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp == "bf16"):
            logits = model(**b).logits.float()
        out[idx] = torch.softmax(logits, -1)[:, 1].cpu().numpy()
    return out


def train(texts, y, args, device, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)
    enc = tok(texts, truncation=True, max_length=args.max_len)["input_ids"]
    data = list(zip(enc, y.tolist()))

    def collate(b):
        p = tok.pad({"input_ids": [x for x, _ in b]}, return_tensors="pt")
        p["labels"] = torch.tensor([l for _, l in b])
        return p

    dl = DataLoader(data, batch_size=args.bs, shuffle=True, collate_fn=collate, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * len(dl) // args.accum
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    model.train(); t0 = time.time(); step = 0; micro = 0
    for ep in range(args.epochs):
        for b in dl:
            b = b.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp == "bf16"):
                loss = model(**b).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss {loss.item()} at step {step + 1} (amp={args.amp})")
            (loss / args.accum).backward()
            micro += 1
            if micro % args.accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True); step += 1
            if step in (1, 10) or step % 100 == 0:
                print(f"  step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    return model, tok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="microsoft/deberta-v3-base")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--clusters", type=Path, default=Path("clusters.csv"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--nl-marker", choices=("on", "off"), default="on",
                    help="replace newlines by ' ¶ ' (needed for DeBERTa's tokenizer; off for RoBERTa)")
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps (effective batch = bs * accum)")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", choices=("none", "bf16"), default="none",
                    help="bf16 autocast gave NaN loss from step 1 with transformers 5.22; fp32+TF32 is the default")
    ap.add_argument("--folds", type=int, nargs="*", default=None, help="subset of clusters (default all)")
    ap.add_argument("--skip-full", action="store_true")
    ap.add_argument("--fixed-split", type=Path, default=None,
                    help="folds.csv: train on folds 1-4, write fixed_val.csv / fixed_train.csv (report protocol) and stop")
    args = ap.parse_args()
    global NL_MARKER
    NL_MARKER = args.nl_marker == "on"
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"gpu {torch.cuda.get_device_name()}  model {args.model}", flush=True)
    _t = AutoTokenizer.from_pretrained(args.model)
    print("newline marker tokens:", _t.tokenize(prep(pd.Series(["Line one\n\nLine two"]))[0]), flush=True)
    tr, te, un = (pd.read_csv(args.data / f"{s}.csv") for s in ("train", "test", "unlabeled"))
    cl = pd.read_csv(args.clusters)
    fold = tr["id"].map(cl[cl["split"] == "train"].set_index("id")["cluster"]).to_numpy()
    assert not np.isnan(fold.astype(float)).any()
    y = tr["label"].to_numpy()
    X, Xte, Xun = prep(tr["text"]), prep(te["text"]), prep(un["text"])

    if args.fixed_split is not None:
        # Report protocol: train on folds 1-4 of folds.csv (8,000 rows), score fold 0 (2,000 rows)
        f = tr["id"].map(pd.read_csv(args.fixed_split).set_index("id")["fold"]).to_numpy()
        tri, vai = np.where(f != 0)[0], np.where(f == 0)[0]
        t = time.time()
        model, tok = train([X[i] for i in tri], y[tri], args, device, args.seed + 200)
        pv = predict(model, tok, [X[i] for i in vai], args, device)
        ptr = predict(model, tok, [X[i] for i in tri], args, device)
        pd.DataFrame({"id": tr["id"].iloc[vai].to_numpy(), "label": y[vai], "pred": pv}).to_csv(args.out / "fixed_val.csv", index=False)
        pd.DataFrame({"id": tr["id"].iloc[tri].to_numpy(), "label": y[tri], "pred": ptr}).to_csv(args.out / "fixed_train.csv", index=False)
        print(json.dumps(dict(fixed_split=True, n_train=len(tri), n_val=len(vai), train_auc=roc_auc_score(y[tri], ptr),
                              val_auc=roc_auc_score(y[vai], pv), seconds=round(time.time() - t))), flush=True)
        print("DONE", flush=True)
        return

    oof_path = args.out / "oof_lco.csv"
    oof = pd.read_csv(oof_path) if oof_path.exists() else pd.DataFrame(
        {"id": tr["id"], "cluster": fold, "label": y, "pred": np.nan})
    test_lco = {}
    for k in (args.folds if args.folds is not None else sorted(np.unique(fold))):
        if oof.loc[oof["cluster"] == k, "pred"].notna().all():
            print(f"fold {k}: done, skipping"); continue
        t = time.time()
        tri, vai = np.where(fold != k)[0], np.where(fold == k)[0]
        model, tok = train([X[i] for i in tri], y[tri], args, device, args.seed + k)
        pv = predict(model, tok, [X[i] for i in vai], args, device)
        oof.loc[vai, "pred"] = pv
        oof.to_csv(oof_path, index=False)
        test_lco[k] = predict(model, tok, Xte, args, device)
        np.save(args.out / f"test_lco{k}.npy", test_lco[k])
        auc = roc_auc_score(y[vai], pv) if len(np.unique(y[vai])) > 1 else float("nan")
        print(json.dumps(dict(fold=int(k), n_val=len(vai), auc=auc, seconds=round(time.time() - t))), flush=True)
        del model; torch.cuda.empty_cache()
    files = sorted(args.out.glob("test_lco*.npy"))
    if files:
        pd.DataFrame({"id": te["id"], "label": np.mean([np.load(f) for f in files], axis=0)}).to_csv(
            args.out / "test_lco_mean.csv", index=False)
    if oof["pred"].notna().all():
        print("OOF LCO pooled AUC", roc_auc_score(oof["label"], oof["pred"]), flush=True)
    if not args.skip_full:
        t = time.time()
        model, tok = train(X, y, args, device, args.seed + 100)
        pd.DataFrame({"id": te["id"], "label": predict(model, tok, Xte, args, device)}).to_csv(
            args.out / "test_full.csv", index=False)
        pd.DataFrame({"id": un["id"], "label": predict(model, tok, Xun, args, device)}).to_csv(
            args.out / "unlabeled_full.csv", index=False)
        print(f"full model: {time.time() - t:.0f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
