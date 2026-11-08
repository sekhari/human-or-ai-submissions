"""Fine-tune a small transformer and cache its scores (settings fixed in advance, not tuned on validation).

Each job trains one model and saves log-odds under cache/transformer/<model>/; finished jobs are
skipped, so an interrupted run resumes.

Jobs:
    val    fit on the 8,000-row training partition, score the fixed validation set
    rare   fit on the training partition minus the fiction/poetry clusters, score those clusters
    oof8k  3-fold out-of-fold scores inside the training partition (stacking inputs for validation)
    test   fit on all 10,000 labeled rows, score test.csv
    oof10k 3-fold out-of-fold scores on all 10,000 rows (stacking inputs for the submission)

Usage:
    python -m experiments.finetune --jobs val
    python -m experiments.finetune --jobs val oof8k test oof10k
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from experiments.robustness import load_domains
from src.data import load_split
from src.neural import DEVICE, _batches
from src.split import load_train_val

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
N_FOLDS = 3


def train_model(model_name, texts, y, epochs, max_length, batch_size, lr, seed=SEED):
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(texts) / batch_size)
    schedule = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.float32)
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(texts))
        total = 0.0
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True,
                      max_length=max_length).to(DEVICE)
            loss = loss_fn(model(**enc).logits.squeeze(-1), torch.tensor(y[idx], device=DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
            total += loss.item() * len(idx)
        print(f"    epoch {epoch + 1}: train loss {total / len(texts):.4f}", flush=True)
    return tok, model


@torch.no_grad()
def predict(tok, model, texts, max_length, batch_size=32):
    model.eval()
    out = np.zeros(len(texts))
    for idx, chunk in _batches(texts, batch_size):
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(DEVICE)
        out[idx] = model(**enc).logits.squeeze(-1).cpu().numpy()
    return out


def run_fit(path, args, fit_texts, y, eval_texts):
    """Train once and save scores for eval_texts, unless a previous run already did."""
    if path.exists():
        return np.load(path)
    t0 = time.time()
    tok, model = train_model(args.model, fit_texts, y, args.epochs, args.max_length, args.batch_size, args.lr)
    scores = predict(tok, model, eval_texts, args.max_length)
    np.save(path, scores)
    print(f"  {path.name}: {time.time() - t0:.0f}s", flush=True)
    del model
    torch.mps.empty_cache() if DEVICE == "mps" else None
    return scores


def run_oof(out_dir, prefix, args, df):
    texts, y = df.text.tolist(), df.label.astype(int).values
    oof = np.zeros(len(df))
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (fit, held) in enumerate(cv.split(texts, y)):
        oof[held] = run_fit(out_dir / f"{prefix}_fold{fold}.npy", args,
                            [texts[i] for i in fit], y[fit], [texts[i] for i in held])
    np.savez(out_dir / f"{prefix}.npz", id=df.id.values, score=oof)
    print(f"{prefix} OOF AUROC {roc_auc_score(y, oof):.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="distilroberta-base")
    ap.add_argument("--jobs", nargs="+", choices=["val", "rare", "oof8k", "test", "oof10k"], default=["val"])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()
    out_dir = ROOT / "cache" / "transformer" / args.model.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    train, val = load_train_val()
    if "val" in args.jobs:
        s = run_fit(out_dir / "val.npy", args, train.text.tolist(), train.label.astype(int).values, val.text.tolist())
        print(f"val AUROC {roc_auc_score(val.label.astype(int), s):.4f}", flush=True)
    if "rare" in args.jobs:
        labeled = load_domains()[2]
        labeled = labeled[~labeled.id.isin(val.id)]  # same fold as the rare_domain column in experiments.stacking
        source, rare = labeled[labeled.domain != "rare_domain"], labeled[labeled.domain == "rare_domain"]
        s = run_fit(out_dir / "rare.npy", args, source.text.tolist(), source.label.astype(int).values, rare.text.tolist())
        print(f"rare_domain AUROC {roc_auc_score(rare.label.astype(int), s):.4f} (n={len(rare)})", flush=True)
    if "oof8k" in args.jobs:
        run_oof(out_dir, "oof8k", args, train)
    full = load_split("train")
    if "test" in args.jobs:
        run_fit(out_dir / "test.npy", args, full.text.tolist(), full.label.astype(int).values,
                load_split("test").text.tolist())
    if "oof10k" in args.jobs:
        run_oof(out_dir, "oof10k", args, full)


if __name__ == "__main__":
    main()
