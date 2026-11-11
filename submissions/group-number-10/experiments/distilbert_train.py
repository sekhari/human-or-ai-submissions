"""Fine-tune DistilBERT on the shared 8,000/2,000 split and record results.

Examples (run from submissions/group-number-10/):
    python -m experiments.distilbert_train                               # baseline: lr 2e-5, bs 16, 3 epochs
    python -m experiments.distilbert_train --lr 3e-5 --epochs 4
    python -m experiments.distilbert_train --limit 64 --max-length 128 --epochs 1   # smoke test
    python -m experiments.distilbert_train --lr 3e-5 --epochs 4 --model-name distilbert-base-cased --newline-marker "[NL]"
    python -m experiments.distilbert_train ... --extra-train results/pseudo/large8k.csv   # self-training
    python -m experiments.distilbert_train ... --train-size 2000   # learning curve: balanced subset of the 8k

Outputs (full runs):
    results/metrics/<run>.json                    metrics + per-epoch history (committed)
    results/predictions/<run>_val.csv             id,label,prob_ai for the 2,000 validation rows
    results/distilbert_experiments.csv            one row per run (committed)
    checkpoints/distilbert/<run>/                 best-val-AUROC checkpoint (git-ignored)
Smoke runs (--limit) write under results/smoke/ and checkpoints/smoke/ instead.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.classifiers.distilbert import DistilBertClassifier, DistilBertConfig
from src.data import RESULTS_DIR, TEAM_ROOT, SplitData, load_split
from src.evaluate import Timer, record_experiment

CHECKPOINT_ROOT = TEAM_ROOT / "checkpoints"
EXPERIMENTS_CSV = RESULTS_DIR / "distilbert_experiments.csv"
CSV_FIELDS = ["run_name", "learning_rate", "epochs", "batch_size", "max_length", "seed", "best_epoch",
              "device", "train_time_seconds", "train_auc", "val_auc", "val_accuracy", "notes"]


def model_family(cfg: DistilBertConfig) -> str:
    """'distilbert' for DistilBERT-based models (including DAPT checkpoints), else e.g. 'deberta-v3-base', 'roberta-large', 'electra-large-discriminator'."""
    base = cfg.model_name.rstrip("/").split("/")[-1]
    return base if base.startswith(("deberta", "roberta", "electra")) else "distilbert"


def run_name_for(cfg: DistilBertConfig) -> str:
    # Default model + no marker keeps the original names (e.g. distilbert_lr2e-05_ep3_...).
    family = model_family(cfg)
    tag = ""
    if family == "distilbert" and cfg.model_name != DistilBertConfig().model_name:
        tag += "_" + cfg.model_name.split("/")[-1].removeprefix("distilbert-base-")
    if cfg.newline_marker:
        tag += "_nl"
    if cfg.space_marker:
        tag += "_ds"
    if cfg.warmup_ratio:
        tag += f"_wu{cfg.warmup_ratio:g}"
    if cfg.bf16:
        tag += "_bf16"
    if cfg.layout_aug:
        tag += f"_aug{cfg.layout_aug:g}"
    return f"{family}{tag}_lr{cfg.learning_rate:g}_ep{cfg.epochs}_bs{cfg.batch_size}_len{cfg.max_length}_seed{cfg.seed}"


def load_extra_train(path: Path | None) -> tuple[list[str], np.ndarray, str]:
    """Extra (e.g. pseudo-labeled) training passages from an id,text,label CSV and the run-name suffix."""
    if path is None:
        return [], np.zeros(0, dtype=int), ""
    df = pd.read_csv(path)
    return df["text"].astype(str).tolist(), df["label"].to_numpy(dtype=int), f"_pl-{Path(path).stem}"


def subsample_train(split: SplitData, n: int, seed: int = 42) -> SplitData:
    """Class-balanced n-row subset of the training split; validation is unchanged.

    Fixed seed and per-class permutations make the subsets nested (1k within 2k within 4k), so
    learning-curve points differ only by the added data, not by which rows were drawn.
    """
    rng = np.random.default_rng(seed)
    idx = np.sort(np.concatenate([rng.permutation(np.flatnonzero(split.train_labels == c))[: n // 2] for c in (0, 1)]))
    return SplitData(split.train_ids[idx], [split.train_texts[i] for i in idx], split.train_labels[idx],
                     split.val_ids, split.val_texts, split.val_labels, split.seed)


def balanced_subset(split: SplitData, n: int) -> SplitData:
    """First n/2 of each class from train and from val — for quick smoke tests only."""
    def pick(labels: np.ndarray) -> np.ndarray:
        return np.sort(np.concatenate([np.flatnonzero(labels == c)[: n // 2] for c in (0, 1)]))
    tr, va = pick(split.train_labels), pick(split.val_labels)
    return SplitData(split.train_ids[tr], [split.train_texts[i] for i in tr], split.train_labels[tr],
                     split.val_ids[va], [split.val_texts[i] for i in va], split.val_labels[va], split.seed)


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def train(cfg: DistilBertConfig, limit: int | None = None, device: str | None = None,
          save_checkpoint: bool = True, notes: str = "", extra_train: Path | None = None,
          train_size: int | None = None) -> dict:
    split = load_split()
    if train_size is not None:
        split = subsample_train(split, train_size)
    smoke = limit is not None
    if smoke:
        split = balanced_subset(split, limit)
    extra_texts, extra_labels, extra_tag = load_extra_train(extra_train)
    extra_texts, extra_labels = extra_texts[:limit], extra_labels[:limit]
    size_tag = f"_n{train_size}" if train_size is not None else ""
    run_name = run_name_for(cfg) + size_tag + extra_tag + (f"_smoke{limit}" if smoke else "")
    results_dir = RESULTS_DIR / "smoke" if smoke else RESULTS_DIR
    ckpt_dir = (CHECKPOINT_ROOT / ("smoke" if smoke else model_family(cfg).split("-")[0]) / run_name) if save_checkpoint else None

    clf = DistilBertClassifier(cfg, device=device)
    print(f"[{run_name}] device={clf.device} train={len(split.train_texts)} extra={len(extra_texts)} "
          f"val={len(split.val_texts)}", flush=True)

    # Validation is used only to pick the best epoch (checkpoint selection), never for gradient updates.
    with Timer() as t:
        clf.fit(split.train_texts + extra_texts, np.concatenate([split.train_labels, extra_labels]),
                split.val_texts, split.val_labels,
                checkpoint_dir=ckpt_dir, log=lambda s: print(s, flush=True))
    print(f"[{run_name}] training finished in {t.seconds:.0f}s; scoring train/val with best epoch {clf.best_epoch}",
          flush=True)
    # Release the allocator cache held from training before inference (see experiments/distilbert_full.py).
    if clf.device.type == "cuda":
        torch.cuda.empty_cache()

    train_probs = clf.predict_proba(split.train_texts)
    val_probs = clf.predict_proba(split.val_texts)
    hyper = {**clf.get_config(), "device": str(clf.device),
             "checkpoint": str(ckpt_dir.relative_to(TEAM_ROOT)) if ckpt_dir else None}
    result = record_experiment(
        model_family(cfg), split, train_probs, val_probs, t.seconds, seed=cfg.seed, hyperparameters=hyper,
        notes=notes, history=clf.history, run_name=run_name,
        metrics_dir=results_dir / "metrics", predictions_dir=results_dir / "predictions",
    )
    row = {"run_name": run_name, "learning_rate": cfg.learning_rate, "epochs": cfg.epochs,
           "batch_size": cfg.batch_size, "max_length": cfg.max_length, "seed": cfg.seed,
           "best_epoch": clf.best_epoch, "device": str(clf.device),
           "train_time_seconds": result.train_time_seconds, "train_auc": round(result.train_auc, 6),
           "val_auc": round(result.val_auc, 6), "val_accuracy": round(result.val_metrics["accuracy"], 6),
           "notes": notes}
    append_csv(results_dir / EXPERIMENTS_CSV.name, row)
    print(f"[{run_name}] train_auc={result.train_auc:.4f} val_auc={result.val_auc:.4f} "
          f"time={result.train_time_seconds:.0f}s", flush=True)
    return row


def parse_args(argv=None) -> argparse.Namespace:
    d = DistilBertConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lr", type=float, default=d.learning_rate)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--max-length", type=int, default=d.max_length)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--model-name", default=d.model_name)
    p.add_argument("--newline-marker", default=d.newline_marker, help='e.g. "[NL]": encode line breaks as a token')
    p.add_argument("--space-marker", default=d.space_marker, help='e.g. "[DS]": encode runs of 2+ spaces as a token')
    p.add_argument("--warmup-ratio", type=float, default=d.warmup_ratio, help="share of steps with linear LR warmup")
    p.add_argument("--bf16", action="store_true", help="bfloat16 autocast for forward passes; weights stay fp32")
    p.add_argument("--grad-checkpointing", action="store_true", help="trade compute for activation memory")
    p.add_argument("--layout-aug", type=float, default=0.0,
                   help="share of training passages given a random line layout each epoch (src.layout)")
    p.add_argument("--device", default=None, help="force cuda / mps / cpu (default: auto)")
    p.add_argument("--limit", type=int, default=None, help="smoke test: use N balanced train and N val examples")
    p.add_argument("--no-checkpoint", action="store_true")
    p.add_argument("--extra-train", type=Path, default=None,
                   help="id,text,label CSV appended to the training split (e.g. results/pseudo/<name>.csv)")
    p.add_argument("--train-size", type=int, default=None,
                   help="learning curve: train on a class-balanced subset of N training-split rows")
    p.add_argument("--notes", default="")
    return p.parse_args(argv)


def main(argv=None) -> None:
    a = parse_args(argv)
    if a.max_length > 512:
        sys.exit("DistilBERT supports at most 512 tokens")
    cfg = DistilBertConfig(model_name=a.model_name, max_length=a.max_length, learning_rate=a.lr,
                           batch_size=a.batch_size, epochs=a.epochs, seed=a.seed,
                           newline_marker=a.newline_marker, space_marker=a.space_marker,
                           warmup_ratio=a.warmup_ratio, bf16=a.bf16,
                           gradient_checkpointing=a.grad_checkpointing, layout_aug=a.layout_aug)
    train(cfg, limit=a.limit, device=a.device, save_checkpoint=not a.no_checkpoint, notes=a.notes,
          extra_train=a.extra_train, train_size=a.train_size)


if __name__ == "__main__":
    main()
