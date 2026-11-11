"""Final models: fine-tune on all 10,000 labeled passages and predict test.csv, one run per seed.

Hyperparameters come from the validation-split experiments. With no held-out data the last
epoch is kept. Test probabilities of all seeds are averaged into one submission file.
Resumable: seeds whose test predictions already exist are loaded instead of retrained.

    python -m experiments.distilbert_full --lr 3e-5 --epochs 4 --newline-marker "[NL]" --seeds 42 43 44

Outputs:
    checkpoints/full/<run>/                              model per seed (git-ignored)
    results/full/<run>.json                              config, train time, train AUROC, history
    results/submissions/<run>.csv                        test predictions per seed (id,label)
    results/submissions/<config>_full_seeds<a-b-c>.csv   seed average (id,label)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from experiments.distilbert_train import CHECKPOINT_ROOT, load_extra_train, run_name_for
from predict import check_against_sample
from src.classifiers.distilbert import DistilBertClassifier, DistilBertConfig
from src.data import DATA_DIR, RESULTS_DIR, load_labeled, load_unlabeled
from src.evaluate import Timer


def main(argv=None) -> None:
    d = DistilBertConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lr", type=float, default=d.learning_rate)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--max-length", type=int, default=d.max_length)
    p.add_argument("--model-name", default=d.model_name)
    p.add_argument("--newline-marker", default=None)
    p.add_argument("--space-marker", default=None)
    p.add_argument("--warmup-ratio", type=float, default=d.warmup_ratio)
    p.add_argument("--bf16", action="store_true", help="bfloat16 autocast for forward passes; weights stay fp32")
    p.add_argument("--grad-checkpointing", action="store_true", help="trade compute for activation memory")
    p.add_argument("--layout-aug", type=float, default=0.0,
                   help="share of training passages given a random line layout each epoch (src.layout)")
    p.add_argument("--extra-train", type=Path, default=None,
                   help="id,text,label CSV appended to the labeled data (e.g. results/pseudo/<name>.csv)")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=None, help="smoke test: first N training passages")
    a = p.parse_args(argv)

    data = load_labeled(DATA_DIR / "train.csv")
    extra_texts, extra_labels, extra_tag = load_extra_train(a.extra_train)
    train_texts = data.texts[: a.limit] + extra_texts[: a.limit]
    train_labels = np.concatenate([data.labels[: a.limit], extra_labels[: a.limit]])
    test_ids, test_texts = load_unlabeled(DATA_DIR / "test.csv")
    sub_dir, meta_dir = RESULTS_DIR / "submissions", RESULTS_DIR / "full"
    sub_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    all_probs = []
    for seed in a.seeds:
        cfg = DistilBertConfig(model_name=a.model_name, max_length=a.max_length, learning_rate=a.lr,
                               batch_size=a.batch_size, epochs=a.epochs, seed=seed,
                               newline_marker=a.newline_marker, space_marker=a.space_marker,
                               warmup_ratio=a.warmup_ratio, bf16=a.bf16,
                               gradient_checkpointing=a.grad_checkpointing, layout_aug=a.layout_aug)
        run = run_name_for(cfg) + extra_tag + "_full"
        out_csv = sub_dir / f"{run}.csv"
        if out_csv.exists():
            print(f"skip {run} (test predictions exist)", flush=True)
            all_probs.append(pd.read_csv(out_csv)["label"].to_numpy())
            continue

        clf = DistilBertClassifier(cfg, device=a.device)
        print(f"[{run}] device={clf.device} train={len(train_texts)} (all labeled data + {len(extra_texts[: a.limit])} extra)",
              flush=True)
        with Timer() as t:
            clf.fit(train_texts, train_labels, checkpoint_dir=CHECKPOINT_ROOT / "full" / run,
                    log=lambda s: print(s, flush=True))
        # Release the allocator cache held from training before inference. On the DGX Spark's unified
        # memory DeBERTa-large otherwise keeps ~105 GB reserved and the eval batches run out of memory.
        if clf.device.type == "cuda":
            torch.cuda.empty_cache()
        n_labeled = len(train_texts) - len(extra_texts[: a.limit])  # train AUROC on real labels only
        train_auc = float(roc_auc_score(train_labels[:n_labeled], clf.predict_proba(train_texts[:n_labeled])))
        probs = clf.predict_proba(test_texts)

        sub = pd.DataFrame({"id": test_ids, "label": probs})
        check_against_sample(sub, DATA_DIR / "sample_submission.csv")
        sub.to_csv(out_csv, index=False)
        meta = {"run": run, "config": asdict(cfg), "n_train": len(train_texts), "train_auc": train_auc,
                "train_time_seconds": round(t.seconds, 1), "history": clf.history, "test_predictions": out_csv.name}
        (meta_dir / f"{run}.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"[{run}] train_auc={train_auc:.4f} time={t.seconds:.0f}s -> {out_csv.name}", flush=True)
        all_probs.append(probs)

    ens_name = run_name_for(cfg).rsplit("_seed", 1)[0] + extra_tag + "_full_seeds" + "-".join(map(str, a.seeds))
    ens = pd.DataFrame({"id": test_ids, "label": np.mean(all_probs, axis=0)})
    check_against_sample(ens, DATA_DIR / "sample_submission.csv")
    ens.to_csv(sub_dir / f"{ens_name}.csv", index=False)
    if len(all_probs) > 1:
        corr = pd.DataFrame(np.array(all_probs).T).corr(method="spearman").to_numpy()
        print(f"seed-to-seed Spearman (min) {corr[np.triu_indices(len(all_probs), 1)].min():.3f}", flush=True)
    print(f"wrote {ens_name}.csv", flush=True)


if __name__ == "__main__":
    main()
