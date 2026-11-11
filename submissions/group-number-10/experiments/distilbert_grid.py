"""Small, staged hyperparameter comparison for DistilBERT (validation split fixed).

Stage 1: learning rate in {2e-5, 3e-5, 5e-5} at 3 epochs (2e-5/3 is the baseline).
Stage 2: epochs in {2, 4} at the stage-1 best learning rate (by validation AUROC).
=> 5 runs instead of the full 3x3 = 9. Runs whose metrics JSON already exists are skipped,
so the script can be resumed after an interruption.

    python -m experiments.distilbert_grid
"""

from __future__ import annotations

import argparse
import json

from experiments.distilbert_train import run_name_for, train
from src.classifiers.distilbert import DistilBertConfig
from src.data import RESULTS_DIR

LEARNING_RATES = (2e-5, 3e-5, 5e-5)
STAGE1_EPOCHS = 3
STAGE2_EPOCHS = (2, 4)


def run_or_load(cfg: DistilBertConfig, device: str | None) -> float:
    path = RESULTS_DIR / "metrics" / f"{run_name_for(cfg)}.json"
    if path.exists():
        val_auc = json.loads(path.read_text())["val_auc"]
        print(f"skip {path.stem} (exists, val_auc={val_auc:.4f})", flush=True)
        return val_auc
    return train(cfg, device=device, notes="grid")["val_auc"]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)

    stage1 = {lr: run_or_load(DistilBertConfig(learning_rate=lr, epochs=STAGE1_EPOCHS, seed=a.seed), a.device)
              for lr in LEARNING_RATES}
    best_lr = max(stage1, key=stage1.get)
    print(f"stage 1 val AUROC: {stage1} -> best lr {best_lr:g}", flush=True)
    for ep in STAGE2_EPOCHS:
        run_or_load(DistilBertConfig(learning_rate=best_lr, epochs=ep, seed=a.seed), a.device)


if __name__ == "__main__":
    main()
