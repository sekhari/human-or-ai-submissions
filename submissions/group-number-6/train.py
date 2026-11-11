"""Train the final Human-or-AI detector.

Two modes:
  --validate   fit on the 8,000-row training partition and report AUROC on the fixed
               2,000-row validation split (the protocol used for every comparison)
  (default)    fit on ALL labeled rows of train.csv and save the model for predict.py

Examples
  python train.py --validate
  python train.py --train data/train.csv --model-out models/final.joblib
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from src.data import ROOT, get_split, load_csv
from src.ensemble import LogitAverage
from src.models import build

DEFAULT_CONFIG = ROOT / "configs" / "final.json"


def build_from_config(cfg: dict):
    members = [(m["name"], build(m["name"], normalize=cfg.get("normalize", False),
                                 **{k: (tuple(v) if isinstance(v, list) else v) for k, v in m["params"].items()}))
               for m in cfg["members"]]
    if len(members) == 1:
        return members[0][1]
    return LogitAverage(members)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Human-or-AI detector")
    ap.add_argument("--train", type=Path, default=ROOT / "data" / "train.csv")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--model-out", type=Path, default=ROOT / "models" / "final.joblib")
    ap.add_argument("--validate", action="store_true", help="fit on 8k partition, score fixed validation split")
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = json.loads(args.config.read_text())
    model = build_from_config(cfg)
    t = time.time()
    if args.validate:
        tr, va = get_split(args.train.parent)
        model.fit(list(tr.text), tr.label.to_numpy())
        p_tr = model.predict_proba(list(tr.text))[:, 1]
        p_va = model.predict_proba(list(va.text))[:, 1]
        print(f"config={args.config.name} fit={time.time()-t:.0f}s "
              f"train AUROC={roc_auc_score(tr.label, p_tr):.4f} val AUROC={roc_auc_score(va.label, p_va):.4f}")
        return
    df = load_csv(args.train.name, args.train.parent)
    model.fit(list(df.text), df.label.to_numpy())
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)
    print(f"trained on {len(df)} rows in {time.time()-t:.0f}s -> {args.model_out}")


if __name__ == "__main__":
    main()
