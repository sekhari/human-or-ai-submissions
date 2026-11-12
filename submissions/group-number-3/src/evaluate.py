"""Evaluate a model on both shared validation splits and log the result.

    python src/evaluate.py --model tfidf_logreg
"""
import argparse
import json
import time
from datetime import datetime

from sklearn.metrics import roc_auc_score

from data import ROOT, get_split, load_splits, load_train
from models import MODELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tfidf_logreg", choices=MODELS)
    args = parser.parse_args()

    train, splits = load_train(), load_splits()
    result = {"model": args.model, "timestamp": datetime.now().isoformat(timespec="seconds")}

    for fold in ["random_fold", "ood_fold"]:
        tr, va = get_split(train, splits, fold)
        start = time.time()
        model = MODELS[args.model]()
        model.fit(tr["text"], tr["label"])
        auc = roc_auc_score(va["label"], model.predict_proba(va["text"])[:, 1])
        result[f"{fold}_auroc"] = round(auc, 4)
        print(f"{fold}: AUROC = {auc:.4f}  ({time.time() - start:.0f}s)")

    log = ROOT / "results" / "val_results.jsonl"
    log.parent.mkdir(exist_ok=True)
    with open(log, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"Logged to {log.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
