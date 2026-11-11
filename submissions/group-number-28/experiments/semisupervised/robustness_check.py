import argparse
import json
import re
import time
import unicodedata

import numpy as np
from sklearn.metrics import roc_auc_score

from experiments.semisupervised.run_q3 import (RESULTS, SEED, classifier,
                                               environment, git_state, vectorizer)
from human_ai.data import load_partitions, raw_path, read_csv

C = 30.0
ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿"))


def normalize(text):
    text = unicodedata.normalize("NFKC", text).translate(ZERO_WIDTH)
    return re.sub(r"\s+", " ", text).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    args = ap.parse_args()
    out = RESULTS / f"{args.prefix}_study"
    out.mkdir(parents=True, exist_ok=False)

    train, val = load_partitions()
    tr = [r["text"] for r in train]
    y = np.array([int(r["label"]) for r in train])
    va = [r["text"] for r in val]
    vy = np.array([int(r["label"]) for r in val])
    others = {"unlabeled": [r["text"] for r in read_csv(raw_path("unlabeled.csv"))],
              "test": [r["text"] for r in read_csv(raw_path("test.csv"))]}

    res = {"C": C, "seed": SEED, "git": git_state(), "environment": environment(),
           "normalization": "NFKC + remove zero-width chars + collapse whitespace"}
    # Single-feature shortcut: score = number of blank lines.
    res["blank_line_count_only_val_auroc"] = roc_auc_score(vy, [t.count("\n\n") for t in va])
    res["blank_line_share"] = {
        "train_human": float(np.mean(["\n\n" in t for t, l in zip(tr, y) if l == 0])),
        "train_ai": float(np.mean(["\n\n" in t for t, l in zip(tr, y) if l == 1])),
        **{k: float(np.mean(["\n\n" in t for t in v])) for k, v in others.items()}}

    for name, prep in (("raw", lambda t: t), ("normalized", normalize)):
        t0 = time.perf_counter()
        vec = vectorizer()
        clf = classifier(C).fit(vec.fit_transform([prep(t) for t in tr]), y)
        secs = time.perf_counter() - t0
        p_val_same = clf.predict_proba(vec.transform([prep(t) for t in va]))[:, 1]
        p_val_norm = clf.predict_proba(vec.transform([normalize(t) for t in va]))[:, 1]
        conf = {}
        for k, texts in {"validation": va, **others}.items():
            p = clf.predict_proba(vec.transform([prep(t) for t in texts]))[:, 1]
            c = np.maximum(p, 1 - p)
            conf[k] = {"share_conf_ge_0.95": float((c >= 0.95).mean()),
                       "share_conf_ge_0.80": float((c >= 0.80).mean()),
                       "mean_p_ai": float(p.mean()),
                       "share_pred_ai": float((p >= 0.5).mean())}
        res[name] = {"fit_seconds": secs,
                     "val_auroc_same_preprocessing": roc_auc_score(vy, p_val_same),
                     "val_auroc_on_normalized_validation": roc_auc_score(vy, p_val_norm),
                     "confidence_by_split": conf}
        print(name, json.dumps(res[name], indent=1), flush=True)

    (out / "metrics.json").write_text(json.dumps(res, indent=2) + "\n")
    print("blank-line-only AUROC", res["blank_line_count_only_val_auroc"])
    print(res["blank_line_share"])


if __name__ == "__main__":
    main()
