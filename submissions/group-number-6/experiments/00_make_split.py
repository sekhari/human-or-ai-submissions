"""Create the fixed validation split and check for near-duplicate leakage.

Near-duplicates are measured with word 1-2 gram TF-IDF cosine similarity. Pairs above
0.8 between the training partition and validation (or test) would make validation scores
optimistic, so we report them.

Usage: python experiments/00_make_split.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import ROOT, SPLIT_FILE, get_split, load_csv  # noqa: E402


def max_cross_sim(vec, a_texts, b_texts):
    A = vec.transform(a_texts)
    B = vec.transform(b_texts)
    best = np.zeros(B.shape[0])
    for start in range(0, B.shape[0], 500):
        s = (B[start:start + 500] @ A.T).toarray()
        best[start:start + 500] = s.max(axis=1)
    return best


def main():
    tr, va = get_split()
    te = load_csv("test.csv")
    un = load_csv("unlabeled.csv")
    print(f"split saved to {SPLIT_FILE}: train={len(tr)} val={len(va)}")
    print("val label balance:", va.label.value_counts().to_dict())

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    vec.fit(np.concatenate([tr.text, va.text, te.text, un.text]))
    out = {}
    for name, texts in [("val", va.text), ("test", te.text), ("unlabeled", un.text)]:
        s = max_cross_sim(vec, tr.text, texts)
        out[name] = {
            "n": int(len(s)),
            "max_sim_to_train_p50": float(np.median(s)),
            "max_sim_to_train_p99": float(np.quantile(s, 0.99)),
            "n_above_0.8": int((s > 0.8).sum()),
            "n_above_0.5": int((s > 0.5).sum()),
        }
    print(json.dumps(out, indent=2))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "split_leakage_check.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
