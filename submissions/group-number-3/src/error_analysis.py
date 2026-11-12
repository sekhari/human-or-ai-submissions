"""Error analysis for the word+char TF-IDF LogReg and the style GBM.

1. AUROC by passage length (word-count quartiles of the validation set)
2. Top word / char n-gram features of the LogReg for each class
3. Most confident mistakes (false positives and false negatives) -> text file
4. Near-duplicate leakage: validation passages whose closest training passage has
   TF-IDF cosine similarity >= 0.8, and AUROC with those removed

    python src/error_analysis.py
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

from data import ROOT, get_split, load_splits, load_train
from models import style_gbm, tfidf_logreg

DUP_THRESHOLD = 0.8


def safe_auc(y, p):
    return roc_auc_score(y, p) if len(set(y)) == 2 else float("nan")


def main():
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    train, splits = load_train(), load_splits()
    tr, va = get_split(train, splits, "random_fold")
    va = va.reset_index(drop=True)
    y = va["label"].values

    lr = tfidf_logreg().fit(tr["text"], tr["label"])
    gbm = style_gbm().fit(tr["text"], tr["label"])
    va["p_lr"] = lr.predict_proba(va["text"])[:, 1]
    va["p_gbm"] = gbm.predict_proba(va["text"])[:, 1]
    va["n_words"] = va["text"].str.split().str.len()
    va["has_blank_line"] = va["text"].str.contains("\n\n")

    # 1. By length
    va["length_bucket"] = pd.qcut(va["n_words"], 4)
    by_len = va.groupby("length_bucket", observed=True).apply(
        lambda g: pd.Series({"n": len(g), "frac_ai": g["label"].mean(),
                             "auroc_lr": safe_auc(g["label"], g["p_lr"]),
                             "auroc_gbm": safe_auc(g["label"], g["p_gbm"])}),
        include_groups=False).round(4)
    print("1. AUROC by passage length (word-count quartiles):\n")
    print(by_len.to_string(), "\n")

    # AUROC split by blank-line presence (the formatting artifact)
    print("   AUROC on passages WITHOUT blank lines (artifact unavailable):")
    sub = va[~va["has_blank_line"]]
    print(f"   n={len(sub)}  frac_ai={sub['label'].mean():.3f}  "
          f"LogReg={safe_auc(sub['label'], sub['p_lr']):.4f}  "
          f"GBM={safe_auc(sub['label'], sub['p_gbm']):.4f}\n")

    # 2. Top features
    feats = lr.named_steps["features"]
    names = np.concatenate([
        ["w:" + n for n in feats.transformer_list[0][1].get_feature_names_out()],
        ["c:" + repr(n) for n in feats.transformer_list[1][1].get_feature_names_out()],
    ])
    coef = lr.named_steps["clf"].coef_[0]
    order = np.argsort(coef)
    top_ai = [(names[i], round(coef[i], 2)) for i in order[::-1][:25]]
    top_h = [(names[i], round(coef[i], 2)) for i in order[:25]]
    lines = ["Top features -> AI:"] + [f"  {n:<30} {c}" for n, c in top_ai]
    lines += ["", "Top features -> human:"] + [f"  {n:<30} {c}" for n, c in top_h]
    (ROOT / "results" / "top_features.txt").write_text("\n".join(lines))
    print("2. Top LogReg features:\n")
    print("\n".join(lines[:16]), "\n   ...")
    print("\n".join(lines[26:42]), "\n   ... (full list in results/top_features.txt)\n")

    # 3. Most confident mistakes
    out = []
    for model_col, name in [("p_lr", "TF-IDF LogReg"), ("p_gbm", "Style GBM")]:
        fp = va[va["label"] == 0].nlargest(5, model_col)
        fn = va[va["label"] == 1].nsmallest(5, model_col)
        for kind, df in [("FALSE POSITIVES (human predicted AI)", fp),
                         ("FALSE NEGATIVES (AI predicted human)", fn)]:
            out.append(f"===== {name}: {kind} =====")
            for _, r in df.iterrows():
                snippet = r["text"][:400].replace("\n", "\\n")
                out.append(f"[id={r['id']}  p_AI={r[model_col]:.3f}  words={r['n_words']}  "
                           f"blank_line={r['has_blank_line']}]\n{snippet}...\n")
    (ROOT / "results" / "misclassified.txt").write_text("\n".join(out))
    n_err_lr = ((va["p_lr"] >= 0.5) != (y == 1)).sum()
    n_err_gbm = ((va["p_gbm"] >= 0.5) != (y == 1)).sum()
    print(f"3. Errors at threshold 0.5: LogReg {n_err_lr}/2000, GBM {n_err_gbm}/2000. "
          f"Most confident mistakes saved to results/misclassified.txt\n")

    # 4. Near-duplicate leakage
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(tr["text"])
    sim = (vec.transform(va["text"]) @ vec.transform(tr["text"]).T).max(axis=1).toarray().ravel()
    va["max_sim_to_train"] = sim
    dup = sim >= DUP_THRESHOLD
    print(f"4. Near-duplicates (cosine >= {DUP_THRESHOLD}): {dup.sum()} of 2000 validation passages")
    print(f"   Max similarity: median={np.median(sim):.3f}, 99th pct={np.percentile(sim, 99):.3f}, max={sim.max():.3f}")
    if dup.sum():
        print(f"   LogReg AUROC without them: {safe_auc(y[~dup], va.loc[~dup, 'p_lr']):.4f} "
              f"(all: {safe_auc(y, va['p_lr']):.4f})")

    va.drop(columns=["text", "length_bucket"]).to_csv(ROOT / "results" / "val_predictions.csv", index=False)
    by_len.to_csv(ROOT / "results" / "auroc_by_length.csv")
    print("\nSaved results/auroc_by_length.csv, top_features.txt, misclassified.txt, val_predictions.csv")


if __name__ == "__main__":
    main()
