"""Comparing three approaches and selecting their hyperparameters on the validation split

  - tfidf_lr:  word n-gram + character 3-5gram TF-IDF, logistic regression
  - masked_lr: masked function-word n-gram TF-IDF, logistic regression
  - style_gb:  24 hand-crafted style features, histogram gradient boosting

Run with:  python -m src.q1_model_selection
Writes to results/q1/: q12_models.csv, q13_<model>_grid.csv and summary.md
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .data import RESULTS_DIR, SPLIT_SEED, load_split
from .models import make_masked_lr, make_style_gb, make_tfidf_lr

CANDIDATES = {
    "tfidf_lr": ("Word n-gram + char 3-5gram TF-IDF", make_tfidf_lr, {
        "logisticregression__C": [0.25, 1, 4, 16, 64, 256],
        "featureunion__tfidfvectorizer-1__ngram_range": [(1, 1), (1, 2), (1, 3)],
    }),
    "masked_lr": ("Masked function-word n-gram TF-IDF", make_masked_lr, {
        "logisticregression__C": [0.25, 1, 4, 16, 64, 256],
        "tfidfvectorizer__ngram_range": [(1, 2), (1, 3), (1, 4), (1, 5)],
    }),
    "style_gb": ("24 style features", make_style_gb, {
        "histgradientboostingclassifier__learning_rate": [0.03, 0.1, 0.3, 0.6],
        "histgradientboostingclassifier__max_leaf_nodes": [15, 31, 63, 127],
    }),
}


def short(params):
    return ", ".join(f"{k.split('__')[-1]}={v}" for k, v in params.items())


def main():
    out = RESULTS_DIR / "q1"
    out.mkdir(parents=True, exist_ok=True)
    train, val = load_split()
    X_tr, y_tr = train["text"], train["label"].values
    X_val, y_val = val["text"], val["label"].values
    cv = StratifiedKFold(5, shuffle=True, random_state=SPLIT_SEED)

    models, grids = [], {}
    for name, (representation, make_model, grid) in CANDIDATES.items():
        start = time.time()
        search = GridSearchCV(make_model(), grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=False).fit(X_tr, y_tr)
        res = pd.DataFrame(search.cv_results_)
        table = pd.DataFrame({
            "setting": res["params"].map(short),
            "CV AUROC (mean)": res["mean_test_score"],
            "CV AUROC (std)": res["std_test_score"],
        }).sort_values("CV AUROC (mean)", ascending=False)
        table.to_csv(out / f"q13_{name}_grid.csv", index=False)
        grids[name] = table
        best = res["params"][int(np.argmax(res["mean_test_score"]))]
        print(f"{name}: searched {len(res)} settings in {time.time() - start:.0f}s, chose {short(best)}")

        start = time.time()
        model = make_model().set_params(**best).fit(X_tr, y_tr)
        fit_seconds = time.time() - start
        models.append({
            "model": name, "text representation": representation, "chosen setting": short(best),
            "CV AUROC": res["mean_test_score"].max(),
            "train AUROC": roc_auc_score(y_tr, model.predict_proba(X_tr)[:, 1]),
            "val AUROC": roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]),
            "fit seconds": fit_seconds,
        })

    models = pd.DataFrame(models)
    models.to_csv(out / "q12_models.csv", index=False)
    with open(out / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Q1 results (fixed split seed 42: 8,000 train / 2,000 validation)\n\n")
        f.write("Settings chosen by 5-fold CV inside the 8,000 training rows; validation scored once.\n\n")
        f.write("## Q1.2 models\n\n" + models.round(4).to_markdown(index=False) + "\n\n")
        for name, table in grids.items():
            f.write(f"## Q1.3 grid: {name}\n\n" + table.round(4).to_markdown(index=False) + "\n\n")
    print(open(out / "summary.md", encoding="utf-8").read())


if __name__ == "__main__":
    main()
