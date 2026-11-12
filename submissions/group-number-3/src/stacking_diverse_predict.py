"""Train the diverse stack (TF-IDF LR + TF-IDF SVM + perplexity GBM, cv=5) on all
10,000 labeled rows and write Kaggle predictions for test.csv.

Needs results/ppl2_features.csv (run `python src/perplexity2.py` once to create it).
Writes sub_stack_diverse.csv (id,label) in the project folder; it does not touch
submission.csv or any other submission file.

    python src/stacking_diverse_predict.py
"""
import pandas as pd
from scipy import sparse

from data import ROOT, load_test, load_train
from stacking_diverse import CACHE, stack, tfidf

if __name__ == "__main__":
    ppl = pd.read_csv(CACHE)
    pcols = [c for c in ppl.columns if c not in ("id", "split")]
    ptr = ppl[ppl.split == "train"].set_index("id")[pcols]
    pte = ppl[ppl.split == "test"].set_index("id")[pcols]

    train, test = load_train(), load_test()
    f = tfidf()
    Ttr, Tte = f.fit_transform(train["text"]), f.transform(test["text"])
    Xtr = sparse.hstack([Ttr, sparse.csr_matrix(ptr.loc[train["id"]].values)]).tocsr()
    Xte = sparse.hstack([Tte, sparse.csr_matrix(pte.loc[test["id"]].values)]).tocsr()

    model = stack(Ttr.shape[1], cv=5).fit(Xtr, train["label"].values)
    sub = pd.DataFrame({"id": test["id"], "label": model.predict_proba(Xte)[:, 1]})
    assert len(sub) == 3000 and sub["id"].is_unique and sub["label"].between(0, 1).all()
    sub.to_csv(ROOT / "sub_stack_diverse.csv", index=False)
    print(f"Wrote sub_stack_diverse.csv ({len(sub)} rows)")
