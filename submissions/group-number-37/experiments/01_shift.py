"""how different are train, unlabeled and test? adversarial validation plus format checks."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score

from src.data import load_all, SEED

train, unl, test = load_all()


def adv(a, b, name, analyzer="word", ng=(1, 2)):
    t = pd.concat([a, b]).to_numpy()
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    v = TfidfVectorizer(analyzer=analyzer, ngram_range=ng, min_df=3, sublinear_tf=True, max_features=200000)
    x = v.fit_transform(t)
    m = LogisticRegression(C=5, max_iter=400, class_weight="balanced")
    p = cross_val_predict(m, x, y, cv=StratifiedKFold(4, shuffle=True, random_state=SEED), method="decision_function")
    print(f"{name:28s} {analyzer:5s} auc={roc_auc_score(y, p):.4f}", flush=True)
    return p, v, m.fit(x, y)


for ana, ng in (("word", (1, 2)), ("char", (2, 4))):
    adv(train.text, test.text, "train vs test", ana, ng)
    adv(train.text, unl.text, "train vs unlabeled", ana, ng)
    adv(unl.text, test.text, "unlabeled vs test", ana, ng)
adv(train.text[train.label == 0], test.text, "human vs test")
adv(train.text[train.label == 1], test.text, "ai vs test")

_, v, m = adv(train.text, test.text, "train vs test (refit)")
names = np.array(v.get_feature_names_out())
o = np.argsort(m.coef_[0])
print("more train:", names[o[:40]].tolist())
print("more test :", names[o[-60:]].tolist())

dnl = {k: f.text.str.contains("\n\n") for k, f in dict(train=train, unl=unl, test=test).items()}
print("double newline rate", {k: round(v.mean(), 4) for k, v in dnl.items()})
print(pd.crosstab(train.label, dnl["train"]))
for k, f in dict(train=train, unl=unl, test=test).items():
    print(k, "trailing punct missing", round((~f.text.str.rstrip().str[-1:].isin(list('.!?"”'))).mean(), 3),
          "ends ws", round(f.text.str[-1:].str.isspace().mean(), 3),
          "nbsp", round(f.text.str.contains("\xa0").mean(), 3),
          "line_len_med", round(f.text.map(lambda t: np.median([len(l) for l in t.split("\n")])).mean(), 1),
          "cr", round(f.text.str.contains("\r").mean(), 3))
print("human trailing", (~train[train.label == 0].text.str.rstrip().str[-1:].isin(list('.!?"”'))).mean(),
      "ai trailing", (~train[train.label == 1].text.str.rstrip().str[-1:].isin(list('.!?"”'))).mean())
