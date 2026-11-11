"""look for attack-like edits and infer rough domains across all three files."""
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.data import load_all, SEED

train, unl, test = load_all()
pd.set_option("display.width", 220, "display.max_columns", 40)


def script(c):
    if not c.isalpha():
        return None
    n = unicodedata.name(c, "")
    return n.split(" ")[0]


def sig(t):
    w = re.findall(r"\S+", t)
    n = max(1, len(w))
    low = [x.lower().strip('.,;:!?"()') for x in w]
    mixed = sum(1 for x in w if len({script(c) for c in x if script(c)}) > 1)
    return dict(
        zero_width=int(bool(re.search("[​-‏⁠﻿]", t))),
        mixed_script=int(mixed > 0),
        any_cyr_greek=int(bool(re.search("[Ͱ-ϿЀ-ӿ]", t))),
        devanagari=int(bool(re.search("[ऀ-ॿ]", t))),
        no_articles=int(sum(x in ("the", "a", "an") for x in low) / n < 0.015),
        odd_caps=int(len(re.findall(r"\b[a-z]+[A-Z][a-z]*\b", t)) >= 3),
        multi_space=int(len(re.findall(r"[^\s]  +[^\s]", t)) >= 3),
        no_digits=int(not re.search(r"\d", t)),
        british=int(bool(re.search(r"\b\w+(isation|ise|our|ised|ising)\b", t))),
        any_dnl=int("\n\n" in t),
    )


rows = {k: pd.DataFrame([sig(t) for t in f.text]).mean() for k, f in
        dict(human=train[train.label == 0], ai=train[train.label == 1], unl=unl, test=test).items()}
print(pd.DataFrame(rows).round(4))

allt = pd.concat([train.text, unl.text, test.text], ignore_index=True)
src = np.r_[np.where(train.label == 0, "human", "ai"), ["unl"] * len(unl), ["test"] * len(test)]
v = TfidfVectorizer(stop_words="english", min_df=5, max_df=0.4, sublinear_tf=True, max_features=40000)
x = v.fit_transform(allt)
z = normalize(TruncatedSVD(60, random_state=SEED).fit_transform(x))
k = KMeans(10, n_init=5, random_state=SEED).fit_predict(z)
names = np.array(v.get_feature_names_out())
print(pd.crosstab(k, src))
for c in range(10):
    m = np.asarray(x[k == c].mean(0)).ravel()
    print(c, names[np.argsort(m)[-12:]].tolist())
