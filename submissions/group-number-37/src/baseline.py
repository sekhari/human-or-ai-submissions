"""the first notebook's lexical and catboost style recipe, kept as the reference to beat."""
import math
import re
import zlib
from collections import Counter

import numpy as np
from catboost import CatBoostClassifier
from scipy.sparse import hstack
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

SEED = 5785
WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
FUNCTION_WORDS = sorted(ENGLISH_STOP_WORDS)
PUNCT = ".,!?;:-—–'’“”()[]/#*%&" + chr(34) + "\n\t"


def _dist(v):
    return [np.mean(v), np.std(v), *np.quantile(v, [.1, .5, .9])] if v else [0.] * 5


def style_features(text):
    words = WORD.findall(text)
    lower = [w.lower() for w in words]
    counts = Counter(lower)
    chars = Counter(text)
    nc, nw = max(1, len(text)), max(1, len(words))
    sents = [p for p in re.split(r'(?<=[.!?])\s+|\n+', text) if p.strip()]
    bg = Counter(zip(lower, lower[1:]))
    b = text.encode()
    return [
        math.log1p(len(text)), math.log1p(len(words)), len(sents),
        len(counts) / nw, len(counts) / math.sqrt(nw), sum(v == 1 for v in counts.values()) / nw,
        max(counts.values(), default=0) / nw, len(bg) / nw, sum(v > 1 for v in bg.values()) / nw,
        len(zlib.compress(b)) / max(1, len(b)), len(zlib.compress(" ".join(lower).encode())) / nw,
        sum(c.isdigit() for c in text) / nc, sum(ord(c) > 127 for c in text) / nc,
        sum(c.isupper() for c in text) / nc, sum(w.isupper() for w in words) / nw, sum(w.istitle() for w in words) / nw,
        *_dist([len(w) for w in words]), *_dist([len(WORD.findall(s)) for s in sents]),
        *_dist([len(s) for s in text.splitlines() if s.strip()]),
        *[chars[c] / nc for c in PUNCT], *[counts[w] / nw for w in FUNCTION_WORDS],
    ]


def style_matrix(texts):
    return np.asarray([style_features(t) for t in texts], dtype=np.float32)


def fit_lexical(tr, y):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=80000, sublinear_tf=True,
                           strip_accents="unicode", dtype=np.float32)
    char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=3, max_features=100000,
                           sublinear_tf=True, dtype=np.float32)
    x = hstack((word.fit_transform(tr), char.fit_transform(tr)), format="csr")
    clf = LogisticRegression(C=8, solver="liblinear", max_iter=500, random_state=SEED).fit(x, y)
    return lambda t: clf.decision_function(hstack((word.transform(t), char.transform(t)), format="csr"))


def fit_style(tr, y, iterations=850, depth=6, threads=8):
    clf = CatBoostClassifier(iterations=iterations, depth=depth, learning_rate=.06, verbose=False, thread_count=threads,
                             random_seed=SEED, allow_writing_files=False)
    clf.fit(style_matrix(tr), y)
    return lambda t: clf.predict(style_matrix(t), prediction_type="RawFormulaVal")
