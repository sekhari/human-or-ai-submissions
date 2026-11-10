"""Component models shared by the masking experiments."""
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.feature_extraction.text import TfidfVectorizer as V
from sklearn.linear_model import LogisticRegression as LR
from common import SEED
from distort import distort, top_words
from features import NAMES, normalize, stylometric_matrix

B = dict(sublinear_tf=True, strip_accents="unicode", lowercase=True)
FMT = [NAMES.index(n) for n in ("newline", "bullet")]; NOFMT = [i for i in range(len(NAMES)) if i not in FMT]

def fit_word(t, yy):
    n = [normalize(x) for x in t]; v = V(analyzer="word", ngram_range=(1, 2), min_df=2, max_df=.95, **B); m = LR(C=30, max_iter=3000).fit(v.fit_transform(n), yy)
    return lambda x: m.predict_proba(v.transform([normalize(s) for s in x]))[:, 1]
def fit_char(t, yy):
    n = [normalize(x) for x in t]; v = V(analyzer="char_wb", ngram_range=(2, 6), min_df=3, max_features=400000, **B); m = LR(C=10, max_iter=3000).fit(v.fit_transform(n), yy)
    return lambda x: m.predict_proba(v.transform([normalize(s) for s in x]))[:, 1]
def fit_gb(t, yy, cols):
    m = HGB(learning_rate=.1, max_leaf_nodes=31, max_iter=300, random_state=SEED).fit(stylometric_matrix(t)[:, cols], yy)
    return lambda x: m.predict_proba(stylometric_matrix(x)[:, cols])[:, 1]
def fit_dist(t, yy, k, c):
    keep = top_words(t, k); d = [distort(s, keep) for s in t]
    v = V(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 3), min_df=3, sublinear_tf=True, lowercase=False, max_features=300000)
    m = LR(C=c, max_iter=3000).fit(v.fit_transform(d), yy)
    return lambda x: m.predict_proba(v.transform([distort(s, keep) for s in x]))[:, 1]
