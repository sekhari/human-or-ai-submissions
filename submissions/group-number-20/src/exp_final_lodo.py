"""Evaluate the exact final pipelines on pseudo-domains and bootstrap their validation gains."""
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer as V
from sklearn.metrics import roc_auc_score as auc
from common import SEED, load_train, split
from pipeline import Detector
tr, va = split(load_train("data/train.csv")); y = tr.label.values
X = TruncatedSVD(50, random_state=SEED).fit_transform(V(sublinear_tf=True, min_df=3, stop_words="english").fit_transform(tr.text))
cl = KMeans(6, n_init=10, random_state=SEED).fit_predict(X); out = {}
for name, fmt in (("final", True), ("final_nofmt", False)):
    s = []
    for k in range(6):
        m = cl == k
        if y[m].min() == y[m].max(): continue
        p = Detector(fmt).fit(tr.text[~m].tolist(), y[~m]).predict(tr.text[m].tolist()); s.append(auc(y[m], p)); print(name, k, s[-1], flush=True)
    out[name] = dict(clusters=s, mean=float(np.mean(s)), worst=float(min(s)))
a, b = json.load(open("results/final.json")), json.load(open("results/final_nofmt.json")); yv = np.array(a["y"]); pa, pb = np.array(a["p"]), np.array(b["p"])
d = Detector(True).fit(tr.text.tolist(), y); w, c, _ = d.parts(va.text.tolist()); pt = (w + c) / 2
rng = np.random.default_rng(SEED); D = []
for _ in range(2000):
    i = rng.integers(0, len(yv), len(yv)); D.append((auc(yv[i], pa[i]) - auc(yv[i], pb[i]), auc(yv[i], pb[i]) - auc(yv[i], pt[i])))
D = np.array(D); out.update(text_only_val=auc(yv, pt), gain_format_ci=np.percentile(D[:, 0], [2.5, 97.5]).tolist(), gain_stylo_ci=np.percentile(D[:, 1], [2.5, 97.5]).tolist())
json.dump(out, open("results/final_lodo.json", "w"), indent=1); print(out)
