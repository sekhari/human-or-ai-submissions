"""check the first notebook's claims: reproduce its audit score and submission, look for leaks and inflated validation."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score as auc
from sklearn.model_selection import train_test_split

from src.baseline import fit_lexical, fit_style
from src.data import load, load_all
from src.harness import REPORTS

ROOT = Path(__file__).resolve().parents[1]
tr, un, te = load_all()
t, y = tr.text.values, tr.label.values
sub, backup = pd.read_csv(ROOT / "legacy/submission.csv"), pd.read_csv(ROOT / "legacy/submission_format_backup.csv")
res = {}

# 1. their exact audit split and recipe
dev, audit = train_test_split(np.arange(len(t)), test_size=.2, stratify=y, random_state=5785)
lx, st = fit_lexical(list(t[dev]), y[dev]), fit_style(list(t[dev]), y[dev])
a = .35 * lx(list(t[audit])) + .65 * st(list(t[audit]))
res["claimed_audit"], res["reproduced_audit"] = .998243, auc(y[audit], a)
print("audit claimed 0.998243 reproduced", round(res["reproduced_audit"], 6), flush=True)

# 2. does their csv come from that recipe, refit on all rows
lx, st = fit_lexical(list(t), y), fit_style(list(t), y)
mine = .35 * lx(list(te.text.values)) + .65 * st(list(te.text.values))
res["submission_ids_match_test"] = bool((sub.id.values == te.id.values).all())
res["spearman_submission_vs_rerun"] = float(spearmanr(sub.label, mine)[0])
res["spearman_backup_vs_submission"] = float(spearmanr(backup.label, sub.label)[0])
res["submission_range"] = [float(sub.label.min()), float(sub.label.max())]
res["submission_unique"] = int(sub.label.nunique())
res["submission_mean"] = float(sub.label.mean())
print("spearman with rerun", round(res["spearman_submission_vs_rerun"], 4), flush=True)

# 3. leaks through row order or id
res["train_label_vs_id_spearman"] = float(spearmanr(tr.id, y)[0])
res["id_auc"] = float(auc(y, tr.id))
res["submission_vs_id_spearman"] = float(spearmanr(sub.id, sub.label)[0])
res["label_run_longest"] = int(max(len(g) for g in "".join(map(str, y)).replace("01", "0 1").replace("10", "1 0").split()))
res["label_flip_rate"] = float((y[1:] != y[:-1]).mean())

# 4. near duplicates that could inflate validation, with no shared exact text
v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=200000, sublinear_tf=True, dtype=np.float32)
x = v.fit_transform(t)
sim = x[audit] @ x[dev].T
top = np.asarray(sim.max(axis=1).todense()).ravel()
nn = np.asarray(sim.argmax(axis=1)).ravel()
res["near_dup_audit_gt_0.5"], res["near_dup_audit_gt_0.8"] = int((top > .5).sum()), int((top > .8).sum())
res["max_sim_quantiles"] = np.quantile(top, [.5, .9, .99, 1]).round(3).tolist()
far = top < np.quantile(top, .5)
res["audit_auc_on_rows_with_least_similar_neighbor"] = float(auc(y[audit][far], a[far]))
res["nearest_neighbor_label_agreement"] = float((y[dev][nn] == y[audit]).mean())

# 5. a nearest neighbor alone. high value would mean validation is carried by copies
res["one_nn_similarity_signed_auc"] = float(auc(y[audit], np.where(y[dev][nn] == 1, top, -top)))
print(json.dumps(res, indent=1))
(REPORTS / "17_audit_first_notebook.json").write_text(json.dumps(res, indent=1))
