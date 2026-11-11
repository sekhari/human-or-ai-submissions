"""Baselines experiment."""

# PART 0: SETUP AND SETTINGS
import os, sys, time, re, random, warnings, subprocess
from collections import Counter
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.model_selection import train_test_split, GroupKFold, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
try:
    from IPython.display import display
except ImportError:
    display = print

# ---------- settings you may change ----------
SEED = 42              # one seed for every random choice, so reruns match
VAL_SIZE = 2000        # fixed stratified validation rows (assignment rule)
N_CLUSTERS = 10        # topic/style clusters used for the harder grouped validation
QUICK = False          # True = smaller tuning grid, for a fast debugging run
TEXT_COL = None        # set by hand if auto-detection picks the wrong column
LABEL_COL = None
ID_COL = None

try:
    import torch
    HAS_GPU = torch.cuda.is_available()
except ImportError:
    HAS_GPU = False
RUN_LM_FEATURES = HAS_GPU    # perplexity features from a small pretrained language model
RUN_EMBEDDINGS = HAS_GPU     # sentence embeddings from a small pretrained model
print("GPU available:", HAS_GPU, "| LM features:", RUN_LM_FEATURES, "| embeddings:", RUN_EMBEDDINGS)

random.seed(SEED); np.random.seed(SEED)

RESULTS = []   # the results log, saved to Drive at the end
def log(part, model, val_auc=None, train_auc=None, grouped_auc=None, secs=None, notes=""):
    RESULTS.append(dict(part=part, model=model, train_auc=train_auc, val_auc=val_auc,
                        grouped_cv_auc=grouped_auc, seconds=secs, seed=SEED, notes=notes))
    f = lambda v: "n/a" if v is None else f"{v:.4f}"
    print(f"[{part}] {model}: val={f(val_auc)} train={f(train_auc)} grouped_cv={f(grouped_auc)} {notes}")

# Load the competition files.
from pathlib import Path
DATA_DIR = os.environ.get("DATA_DIR", "data")
OUT_DIR = os.environ.get("OUT_DIR", "outputs")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
train = pd.read_csv(f"{DATA_DIR}/train.csv")
unl = pd.read_csv(f"{DATA_DIR}/unlabeled.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
assert len(train) == 10000 and train.id.is_unique
assert train.label.value_counts().to_dict() == {0: 5000, 1: 5000}
assert len(test) == len(unl) == 3000

# PART 0 (continued): FIND THE TEXT, LABEL AND ID COLUMNS
# If the guess below is wrong, set TEXT_COL / LABEL_COL / ID_COL in the settings cell and rerun.
obj_cols = [c for c in train.columns if not pd.api.types.is_numeric_dtype(train[c])]
if TEXT_COL is None:
    TEXT_COL = max(obj_cols, key=lambda c: train[c].astype(str).str.len().mean())
if LABEL_COL is None:
    cands = [c for c in train.columns if c not in test.columns and train[c].nunique() <= 2]
    assert cands, "Could not find the label column. Set LABEL_COL in the settings cell."
    LABEL_COL = cands[0]
if ID_COL is None:
    cands = [c for c in train.columns if c in test.columns and "id" in c.lower() and c != TEXT_COL]
    ID_COL = cands[0] if cands else "row_id"
if ID_COL == "row_id":
    for d in (train, unl, test):
        d["row_id"] = np.arange(len(d))
print(f"text column={TEXT_COL!r}  label column={LABEL_COL!r}  id column={ID_COL!r}")

for d in (train, unl, test):
    d[TEXT_COL] = d[TEXT_COL].fillna("").astype(str)
texts = train[TEXT_COL].tolist()
unl_texts = unl[TEXT_COL].tolist()
test_texts = test[TEXT_COL].tolist()
y = train[LABEL_COL].astype(int).values
assert set(np.unique(y)) <= {0, 1}, "label must be 0 (human) or 1 (AI)"
print("class counts [human, AI]:", np.bincount(y))

# Load or recreate the seed-42 split, then use input order as in the recorded run.
all_idx = np.arange(len(train))
val_ids_path = Path(OUT_DIR) / "val_ids.csv"
if not val_ids_path.exists():
    _, selected = train_test_split(all_idx, test_size=VAL_SIZE, stratify=y, random_state=SEED)
    train[[ID_COL]].iloc[selected].to_csv(val_ids_path, index=False)
saved = pd.read_csv(val_ids_path)[ID_COL]
assert len(saved) == VAL_SIZE and saved.is_unique and saved.isin(train[ID_COL]).all()
is_val = train[ID_COL].isin(saved).to_numpy()
tr_idx, val_idx = all_idx[~is_val], all_idx[is_val]
assert np.bincount(y[val_idx]).tolist() == [1000, 1000]
print("Train and validation counts", len(tr_idx), len(val_idx))

# PART 1: LENGTH AND CLASS BALANCE
wc = np.array([len(t.split()) for t in texts])
print(pd.Series(wc).describe().round(1))
print("mean words per passage by class:", {name: round(wc[y == c].mean(), 1) for c, name in ((0, "human"), (1, "AI"))})
fig, ax = plt.subplots(figsize=(7, 3.5))
for c, name in ((0, "human"), (1, "AI")):
    ax.hist(wc[y == c], bins=40, alpha=0.6, label=name)
ax.set_xlabel("words per passage"); ax.set_ylabel("count"); ax.legend(); ax.set_title("Passage length by class")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/length_hist.png", dpi=120); plt.show()

# PART 2: TF-IDF FEATURES
F = {}   # name -> {"train": rows aligned with train.csv, "unl": ..., "test": ...}
fit_texts = texts + unl_texts

t0 = time.time()
word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=300_000,
                           sublinear_tf=True, dtype=np.float32).fit(fit_texts)
F["word"] = {k: word_vec.transform(v) for k, v in (("train", texts), ("unl", unl_texts), ("test", test_texts))}
print("word TF-IDF", F["word"]["train"].shape, f"{time.time() - t0:.0f}s")

t0 = time.time()
char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=200_000,
                           sublinear_tf=True, dtype=np.float32).fit(fit_texts)
F["char"] = {k: char_vec.transform(v) for k, v in (("train", texts), ("unl", unl_texts), ("test", test_texts))}
print("char TF-IDF", F["char"]["train"].shape, f"{time.time() - t0:.0f}s")

from distilroberta.features import PUNCT, STYLO_NAMES, stylo

t0 = time.time()
F["stylo"] = {"train": np.array([stylo(t) for t in texts], dtype=np.float32),
              "test": np.array([stylo(t) for t in test_texts], dtype=np.float32)}
print("stylometric features", F["stylo"]["train"].shape, f"{time.time() - t0:.0f}s")

# PART 3a: CLUSTERS (used for grouped validation from here on)
t0 = time.time()
Xfit = sp.vstack([F["word"]["train"], F["word"]["unl"]])
Z = Normalizer().fit_transform(TruncatedSVD(n_components=100, random_state=SEED).fit_transform(Xfit))
km = KMeans(n_clusters=N_CLUSTERS, n_init=3, random_state=SEED).fit(Z)
groups_all = km.labels_[:len(texts)]          # cluster id for every train row
print("rows per cluster:", np.bincount(groups_all))
print("AI share per cluster:", np.round([y[groups_all == g].mean() for g in range(N_CLUSTERS)], 2))
print(f"{time.time() - t0:.0f}s")

# PART 3b: ADVERSARIAL VALIDATION (no labels used)
def adversarial_auc(A, B, name):
    X = sp.vstack([A, B]).tocsr()
    lab = np.r_[np.zeros(A.shape[0]), np.ones(B.shape[0])]
    oof = np.zeros(len(lab))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, lab):
        m = LogisticRegression(C=3, solver="liblinear").fit(X[a], lab[a])
        oof[b] = m.decision_function(X[b])
    auc = roc_auc_score(lab, oof)
    print(f"train vs {name}: AUROC {auc:.3f}   (0.5 = indistinguishable, 1.0 = easy to tell apart)")
    return auc

adv_unl = adversarial_auc(F["word"]["train"], F["word"]["unl"], "unlabeled")
adv_test = adversarial_auc(F["word"]["train"], F["word"]["test"], "test")

# PART 4: MODEL DEFINITIONS AND HELPERS
C_GRID = [3, 10, 30] if QUICK else [1, 3, 10, 30, 100]

def make_model(name, C=None):
    if name in ("word", "char"):
        return LogisticRegression(C=C, solver="liblinear")
    if name == "emb":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
    if name == "stylo_lm":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                              early_stopping=False, random_state=SEED)
    raise ValueError(name)

def score(model, X):
    return model.decision_function(X)       # higher = more likely AI (fine for AUROC)

def grouped_cv(name, C, idx, n_splits=5):
    # train on some clusters, test on held-out clusters, return the mean AUROC
    X, yy, g = F[name]["train"][idx], y[idx], groups_all[idx]
    aucs = []
    for a, b in GroupKFold(n_splits).split(X, yy, g):
        if len(np.unique(yy[b])) < 2:
            continue
        m = make_model(name, C).fit(X[a], yy[a])
        aucs.append(roc_auc_score(yy[b], score(m, X[b])))
    return float(np.mean(aucs))

def fit_eval(name, C, tr, ev):
    t0 = time.time()
    m = make_model(name, C).fit(F[name]["train"][tr], y[tr])
    tr_auc = roc_auc_score(y[tr], score(m, F[name]["train"][tr]))
    ev_auc = roc_auc_score(y[ev], score(m, F[name]["train"][ev]))
    return m, tr_auc, ev_auc, time.time() - t0

# PART 4 (continued): TUNE C WITH GROUPED CV ON THE TRAINING PARTITION
best_C, g_auc_best, tuning_rows = {}, {}, []
for name in ("word", "char"):
    scores_by_C = []
    for C in C_GRID:
        t0 = time.time()
        g = grouped_cv(name, C, tr_idx)
        scores_by_C.append((C, g)); tuning_rows.append(dict(model=name, C=C, grouped_cv_auc=g))
        print(f"{name:5s} C={C:<5} grouped-CV AUROC {g:.4f}  ({time.time() - t0:.0f}s)")
    best_C[name], g_auc_best[name] = max(scores_by_C, key=lambda r: r[1])
    print(f"--> chosen C for {name}: {best_C[name]}\n")
pd.DataFrame(tuning_rows).to_csv(f"{OUT_DIR}/tuning_grid.csv", index=False)

# PART 4 (continued): REPORT BOTH BASELINES ON THE FIXED VALIDATION SET
for name, label in (("word", "word TF-IDF + logistic regression"), ("char", "char TF-IDF + logistic regression")):
    m, tr_auc, va_auc, secs = fit_eval(name, best_C[name], tr_idx, val_idx)
    log("4 baseline", f"{label} (C={best_C[name]})", va_auc, tr_auc, g_auc_best[name], secs)

# PART 5: LEARNING CURVE (word TF-IDF + LR)
sizes = [s for s in (500, 1000, 2000, 5000, 8000) if s <= len(tr_idx)]
if sizes[-1] != len(tr_idx):
    sizes.append(len(tr_idx))
order = np.random.RandomState(SEED).permutation(tr_idx)    # nested random subsets
lc = []
for n in sizes:
    m, tr_auc, va_auc, secs = fit_eval("word", best_C["word"], order[:n], val_idx)
    lc.append(dict(train_rows=n, train_auc=tr_auc, val_auc=va_auc))
    print(f"{n:>5} rows: train AUROC {tr_auc:.4f} | validation AUROC {va_auc:.4f}")
lc = pd.DataFrame(lc); lc.to_csv(f"{OUT_DIR}/learning_curve.csv", index=False)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(lc.train_rows, lc.train_auc, "o-", label="training AUROC")
ax.plot(lc.train_rows, lc.val_auc, "o-", label="validation AUROC")
ax.set_xscale("log"); ax.set_xlabel("training rows"); ax.set_ylabel("AUROC"); ax.legend(); ax.set_title("Learning curve")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/learning_curve.png", dpi=120); plt.show()

from distilroberta.features import LM_NAMES, lm_features

# PART 6: PRETRAINED-MODEL FEATURES
def cached(path, fn):
    if os.path.exists(path):
        print("loaded cache:", path); return np.load(path)
    arr = fn(); np.save(path, arr); return arr

def embed(texts_, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(model_name, device="cuda"); st.max_seq_length = 256
    return st.encode(texts_, batch_size=64, show_progress_bar=True, normalize_embeddings=True).astype(np.float32)

if RUN_LM_FEATURES:
    t0 = time.time()
    lm_tr = cached(f"{OUT_DIR}/lm_train.npy", lambda: lm_features(texts))
    lm_te = cached(f"{OUT_DIR}/lm_test.npy", lambda: lm_features(test_texts))
    F["stylo_lm"] = {"train": np.hstack([F["stylo"]["train"], lm_tr]), "test": np.hstack([F["stylo"]["test"], lm_te])}
    print(f"perplexity features ready ({time.time() - t0:.0f}s)")
else:
    print("No GPU: using stylometric features only for the 'stylo_lm' model.")
    F["stylo_lm"] = {"train": F["stylo"]["train"], "test": F["stylo"]["test"]}

if RUN_EMBEDDINGS:
    t0 = time.time()
    F["emb"] = {"train": cached(f"{OUT_DIR}/emb_train.npy", lambda: embed(texts)),
                "test": cached(f"{OUT_DIR}/emb_test.npy", lambda: embed(test_texts))}
    print(f"embeddings ready ({time.time() - t0:.0f}s)")

# PART 6 (continued): EVALUATE THE NEW FEATURE SETS
for name, label in (("stylo_lm", "stylometric + perplexity features, gradient boosting"),
                    ("emb", "sentence embeddings + logistic regression")):
    if name in F:
        g = grouped_cv(name, None, tr_idx)
        m, tr_auc, va_auc, secs = fit_eval(name, None, tr_idx, val_idx)
        log("6 features", label, va_auc, tr_auc, g, secs)

# PART 7: SELF-TRAINING
C = best_C["word"]
Xw_tr, Xw_val, Xw_unl = F["word"]["train"][tr_idx], F["word"]["train"][val_idx], F["word"]["unl"]
base = make_model("word", C).fit(Xw_tr, y[tr_idx])
base_val = roc_auc_score(y[val_idx], score(base, Xw_val))

p_unl = base.predict_proba(Xw_unl)[:, 1]
conf = np.abs(p_unl - 0.5)
mask = conf > 0.4                                   # predicted probability above 0.9 or below 0.1
rule = "p > 0.9 or p < 0.1"
if mask.sum() < 200:                                # too few: fall back to the most confident 30%
    mask = conf >= np.quantile(conf, 0.7); rule = "most confident 30%"
pseudo = (p_unl[mask] > 0.5).astype(int)
print(f"added {mask.sum()} of {len(p_unl)} unlabeled rows ({rule}); pseudo-label AI share {pseudo.mean():.2f}")

st = make_model("word", C).fit(sp.vstack([Xw_tr, Xw_unl[mask]]), np.r_[y[tr_idx], pseudo])
st_val = roc_auc_score(y[val_idx], score(st, Xw_val))
log("7 unlabeled", "word TF-IDF + LR, supervised only", base_val)
log("7 unlabeled", "word TF-IDF + LR, self-training", st_val, notes=f"{mask.sum()} pseudo-labels")
print(f"change from self-training: {st_val - base_val:+.4f} AUROC")

# PART 8: STACKING
BASE = [n for n in ("word", "char", "stylo_lm", "emb") if n in F]
CHOSEN_C = {"word": best_C["word"], "char": best_C["char"], "stylo_lm": None, "emb": None}
print("base models:", BASE)

def stack_fit_predict(tr, eval_specs):
    # eval_specs: {label: (split, idx or None)}. Returns {label: (base scores, stack probability)}
    # plus "oof" (out-of-fold base scores on tr), "meta" (the fitted meta-model) and "groups".
    # Scores for the eval sets are the AVERAGE of the five fold models. That keeps them on the same
    # scale as the out-of-fold scores the meta-model learned from (a separate full refit would not).
    yy, g = y[tr], groups_all[tr]
    folds = list(GroupKFold(5).split(tr, yy, g))
    oof = np.zeros((len(tr), len(BASE)))
    S = {}
    for j, n in enumerate(BASE):
        X = F[n]["train"][tr]
        Xe = {lab: (F[n][sp_] if ix is None else F[n][sp_][ix]) for lab, (sp_, ix) in eval_specs.items()}
        for lab in Xe:
            S.setdefault(lab, np.zeros((Xe[lab].shape[0], len(BASE))))
        for a, b in folds:
            m = make_model(n, CHOSEN_C[n]).fit(X[a], yy[a])
            oof[b, j] = score(m, X[b])
            for lab in Xe:
                S[lab][:, j] += score(m, Xe[lab]) / len(folds)
    meta = make_pipeline(StandardScaler(), LogisticRegression(C=0.1)).fit(oof, yy)
    out = {"oof": oof, "meta": meta, "groups": g}
    for lab in S:
        out[lab] = (S[lab], meta.predict_proba(S[lab])[:, 1])
    return out

t0 = time.time()
res = stack_fit_predict(tr_idx, {"val": ("train", val_idx), "train": ("train", tr_idx)})
S_val, p_val = res["val"]
for j, n in enumerate(BASE):
    print(f"  {n:9s} validation AUROC {roc_auc_score(y[val_idx], S_val[:, j]):.4f}")
stack_val = roc_auc_score(y[val_idx], p_val)
stack_train = roc_auc_score(y[tr_idx], res["train"][1])
# approximate grouped-CV AUROC of the stack: per-fold mean, the same way the base models are scored
fold_aucs = []
for a, b in GroupKFold(5).split(res["oof"], y[tr_idx], res["groups"]):
    mm = make_pipeline(StandardScaler(), LogisticRegression(C=0.1)).fit(res["oof"][a], y[tr_idx][a])
    if len(np.unique(y[tr_idx][b])) == 2:
        fold_aucs.append(roc_auc_score(y[tr_idx][b], mm.decision_function(res["oof"][b])))
stack_grouped = float(np.mean(fold_aucs))
log("8 ensemble", "stack of " + ", ".join(BASE), stack_val, stack_train, stack_grouped,
    time.time() - t0, notes="grouped_cv for the stack is approximate")
print("meta-model weights (after scaling):", dict(zip(BASE, np.round(res["meta"][-1].coef_[0], 2))))

# PART 9a: GAP AND LENGTH BINS
print(f"final stack: training AUROC {stack_train:.4f} vs validation AUROC {stack_val:.4f} "
      f"(gap {stack_train - stack_val:+.4f}; the training figure is partly in-sample, so it is optimistic)")
vdf = pd.DataFrame({"row": val_idx, "id": train[ID_COL].iloc[val_idx].values,
                    "y": y[val_idx], "p": p_val, "words": wc[val_idx]})
vdf["length_bin"] = pd.qcut(vdf.words, 3, labels=["short", "medium", "long"])
rows = []
for b, g in vdf.groupby("length_bin", observed=True):
    rows.append(dict(length_bin=b, n=len(g), words=f"{g.words.min()} to {g.words.max()}",
                     val_auc=roc_auc_score(g.y, g.p)))
display(pd.DataFrame(rows))

# PART 9b: FALSE POSITIVES AND FALSE NEGATIVES (threshold 0.5)
fp_all = vdf[(vdf.y == 0) & (vdf.p >= 0.5)].sort_values("p", ascending=False)
fn_all = vdf[(vdf.y == 1) & (vdf.p < 0.5)].sort_values("p")
print(f"errors at 0.5: {len(fp_all)} false positives, {len(fn_all)} false negatives out of {len(vdf)}")
for title, d in (("FALSE POSITIVES (human, predicted AI)", fp_all.head(5)),
                 ("FALSE NEGATIVES (AI, predicted human)", fn_all.head(5))):
    print("\n" + title)
    for _, r in d.iterrows():
        print(f"  id={r['id']}  P(AI)={r.p:.3f}  words={r.words}")
        print("    " + texts[int(r.row)][:220].replace("\n", " ") + " ...")
vdf.to_csv(f"{OUT_DIR}/val_predictions.csv", index=False)

# PART 10: REFIT ON ALL LABELED ROWS AND PREDICT THE TEST SET
t0 = time.time()
final = stack_fit_predict(np.arange(len(train)), {"test": ("test", None)})
test_p = final["test"][1]
print(f"final fit done in {time.time() - t0:.0f}s")

sample_path = f"{DATA_DIR}/sample_submission.csv"
if os.path.exists(sample_path):
    sub = pd.read_csv(sample_path)
    id_c, val_c = sub.columns[0], sub.columns[1]
    sub[val_c] = sub[id_c].map(dict(zip(test[ID_COL], test_p)))
else:
    print("WARNING: no sample_submission.csv found. Check the column names against the Kaggle sample.")
    sub = pd.DataFrame({ID_COL: test[ID_COL], "label": test_p})
    val_c = "label"

assert len(sub) == len(test), "row count differs from test.csv"
assert sub[val_c].notna().all() and sub[val_c].between(0, 1).all(), "probabilities missing or outside [0, 1]"
sub_path = f"{OUT_DIR}/submission_stack.csv"
sub.to_csv(sub_path, index=False)
print("saved", sub_path); display(sub.head())
print("test probability summary:", sub[val_c].describe().round(3).to_dict())

pd.DataFrame(RESULTS).to_csv(f"{OUT_DIR}/results_log.csv", index=False)
display(pd.DataFrame(RESULTS).round(4))
