"""Robustness experiment."""

from sklearn.inspection import permutation_importance

COLS = STYLO_NAMES[:11] + [f"punct{p!r}" for p in PUNCT] + LM_NAMES      # 11 + 12 + 8 = 31 columns
X_all = F["stylo_lm"]["train"]
assert X_all.shape[1] == len(COLS), "expected stylometric + LM columns; did Part 6 finish on the GPU?"
col_ix = {c: i for i, c in enumerate(COLS)}
NEWLINE = f"punct{chr(10)!r}"                                              # the line-break rate column

def probe(selected):
    # same protocol as the notebook: one fit scored on the fixed validation set, plus grouped CV
    ci = [col_ix[c] for c in selected]
    Xs = X_all[:, ci]
    m = make_model("stylo_lm").fit(Xs[tr_idx], y[tr_idx])
    val = roc_auc_score(y[val_idx], score(m, Xs[val_idx]))
    aucs = []
    for a, b in GroupKFold(5).split(Xs[tr_idx], y[tr_idx], groups_all[tr_idx]):
        mm = make_model("stylo_lm").fit(Xs[tr_idx][a], y[tr_idx][a])
        aucs.append(roc_auc_score(y[tr_idx][b], score(mm, Xs[tr_idx][b])))
    return val, float(np.mean(aucs))

SUSPECTS = ["chars", "words", "n_sentences", NEWLINE]                      # length and layout features
sets = {
    "all 31 features": COLS,
    "LM features only (8)": LM_NAMES,
    "stylometric only (23)": COLS[:23],
    "all except length and line-break features": [c for c in COLS if c not in SUSPECTS],
    "only length and line-break features (4)": SUSPECTS,
}
rows = []
for name, cols in sets.items():
    v, g = probe(cols)
    rows.append(dict(feature_set=name, n_features=len(cols), val_auc=v, grouped_cv_auc=g))
    print(f"{name:45s} val={v:.4f}  grouped={g:.4f}")
display(pd.DataFrame(rows).round(4))

m_full = make_model("stylo_lm").fit(X_all[tr_idx], y[tr_idx])
pi = permutation_importance(m_full, X_all[val_idx], y[val_idx], scoring="roc_auc",
                            n_repeats=5, random_state=SEED)
print("\nTop 10 features by permutation importance (drop in validation AUROC when that column is shuffled):")
print(pd.Series(pi.importances_mean, index=COLS).sort_values(ascending=False).head(10).round(4))

def clean(t):
    # collapse every run of whitespace and line breaks into one space (a lazy "adversarial" edit)
    return re.sub(r"\s+", " ", t).strip()

def typo(t, rate=0.02, seed=SEED):
    # swap two adjacent letters inside roughly 2% of the longer words
    r = random.Random(seed)
    out = []
    for w in t.split(" "):
        if len(w) > 3 and r.random() < rate:
            i = r.randrange(len(w) - 1)
            w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        out.append(w)
    return " ".join(out)

def feats_for(tx):
    return np.hstack([np.array([stylo(t) for t in tx], dtype=np.float32), lm_features(tx)])

val_texts = [texts[i] for i in val_idx]
variants = {"original": val_texts,
            "whitespace cleaned": [clean(t) for t in val_texts],
            "cleaned + 2% typos": [typo(clean(t)) for t in val_texts]}

lm_ix = [col_ix[c] for c in LM_NAMES]
m_all = make_model("stylo_lm").fit(X_all[tr_idx], y[tr_idx])                       # all 31 features
m_lm = make_model("stylo_lm").fit(X_all[tr_idx][:, lm_ix], y[tr_idx])              # LM features only
m_word = make_model("word", best_C["word"]).fit(F["word"]["train"][tr_idx], y[tr_idx])

rows = []
for vname, tx in variants.items():
    Xv = X_all[val_idx] if vname == "original" else feats_for(tx)
    rows.append(dict(
        text_variant=vname,
        stylo_plus_lm_31=roc_auc_score(y[val_idx], score(m_all, Xv)),
        lm_only_8=roc_auc_score(y[val_idx], score(m_lm, Xv[:, lm_ix])),
        word_tfidf=roc_auc_score(y[val_idx], score(m_word, word_vec.transform(tx)))))
display(pd.DataFrame(rows).round(4))

def drop_words(t, rate=0.05, seed=SEED):
    # randomly delete about 5% of the words; used ONLY for evaluation, never for training
    r = random.Random(seed)
    return " ".join(w for w in t.split(" ") if r.random() >= rate)

tr_texts = [texts[i] for i in tr_idx]
# one edited copy of every training passage: whitespace collapsed, then about 3% of long words get a typo
aug_texts = [typo(clean(t), rate=0.03, seed=SEED + k) for k, t in enumerate(tr_texts)]

t0 = time.time()
X_aug = feats_for(aug_texts)                      # stylometric + LM features of the edited copies
print(f"edited-copy features ready ({time.time() - t0:.0f}s)")

y_tr = y[tr_idx]
X_tr_orig = X_all[tr_idx]
W_tr_orig = F["word"]["train"][tr_idx]

m_orig = make_model("stylo_lm").fit(X_tr_orig, y_tr)
m_aug = make_model("stylo_lm").fit(np.vstack([X_tr_orig, X_aug]), np.r_[y_tr, y_tr])
w_orig = make_model("word", best_C["word"]).fit(W_tr_orig, y_tr)
w_aug = make_model("word", best_C["word"]).fit(sp.vstack([W_tr_orig, word_vec.transform(aug_texts)]),
                                                np.r_[y_tr, y_tr])

val_texts = [texts[i] for i in val_idx]
V = {
    "original": val_texts,
    "whitespace cleaned": [clean(t) for t in val_texts],
    "cleaned + 2% typos (edit family seen in training)":
        [typo(clean(t), rate=0.02, seed=SEED + 10_000 + k) for k, t in enumerate(val_texts)],
    "5% words deleted (edit family NEVER seen in training)":
        [drop_words(t, 0.05, SEED + k) for k, t in enumerate(val_texts)],
}
rows = []
for vname, tx in V.items():
    Xv = X_all[val_idx] if vname == "original" else feats_for(tx)
    Wv = word_vec.transform(tx)
    rows.append({
        "text_variant": vname,
        "stylo+LM, originals only": roc_auc_score(y[val_idx], score(m_orig, Xv)),
        "stylo+LM, with edited copies": roc_auc_score(y[val_idx], score(m_aug, Xv)),
        "word TF-IDF, originals only": roc_auc_score(y[val_idx], score(w_orig, Wv)),
        "word TF-IDF, with edited copies": roc_auc_score(y[val_idx], score(w_aug, Wv)),
    })
robust = pd.DataFrame(rows)
display(robust.round(4))
robust.round(4).to_csv(f"{OUT_DIR}/robustness_table.csv", index=False)

X_te = F["stylo_lm"]["test"]
assert X_te.shape[1] == len(COLS), "test features must have the same 31 columns as train"

# 3a: can a model tell train rows from test rows using only the 31 features? Which features give it away?
Xadv = np.vstack([X_all, X_te])
yadv = np.r_[np.zeros(len(X_all)), np.ones(len(X_te))]
Xa, Xb, ya, yb = train_test_split(Xadv, yadv, test_size=0.3, stratify=yadv, random_state=SEED)
adv = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, early_stopping=False,
                                     random_state=SEED).fit(Xa, ya)
print(f"train vs test from the 31 features: AUROC {roc_auc_score(yb, adv.predict_proba(Xb)[:, 1]):.3f}"
      "   (0.5 = indistinguishable)")
pi_adv = permutation_importance(adv, Xb, yb, scoring="roc_auc", n_repeats=5, random_state=SEED)
print("\nFeatures that give the test set away (drop in AUROC when shuffled):")
print(pd.Series(pi_adv.importances_mean, index=COLS).sort_values(ascending=False).head(8).round(4))

# 3b: how far has each feature's average moved, in units of the train standard deviation?
mu, sd = X_all.mean(0), X_all.std(0) + 1e-9
smd = pd.Series((X_te.mean(0) - mu) / sd, index=COLS)
print("\nBiggest average shifts, test vs train (in train standard deviations):")
print(smd.reindex(smd.abs().sort_values(ascending=False).index).head(8).round(2))

# 3c: the layout artifact, directly
nl = col_ix[NEWLINE]
print(f"\nshare of passages containing any line break:  train {np.mean(X_all[:, nl] > 0):.3f}   test {np.mean(X_te[:, nl] > 0):.3f}")
print(f"median passage length in characters:           train {np.median(X_all[:, col_ix['chars']]):.0f}   test {np.median(X_te[:, col_ix['chars']]):.0f}")

# 3d: what does the edit-trained model do on the test set? (the model never sees test labels, there are none)
p_val = m_aug.predict_proba(X_all[val_idx])[:, 1]
p_te = m_aug.predict_proba(X_te)[:, 1]
print("\nmodel output, validation vs test:")
print(f"  share predicted AI (p >= 0.5):          val {np.mean(p_val >= 0.5):.3f}   test {np.mean(p_te >= 0.5):.3f}")
print(f"  share unsure (0.2 < p < 0.8):           val {np.mean((p_val > 0.2) & (p_val < 0.8)):.3f}   test {np.mean((p_te > 0.2) & (p_te < 0.8)):.3f}")
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.hist(p_val, bins=40, alpha=0.6, density=True, label="validation")
ax.hist(p_te, bins=40, alpha=0.6, density=True, label="test")
ax.set_xlabel("predicted probability of AI"); ax.set_ylabel("density"); ax.legend()
ax.set_title("Model output: validation vs test")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/val_vs_test_predictions.png", dpi=120); plt.show()

SHIFTY = ["digit_ratio", "avg_word_len"] + [f"punct{p!r}" for p in (",", "(", ";", "—")]   # the biggest movers
assert all(c in col_ix for c in SHIFTY), "a feature name did not match COLS"

sets = {
    "all 31 features": COLS,
    "without the 6 most-shifted style features (25)": [c for c in COLS if c not in SHIFTY],
    "LM features only (8)": LM_NAMES,
}

edit_key = "cleaned + 2% typos (edit family seen in training)"
Xv_edit = feats_for(V[edit_key])                      # validation passages after the edit, same edit as training
y_val = y[val_idx]

def adv_auc(ci):
    # how easily can a model tell train from test using only these columns? (lower = more alike)
    A, B, a_lab, b_lab = train_test_split(Xadv[:, ci], yadv, test_size=0.3, stratify=yadv, random_state=SEED)
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, early_stopping=False,
                                       random_state=SEED).fit(A, a_lab)
    return roc_auc_score(b_lab, m.predict_proba(B)[:, 1])

rows = []
for name, cols in sets.items():
    ci = [col_ix[c] for c in cols]
    # trained on the originals plus their edited copies, like the best model from the robustness experiment
    m = make_model("stylo_lm").fit(np.vstack([X_all[tr_idx][:, ci], X_aug[:, ci]]), np.r_[y_tr, y_tr])
    p_val_ = m.predict_proba(X_all[val_idx][:, ci])[:, 1]
    p_te_ = m.predict_proba(X_te[:, ci])[:, 1]
    unsure = lambda p: float(np.mean((p > 0.2) & (p < 0.8)))
    rows.append(dict(feature_set=name, n_features=len(ci),
                     val_original=roc_auc_score(y_val, p_val_),
                     val_edited=roc_auc_score(y_val, m.predict_proba(Xv_edit[:, ci])[:, 1]),
                     train_vs_test_auc=adv_auc(ci),
                     unsure_on_val=unsure(p_val_), unsure_on_test=unsure(p_te_)))
stab = pd.DataFrame(rows)
display(stab.round(4))
stab.round(4).to_csv(f"{OUT_DIR}/stability_table.csv", index=False)

STYLE_COLS = [c for c in COLS if c not in SHIFTY]            # the 25 stable-ish features (selected configuration)
ci_style = [col_ix[c] for c in STYLE_COLS]
ci_lm = [col_ix[c] for c in LM_NAMES]

# three models, all trained on the 8,000 originals plus their edited copies
m_style = make_model("stylo_lm").fit(np.vstack([X_tr_orig[:, ci_style], X_aug[:, ci_style]]), np.r_[y_tr, y_tr])
m_lm = make_model("stylo_lm").fit(np.vstack([X_tr_orig[:, ci_lm], X_aug[:, ci_lm]]), np.r_[y_tr, y_tr])
# w_aug (word TF-IDF + LR trained with edited copies) already exists from the robustness cell

def strip_all(t):
    # a harsh edit NO model is trained on: lowercase everything and delete all punctuation
    return re.sub(r"[^\w\s]", "", t.lower())

V2 = {
    "original": val_texts,
    "cleaned + 2% typos": V[edit_key],
    "5% words deleted": V["5% words deleted (edit family NEVER seen in training)"],
    "lowercase, no punctuation (never seen)": [strip_all(t) for t in val_texts],
}
z = lambda s: (s - s.mean()) / (s.std() + 1e-9)               # put each model's scores on one scale

S = {}
for vname, tx in V2.items():
    if vname == "original":
        Xv = X_all[val_idx]
    elif vname == "cleaned + 2% typos":
        Xv = Xv_edit
    else:
        Xv = feats_for(tx)
    S[vname] = (z(score(m_style, Xv[:, ci_style])), z(score(m_lm, Xv[:, ci_lm])),
                z(score(w_aug, word_vec.transform(tx))))

BLENDS = {                       # weights are (stylo+LM, LM alone, word TF-IDF); fixed in advance, not optimized
    "stylo+LM alone": (1, 0, 0),
    "LM alone": (0, 1, 0),
    "word TF-IDF alone": (0, 0, 1),
    "50% stylo+LM, 50% word": (0.5, 0, 0.5),
    "40% stylo+LM, 20% LM, 40% word": (0.4, 0.2, 0.4),
    "equal thirds": (1 / 3, 1 / 3, 1 / 3),
}
rows = []
for bname, w in BLENDS.items():
    aucs = {v: roc_auc_score(y[val_idx], w[0] * s[0] + w[1] * s[1] + w[2] * s[2]) for v, s in S.items()}
    rows.append({"blend": bname, **aucs, "worst_case": min(aucs.values()), "average": float(np.mean(list(aucs.values())))})
blend_table = pd.DataFrame(rows)
display(blend_table.round(4))
blend_table.round(4).to_csv(f"{OUT_DIR}/blend_table.csv", index=False)

WEIGHTS = (0.5, 0.0, 0.5)         # (stylo+LM, LM alone, word TF-IDF)

t0 = time.time()
aug_val_texts = [typo(clean(texts[i]), rate=0.03, seed=SEED + 50_000 + k) for k, i in enumerate(val_idx)]
X_aug_val = feats_for(aug_val_texts)                          # edited copies of the 2,000 validation rows
y_full = np.r_[y[tr_idx], y[val_idx], y[tr_idx], y[val_idx]]  # originals (tr, val) then edited copies (tr, val)
X_full = np.vstack([X_tr_orig, X_all[val_idx], X_aug, X_aug_val])
W_full = sp.vstack([F["word"]["train"][tr_idx], F["word"]["train"][val_idx],
                    word_vec.transform(aug_texts), word_vec.transform(aug_val_texts)])

f_style = make_model("stylo_lm").fit(X_full[:, ci_style], y_full)
f_lm = make_model("stylo_lm").fit(X_full[:, ci_lm], y_full)
f_word = make_model("word", best_C["word"]).fit(W_full, y_full)

s_te = (z(score(f_style, X_te[:, ci_style])), z(score(f_lm, X_te[:, ci_lm])),
        z(score(f_word, F["word"]["test"])))
blend_te = WEIGHTS[0] * s_te[0] + WEIGHTS[1] * s_te[1] + WEIGHTS[2] * s_te[2]
test_p2 = 1 / (1 + np.exp(-2 * blend_te))                     # squashed into (0, 1); only the ordering matters for AUROC
print(f"final fit done in {time.time() - t0:.0f}s | weights {WEIGHTS}")

sample_path = f"{DATA_DIR}/sample_submission.csv"
if os.path.exists(sample_path):
    sub2 = pd.read_csv(sample_path)
    id_c, val_c = sub2.columns[0], sub2.columns[1]
    sub2[val_c] = sub2[id_c].map(dict(zip(test[ID_COL], test_p2)))
else:
    print("WARNING: no sample_submission.csv found. Check the column names against the Kaggle sample.")
    sub2 = pd.DataFrame({ID_COL: test[ID_COL], "label": test_p2})
    val_c = "label"
assert len(sub2) == len(test) and sub2[val_c].notna().all() and sub2[val_c].between(0, 1).all()
sub2_path = f"{OUT_DIR}/submission_v2.csv"
sub2.to_csv(sub2_path, index=False)
print("saved", sub2_path)
if "test_p" in globals():
    from scipy.stats import spearmanr
    print(f"rank agreement with the Part 10 submission (Spearman): {spearmanr(test_p, test_p2)[0]:.3f}")

# Save the exact final estimators and fitted vocabulary for inference on raw text.
import joblib
from pathlib import Path
from transformers.utils.hub import cached_file
bundle_dir = Path(OUT_DIR) / "distilroberta_v1"
bundle_dir.mkdir(parents=True, exist_ok=True)
lm_config = Path(cached_file("distilgpt2", "config.json", local_files_only=True))
lm_revision = lm_config.parent.name if lm_config.parent.parent.name == "snapshots" else None
bundle = {"word_vec": word_vec, "f_word": f_word, "f_style": f_style, "ci_style": ci_style,
          "lm_source": "distilgpt2", "lm_revision": lm_revision}
pending = bundle_dir / "v2.joblib.tmp"
joblib.dump(bundle, pending, compress=3)
pending.replace(bundle_dir / "v2.joblib")

# Pair all component scores on identical text, preserving the historical v2 edit functions.
from pathlib import Path
run = Path(OUT_DIR) / "distilroberta_v1"
run.mkdir(exist_ok=True)
names = ["original", "clean_typos", "delete_5pct", "lower_no_punct"]
char_model = make_model("char", best_C["char"]).fit(F["char"]["train"][tr_idx], y[tr_idx])
probe_rows, char_rows = [], []
for name, (key, edited) in zip(names, V2.items()):
    style, lm, word = S[key]
    v2_margin = WEIGHTS[0] * style + WEIGHTS[1] * lm + WEIGHTS[2] * word
    probe_rows.append(pd.DataFrame({"id": train[ID_COL].iloc[val_idx].to_numpy(),
        "label": y[val_idx], "variant": name, "text": edited, "v2_score": v2_margin}))
    char_rows.append(pd.DataFrame({"id": train[ID_COL].iloc[val_idx].to_numpy(),
        "variant": name, "char_score": score(char_model, char_vec.transform(edited))}))
pd.concat(probe_rows).to_csv(run / "v2_validation_probes.csv.gz", index=False)
pd.concat(char_rows).to_csv(run / "char_validation_scores.csv", index=False)

# Retain original training-partition component scores for the optional overfitting diagnostic.
tr_style = z(score(m_style, X_tr_orig[:, ci_style]))
tr_word = z(score(w_aug, F["word"]["train"][tr_idx]))
pd.DataFrame({"id": train[ID_COL].iloc[tr_idx].to_numpy(),
              "v2_score": 0.5 * tr_style + 0.5 * tr_word}).to_csv(
    run / "v2_train_scores.csv", index=False)
