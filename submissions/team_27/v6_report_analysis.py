"""Report analyses for the final model (Yang's v6: v5 style blend + frozen GPT-2 LightGBM branch).

Why this file exists
--------------------
The report (Submission V3) needs numbers that svm_frozen_lm_yx752_v6.ipynb does not print:
training AUROC, a learning curve, length-bin AUROC, false positive / false negative examples
and an unlabeled-data experiment, all for the *final* model rather than the older linear SVM.

What it does NOT change
-----------------------
The v6 model itself is untouched. Steps 0-3 of the notebook are executed verbatim from the
.ipynb (cells 1, 3, 4, 6, 8) so the features, tree settings, blend weight (0.75 LM branch /
0.25 v5) and calibration are exactly Yang's. The full-data refit / submission cell (10) is not
run, so no submission file is overwritten. Everything below the notebook replay is additive.

Run (from submissions/team_27, after frozen_lm_features.py has written lm_features/gpt2):
    python v6_report_analysis.py
On macOS the LightGBM wheel needs an OpenMP runtime; if `import lightgbm` fails with
"Library not loaded: libomp.dylib", either `brew install libomp` or point DYLD_LIBRARY_PATH at
scikit-learn's bundled copy (site-packages/sklearn/.dylibs).

Outputs go to v6_results/report/ (CSV + PNG + JSON).
"""
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # notebook cell 8 calls plt.show(); keep it headless here.

NB = Path("svm_frozen_lm_yx752_v6.ipynb")
REPLAY_CELLS = [1, 3, 4, 6, 8]  # Steps 0-3 only; cell 10 (full refit + submission) is skipped.

# ---------------------------------------------------------------------------
# 1. Replay the v6 notebook (unchanged code) and time each step.
# ---------------------------------------------------------------------------
g = {"__name__": "v6_replay"}
cells = json.loads(NB.read_text())["cells"]
step_seconds = {}
for i in REPLAY_CELLS:
    t0 = time.perf_counter()
    exec("".join(cells[i]["source"]), g)
    step_seconds[f"cell_{i}"] = time.perf_counter() - t0
    print(f"[replay] cell {i} done in {step_seconds[f'cell_{i}']:.1f}s", flush=True)

np, pd, plt = g["np"], g["pd"], g["plt"]
roc_auc_score, StratifiedKFold = g["roc_auc_score"], g["StratifiedKFold"]
LogisticRegression = g["LogisticRegression"]
SEED, train, tr, va, y, yv = g["SEED"], g["train"], g["tr"], g["va"], g["y"], g["yv"]
z, zf, lm = g["z"], g["zf"], g["lm"]
chosen, weight = g["chosen"], g["weight"]
fit_base, base_mappings, base_logits = g["fit_base"], g["base_mappings"], g["base_logits"]
new_model, new_score = g["new_model"], g["new_score"]
expanded_features, canonicalize = g["expanded_features"], g["canonicalize"]
DATA = g["DATA"]

OUT = Path("v6_results/report")
OUT.mkdir(parents=True, exist_ok=True)

# EDIT (report analysis only): pin the LightGBM branch to the configuration Yang's recorded v6 run
# selected (candidate 4: 31 leaves, 400 trees, lr 0.03, L2 2; LM weight 0.75). LightGBM's
# row/column subsampling gives OOF AUROCs that differ by ~1e-5 across machines, and on this
# machine (macOS, LightGBM 4.6.0) the blend search picked candidate 5 (15 leaves, 800 trees)
# instead. Candidates 3-5 are within 0.00003 OOF AUROC of each other, but the report must
# describe the submitted model, so the analysis below uses Yang's recorded choice.
RECORDED_CANDIDATE, RECORDED_WEIGHT, RECORDED_DEV_AUROC = 4, 0.75, 0.999590
if (g["chosen_id"], g["weight"]) != (RECORDED_CANDIDATE, RECORDED_WEIGHT):
    print(f"[note] replay selected candidate {g['chosen_id']} (weight {g['weight']}); "
          f"pinning to recorded candidate {RECORDED_CANDIDATE} (weight {RECORDED_WEIGHT})", flush=True)
chosen, weight = g["configs"][RECORDED_CANDIDATE], RECORDED_WEIGHT
assert chosen["view"] == "combined"  # the LM branch input is [23 GPT-2 stats | 73 style features]


# ---------------------------------------------------------------------------
# 2. The v6 training procedure as one function on arrays.
#    This mirrors notebook steps 1-3 line for line (same models, the same 0.75/0.25 original vs
#    flat-format sample weights, same 3-fold OOF calibration, same fixed blend weight), but takes
#    feature arrays instead of global indices so it can be re-run on subsets (learning curve) and
#    on training rows + pseudo-labelled unlabeled rows (Q3). Hyperparameters are NOT re-searched.
# ---------------------------------------------------------------------------
def fit_v6(Z, ZF, LM, labels):
    combined = np.column_stack([LM, Z])
    # (a) v5 baseline OOF scores on original and flat views -> component maps + meta map
    raw = np.zeros((2, len(labels), 2))
    folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(Z, labels))
    for a, b in folds:
        for j in [0, 1]:
            m = fit_base(j, Z[a], ZF[a], labels[a])
            for v, M in enumerate([Z, ZF]):
                raw[v, b, j] = m.decision_function(M[b])
    maps, meta = base_mappings(raw, labels)
    base_oof_logit = base_logits(raw, maps, meta)[0]
    # (b) frozen-LM LightGBM branch OOF -> its own logistic map
    new_oof = np.zeros(len(labels))
    for a, b in folds:
        new_oof[b] = new_score(new_model(chosen).fit(combined[a], labels[a]), combined[b])
    new_map = LogisticRegression(C=10, max_iter=1000, random_state=SEED).fit(new_oof.reshape(-1, 1), labels)
    blended = (1 - weight) * base_oof_logit + weight * new_map.decision_function(new_oof.reshape(-1, 1))
    final_meta = LogisticRegression(C=100, max_iter=1000, random_state=SEED).fit(blended.reshape(-1, 1), labels)
    # (c) refit every component on all supplied rows
    base_models = [fit_base(j, Z, ZF, labels) for j in [0, 1]]
    lm_model = new_model(chosen).fit(combined, labels)

    def predict(Zq, LMq):
        r = np.zeros((1, len(Zq), 2))
        for j in [0, 1]:
            r[0, :, j] = base_models[j].decision_function(Zq)
        b = base_logits(r, maps, meta)[0]
        n = new_map.decision_function(new_score(lm_model, np.column_stack([LMq, Zq])).reshape(-1, 1))
        s = (1 - weight) * b + weight * n
        return s, final_meta.predict_proba(s.reshape(-1, 1))[:, 1]

    return predict


results = {"replay_seconds": step_seconds}

# Sanity check: the re-implementation on the 8,000 training rows must reproduce the development
# AUROC of Yang's recorded run (0.999590), otherwise nothing below is about the real v6 model.
# Tolerance 5e-5 allows only the cross-machine LightGBM subsampling noise described above.
# (A first run without the pin matched the replayed notebook cell exactly, 0.999595 both ways,
# so fit_v6 itself is a faithful copy of the notebook procedure.)
t0 = time.perf_counter()
v6 = fit_v6(z[tr], zf[tr], lm[tr], y)
fit_seconds = time.perf_counter() - t0
dev_score, dev_prob = v6(z[va], lm[va])
dev_auc = roc_auc_score(yv, dev_score)
print(f"dev AUROC re-implementation {dev_auc:.6f} vs recorded v6 run {RECORDED_DEV_AUROC:.6f}")
assert abs(dev_auc - RECORDED_DEV_AUROC) < 5e-5
train_score, train_prob = v6(z[tr], lm[tr])  # in-sample: components were fit on these rows
results["final_model"] = {
    "train_auroc": roc_auc_score(y, train_score),
    "dev_auroc": dev_auc,
    "dev_auroc_probability": roc_auc_score(yv, dev_prob),
    "train_accuracy@0.5": float(((train_prob >= 0.5) == y).mean()),
    "dev_accuracy@0.5": float(((dev_prob >= 0.5) == yv).mean()),
    "fit_seconds_8000_rows_incl_oof_calibration": fit_seconds,
}
print(results["final_model"], flush=True)

# ---------------------------------------------------------------------------
# 3. Learning curve (Q2): nested, class-balanced subsets of the 8,000 training rows,
#    same 2,000 development rows, same fixed hyperparameters and blend weight.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
order_by_class = {c: rng.permutation(np.where(y == c)[0]) for c in [0, 1]}
curve = []
for n in [500, 1000, 2000, 5000, 8000]:
    sub = np.sort(np.concatenate([order_by_class[c][: n // 2] for c in [0, 1]]))  # nested by construction
    t0 = time.perf_counter()
    pred = v6 if n == 8000 else fit_v6(z[tr][sub], zf[tr][sub], lm[tr][sub], y[sub])
    secs = fit_seconds if n == 8000 else time.perf_counter() - t0
    s_tr, _ = pred(z[tr][sub], lm[tr][sub])
    s_va, _ = pred(z[va], lm[va])
    curve.append({"rows": n, "train_auroc": roc_auc_score(y[sub], s_tr), "dev_auroc": roc_auc_score(yv, s_va), "seconds": secs})
    print(curve[-1], flush=True)
curve = pd.DataFrame(curve)
curve.to_csv(OUT / "learning_curve.csv", index=False)
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(curve.rows, curve.train_auroc, "o-", label="Training AUROC")
ax.plot(curve.rows, curve.dev_auroc, "s-", label="Validation AUROC")
ax.set_xscale("log")
ax.set_xticks(curve.rows)
ax.set_xticklabels([f"{r:,}" for r in curve.rows])
ax.set_xlabel("Training rows (log scale)")
ax.set_ylabel("AUROC")
ax.set_title("Final model (v6): AUROC vs training-set size")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "learning_curve.png", dpi=200)

# ---------------------------------------------------------------------------
# 4. Length bins (Q4.2): whitespace word count, same bins as the earlier SVM analysis.
# ---------------------------------------------------------------------------
dev = train.iloc[va][["id", "text", "label"]].copy()
dev["p_ai"] = dev_prob
dev["words"] = dev.text.str.split().str.len()
dev["bin"] = pd.cut(dev.words, [0, 150, 250, np.inf], labels=["<=150", "151-250", ">250"])
bins = dev.groupby("bin", observed=True).apply(
    lambda d: pd.Series({"count": len(d), "ai_share": d.label.mean(), "auroc": roc_auc_score(d.label, d.p_ai),
                         "errors@0.5": int(((d.p_ai >= 0.5) != d.label).sum())})
).reset_index()
bins.to_csv(OUT / "length_bins.csv", index=False)
print(bins, flush=True)

# ---------------------------------------------------------------------------
# 5. Errors (Q4.3/4.4) at threshold 0.5, most confident first. Text is saved for manual review.
# ---------------------------------------------------------------------------
fp = dev[(dev.label == 0) & (dev.p_ai >= 0.5)].sort_values("p_ai", ascending=False)
fn = dev[(dev.label == 1) & (dev.p_ai < 0.5)].sort_values("p_ai")
# If fewer than five errors exist at 0.5, also list the closest correct calls so five per type can be read.
near_fp = dev[(dev.label == 0) & (dev.p_ai < 0.5)].sort_values("p_ai", ascending=False).head(max(0, 5 - len(fp)))
near_fn = dev[(dev.label == 1) & (dev.p_ai >= 0.5)].sort_values("p_ai").head(max(0, 5 - len(fn)))
err = pd.concat([fp.assign(type="FP"), fn.assign(type="FN"),
                 near_fp.assign(type="human, highest P(AI) below 0.5"), near_fn.assign(type="AI, lowest P(AI) above 0.5")])
err[["id", "type", "p_ai", "words", "text"]].to_csv(OUT / "errors.csv", index=False)
results["errors@0.5"] = {"false_positives": len(fp), "false_negatives": len(fn)}
print(results["errors@0.5"], flush=True)

# ---------------------------------------------------------------------------
# 6. Unlabeled data (Q3): one round of confidence-based pseudo-labelling.
#    The supervised v6 (fit on the 8,000 training rows only) scores unlabeled.csv; passages with
#    P(AI) >= 0.99 or <= 0.01 (thresholds fixed before looking at dev results) are added with those
#    labels, and the whole v6 procedure is re-fit on training + pseudo rows. Development rows and
#    test.csv are never used for fitting, so no validation labels leak in.
# ---------------------------------------------------------------------------
unl = pd.read_csv(DATA / "unlabeled.csv")
lm_u = g["load_lm"]("unlabeled.csv", unl)
z_u = pd.DataFrame([expanded_features(str(t)) for t in unl.text]).to_numpy()
zf_u = pd.DataFrame([expanded_features(canonicalize(t)) for t in unl.text]).to_numpy()
_, p_u = v6(z_u, lm_u)
keep = (p_u >= 0.99) | (p_u <= 0.01)
pseudo = (p_u[keep] >= 0.5).astype(int)
t0 = time.perf_counter()
v6_pl = fit_v6(np.vstack([z[tr], z_u[keep]]), np.vstack([zf[tr], zf_u[keep]]), np.vstack([lm[tr], lm_u[keep]]),
               np.concatenate([y, pseudo]))
pl_secs = time.perf_counter() - t0
s_pl, _ = v6_pl(z[va], lm[va])
results["unlabeled"] = {
    "unlabeled_rows": len(unl), "pseudo_labelled": int(keep.sum()), "pseudo_ai": int(pseudo.sum()),
    "pseudo_human": int((1 - pseudo).sum()), "supervised_dev_auroc": dev_auc,
    "pseudo_label_dev_auroc": roc_auc_score(yv, s_pl), "pseudo_fit_seconds": pl_secs,
    "unlabeled_p_ai_mean": float(p_u.mean()),
}
print(results["unlabeled"], flush=True)

results["learning_curve"] = curve.to_dict(orient="records")
results["length_bins"] = bins.astype({"bin": str}).to_dict(orient="records")
(OUT / "summary.json").write_text(json.dumps(results, indent=2, default=float))
print("Saved", OUT)
