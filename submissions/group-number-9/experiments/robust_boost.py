"""Low-compute curvature boosting: fixed small grid, train-only robust selection.

Stages: prepare (text OOF only), select (after GPT-2 cache), evaluate, submit.
No unlabeled.csv use. No Kaggle upload. Existing experiments are preserved.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from experiments.stacking import BEST, text_scores
from src.features import build_extended_features, strip_layout
from src.gpt2_robust_features import text_hash
from src.models import build_tuned
from src.robust_boost import CANDIDATES, predict_candidate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "robust_boost"
CACHE = ROOT / ".cache" / "robust_boost"
MODEL_NAMES = ("word_lr", "masked_lr")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def load():
    # Explicit filenames: never load the additional unlabeled competition rows.
    all_train = pd.read_csv(ROOT / "Data" / "train.csv")
    val_ids = pd.read_csv(ROOT / "splits" / "val_ids.csv").id
    assignments = pd.read_csv(ROOT / "results" / "eda" / "cluster_assignments.csv")
    all_train = all_train.merge(assignments[["id", "cluster"]], on="id", validate="one_to_one")
    train = all_train[~all_train.id.isin(val_ids)].reset_index(drop=True)
    val = all_train[all_train.id.isin(val_ids)].reset_index(drop=True)
    assert len(train) == 8000 and len(val) == 2000
    assert set(train.id).isdisjoint(val.id)
    return train, val, all_train


def splits(train):
    inner = pd.read_csv(ROOT / "results" / "deberta_v3_mac_isolated" / "inner_split_ids.csv")
    is_fit = train.id.map(inner.set_index("id").partition).eq("inner_fit")
    assert is_fit.sum() == 6400 and (~is_fit).sum() == 1600
    folds = {"inner": (train[is_fit], train[~is_fit])}
    for name, topics in [("rare_topics", [0, 2]), ("news", [1]),
                         ("recipes", [3]), ("personal", [10])]:
        holdout = train.cluster.isin(topics)
        folds[name] = (train[~holdout], train[holdout])
    return folds


def protocol():
    OUT.mkdir(exist_ok=True, parents=True)
    CACHE.mkdir(exist_ok=True, parents=True)
    value = {
        "candidates": CANDIDATES,
        "development": "Five splits within the original 8000: existing 6400/1600 inner split; leave out topics 0+2, 1, 3, and 10 separately.",
        "selection": "Mean across five splits of .5 original AUROC + .25 original-no-blank AUROC + .25 flattened AUROC. Select best eligible candidate only if gain >= .00001 over curv_bag; otherwise retain curv_bag. Eligibility: original mean >= baseline-.0001, original minimum >= baseline-.001. Ties by candidate name.",
        "validation": "Score selected method and frozen curv_bag / curv controls only after selection. Fixed validation has been repeatedly viewed earlier; it is not a fresh test.",
        "export_gate": "Selected method differs from baseline, passes train-only selection gate, and fixed-validation AUROC >= reproduced curv_bag-.0001. A file is provisional; no guarantee of Kaggle improvement.",
        "topics": "Existing frozen EDA cluster assignments; the original unsupervised EDA used all competition text including test. No new topic model, test weighting or test-dependent selection. Topic weights use fitting rows only and do not use test/unlabeled frequencies.",
        "augmentation": "Original and genuinely flattened GPT-2/style features; shares .25/.5, paired original-ID bootstrap; total original mass preserved. All views of an ID stay within its original fold. Text OOF fits original texts, transformed scoring reused only after analyzer-equivalence check.",
        "inference": "Frozen GPT-2 125M FP32, first 512 tokens, same 19 statistics as origin/main src.neural; one pass for surprisal and curvature.",
        "no_unlabeled_data": True, "final_fit": "All 10000 labeled rows, text scores five-fold OOF. Test original text only.",
        "reference_remote_commit": "768828d6bfe9f3594333c7033ca4db44af94bd2a",
        "reference_public_auroc_user_reported": .98544,
        "reference_recorded_curv_bag_validation": .999775,
        "reference_recorded_other_stack_validation": .999884,
        "files": {p: sha(ROOT / p) for p in ["Data/train.csv", "splits/val_ids.csv",
                  "src/features.py", "src/models.py", "src/robust_boost.py",
                  "src/gpt2_robust_features.py", "experiments/robust_boost.py",
                  "results/model_comparison/best_params.json",
                  "results/eda/cluster_assignments.csv"]},
    }
    path = OUT / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError("Frozen protocol changed; create a new experiment")
    write(path, value)


def cached_text(key, fit, ev):
    """OOF fits and evaluation fits have distinct IDs; fingerprint prevents stale reuse."""
    path = CACHE / f"text_{key}.npz"
    fingerprint = text_hash(json.dumps({"fit": fit.id.tolist(), "eval": ev.id.tolist(),
                                        "params": {n: BEST[n] for n in MODEL_NAMES},
                                        "sources": {p: sha(ROOT / p) for p in
                                            ["src/models.py", "src/features.py", "Data/train.csv"]}}, sort_keys=True))
    if path.exists():
        d = np.load(path)
        assert str(d["fingerprint"]) == fingerprint
        return d["fit"], d["eval"]
    # Reuse validated original designs where identical; no external or shifted rows fit.
    previous = ROOT / ".cache" / "deberta_feature_boost" / ("inner_base.npz" if key == "inner" else "final_base.npz")
    use_previous = key in ("inner", "validation") and previous.exists()
    if use_previous:
        d = np.load(previous)
        assert np.array_equal(d["fit_ids"], fit.id)
        positions = pd.Index(d["eval_ids"]).get_indexer(ev.id)
        assert (positions >= 0).all()
        assert np.array_equal(d["y"], fit.label.astype(int))
        np.testing.assert_allclose(d["xf"][:, :-2], build_extended_features(fit), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(d["xe"][positions, :-2], build_extended_features(ev), rtol=1e-10, atol=1e-10)
        pf, pe = d["xf"][:, -2:], d["xe"][positions, -2:]
        print(f"Reused matching OOF text cache: {key}", flush=True)
    else:
        pf, pe = [], []
        for name in MODEL_NAMES:
            print(f"Fitting {key} five-fold OOF {name}: {len(fit)} rows", flush=True)
            f, e = text_scores(name, fit, ev)
            pf.append(f)
            pe.append(e)
        pf, pe = np.column_stack(pf), np.column_stack(pe)
    np.savez_compressed(path, fit=pf, eval=pe, fingerprint=fingerprint,
                        fit_ids=fit.id, eval_ids=ev.id)
    return pf, pe


def prepare():
    train, val, all_train = load()
    # Both text representations ignore the changed whitespace; assert it for every passage.
    for name in MODEL_NAMES:
        analyzer = build_tuned(name, BEST[name]).named_steps["tfidf"].build_analyzer()
        for text in all_train.text:
            assert analyzer(text) == analyzer(strip_layout(text)), f"Analyzer changed: {name}"
    write(OUT / "text_analyzer_check.json", {"rows": len(all_train), "models": MODEL_NAMES,
                                            "original_equals_flattened": True})
    for key, (fit, ev) in splits(train).items():
        cached_text(key, fit, ev)
    cached_text("validation", train, val)
    print("Text designs prepared", flush=True)


def designs(key, fit, ev):
    features = pd.read_csv(CACHE / "gpt2_features.csv").set_index("text_sha256")
    pf, pe = cached_text(key, fit, ev)
    matrices = []
    for frame, text in [(fit, pf), (ev, pe)]:
        for flattened in (False, True):
            view = frame.copy()
            if flattened:
                view["text"] = view.text.map(strip_layout)
            gpt = features.loc[view.text.map(text_hash)].to_numpy()
            matrix = np.column_stack([build_extended_features(view).to_numpy(), gpt, text])
            assert matrix.shape[1] == 79 and np.isfinite(matrix).all()
            matrices.append(matrix)
    return matrices


def metrics(frame, p, flat):
    y = frame.label.astype(int).to_numpy()
    keep = build_extended_features(frame).blank_lines.eq(0).to_numpy()
    no_blank = roc_auc_score(y[keep], p[keep]) if len(np.unique(y[keep])) == 2 else np.nan
    return {"original": roc_auc_score(y, p), "no_blank": no_blank,
            "flattened": roc_auc_score(y, flat)}


def select():
    if (OUT / "selection.json").exists():
        print("Selection already frozen", flush=True)
        return
    train, _, _ = load()
    rows = []
    for key, (fit, ev) in splits(train).items():
        x, flat, xe, xef = designs(key, fit, ev)
        for name, spec in CANDIDATES.items():
            t = time.monotonic()
            p, q = predict_candidate(spec, x, flat, fit.label.astype(int).to_numpy(),
                                     fit.cluster.to_numpy(), xe, xef)
            row = {"split": key, "method": name, **metrics(ev, p, q)}
            rows.append(row)
            print(json.dumps({**row, "seconds": round(time.monotonic() - t, 2)}), flush=True)
            pd.DataFrame({"id": ev.id, "label": ev.label, "probability": p,
                          "flattened_probability": q}).to_csv(CACHE / f"selection_{key}_{name}.csv", index=False)
        pd.DataFrame(rows).to_csv(OUT / "development_by_split.csv", index=False)
    per_split = pd.DataFrame(rows)
    per_split["robust_score"] = .5 * per_split.original + .25 * per_split.no_blank + .25 * per_split.flattened
    summary = per_split.groupby("method").agg(original_mean=("original", "mean"),
        original_min=("original", "min"), no_blank_mean=("no_blank", "mean"),
        flattened_mean=("flattened", "mean"), robust_score=("robust_score", "mean"))
    base = summary.loc["curv_bag"]
    summary["gain"] = summary.robust_score - base.robust_score
    summary["eligible"] = ((summary.original_mean >= base.original_mean - .0001) &
                           (summary.original_min >= base.original_min - .001))
    choices = summary[summary.eligible].sort_values(["robust_score", "method"], ascending=[False, True])
    winner = choices.index[0] if choices.iloc[0].gain >= .00001 else "curv_bag"
    summary.sort_values("robust_score", ascending=False).to_csv(OUT / "development_comparison.csv")
    write(OUT / "selection.json", {"method": winner, "train_only_gain": float(summary.loc[winner, "gain"]),
                                   "spec": CANDIDATES[winner], "validation_consulted": False,
                                   "protocol_sha256": sha(OUT / "protocol.json")})
    print(f"FROZEN SELECTION: {winner}", flush=True)


def paired_ci(y, p, reference, repetitions=2000):
    rng = np.random.default_rng(42)
    groups = [np.flatnonzero(y == cls) for cls in [0, 1]]
    values = []
    for _ in range(repetitions):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        values.append(roc_auc_score(y[idx], p[idx]) - roc_auc_score(y[idx], reference[idx]))
    return np.quantile(values, [.025, .975]).tolist()


def evaluate():
    if (OUT / "validation_summary.json").exists():
        print("Validation already scored", flush=True)
        return
    selection = json.loads((OUT / "selection.json").read_text())
    train, val, _ = load()
    x, flat, xe, xef = designs("validation", train, val)
    chosen = list(dict.fromkeys(["curv_bag", "curv", selection["method"]]))
    scores, rows = {}, []
    for name in chosen:
        p, q = predict_candidate(CANDIDATES[name], x, flat, train.label.astype(int).to_numpy(),
                                 train.cluster.to_numpy(), xe, xef)
        scores[name] = p
        rows.append({"method": name, **metrics(val, p, q), "errors_at_half": int(((p >= .5) != val.label).sum())})
        print(json.dumps(rows[-1]), flush=True)
        pd.DataFrame({"id": val.id, "label": val.label, "probability": p,
                      "flattened_probability": q}).to_csv(OUT / f"validation_{name}.csv", index=False)
    table = pd.DataFrame(rows).set_index("method")
    table.to_csv(OUT / "validation_comparison.csv")
    name = selection["method"]
    gain = float(table.loc[name, "original"] - table.loc["curv_bag", "original"])
    interval = paired_ci(val.label.astype(int).to_numpy(), scores[name], scores["curv_bag"])
    gate = name != "curv_bag" and gain >= -.0001
    summary = {"selected": name, "validation_gain_vs_curv_bag": gain,
               "paired_95_ci": interval, "export_gate_passed": gate,
               "reproduction_delta_vs_recorded_curv_bag": float(table.loc["curv_bag", "original"] - .999775),
               "beats_recorded_other_stack": bool(table.loc[name, "original"] > .999884),
               "limitation": "Paired bootstrap conditional on a repeatedly reused validation split. Generalization to Kaggle is unproven."}
    write(OUT / "validation_summary.json", summary)
    print(json.dumps(summary), flush=True)


def submit():
    summary = json.loads((OUT / "validation_summary.json").read_text())
    if not summary["export_gate_passed"]:
        print("Export gate failed; no new Kaggle candidate written", flush=True)
        return
    _, _, train = load()
    test = pd.read_csv(ROOT / "Data" / "test.csv")
    # designs expects both views. Test is scored only in its original view, so use
    # the original feature matrix twice, avoiding any test formatting experiments.
    pf, pe = cached_text("full_fit", train, test)
    gpt = pd.read_csv(CACHE / "gpt2_features.csv").set_index("text_sha256")
    def matrix(frame, text):
        return np.column_stack([build_extended_features(frame), gpt.loc[frame.text.map(text_hash)], text])
    x = matrix(train, pf)
    flat_frame = train.copy()
    flat_frame["text"] = train.text.map(strip_layout)
    flat = matrix(flat_frame, pf)
    xe = matrix(test, pe)
    name = summary["selected"]
    p, _ = predict_candidate(CANDIDATES[name], x, flat, train.label.astype(int).to_numpy(),
                             train.cluster.to_numpy(), xe, xe)
    sub = pd.DataFrame({"id": test.id, "label": p})
    sample = pd.read_csv(ROOT / "Data" / "sample_submission.csv")
    assert len(sub) == 3000 and sub.id.is_unique
    assert np.array_equal(sub.id, test.id) and np.array_equal(sub.id, sample.id)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    path = ROOT / "submissions" / f"robust_{name}_10000.csv"
    sub.to_csv(path, index=False)
    write(path.with_suffix(".json"), {"method": name, "training_rows": len(train),
           "csv_sha256": sha(path), "test_sha256": sha(ROOT / "Data" / "test.csv"),
           "protocol_sha256": sha(OUT / "protocol.json"), "selection": json.loads((OUT / "selection.json").read_text()),
           "validation": summary, "uploaded_to_kaggle": False, "unlabeled_used": False})
    print(f"Wrote {path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["prepare", "select", "evaluate", "submit"], required=True)
    args = ap.parse_args()
    protocol()
    with threadpool_limits(limits=2):
        globals()[args.stage]()


if __name__ == "__main__":
    main()
