"""Evaluate and refit Group 26's selected style-feature KNN classifier."""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import config
from course_models import (
    SEED,
    SIZES,
    default_data_dir,
    make_split,
    read_rows,
    split_rows,
    targets,
    write_report,
)
from features import FEATURES_SCALAR_ONLY
from style_features import FEATURES_WITHOUT_LAYOUT_PUNCTUATION, style_features


K = 15
GRID = (15, 35, 75)


def features(rows):
    return np.asarray([style_features(row["text"]) for row in rows])


def knn(k):
    return KNeighborsClassifier(n_neighbors=k, weights="uniform", metric="euclidean", n_jobs=1)


def evaluate(model, x, y, v, z, training_score=True):
    start = perf_counter()
    model.fit(x, y)
    fit_time = perf_counter() - start
    start = perf_counter()
    p = model.predict_proba(v)[:, 1]
    score_time = perf_counter() - start
    result = {"validation_auc": roc_auc_score(z, p), "fit_s": fit_time, "score_s": score_time}
    if training_score:
        # KNN's training score excludes the queried row from its neighbor set.
        query = None if isinstance(model, KNeighborsClassifier) else x
        result["train_auc"] = roc_auc_score(y, model.predict_proba(query)[:, 1])
    return result, p


def partner_scores(path, validation, column):
    rows = read_rows(path)
    by_id = {row["id"]: row for row in rows}
    if len(rows) != len(by_id) or set(by_id) != {row["id"] for row in validation}:
        raise ValueError(f"{path.name} must contain the exact validation IDs once each.")
    if any(int(by_id[row["id"]]["true_label"]) != int(row["label"]) for row in validation):
        raise ValueError(f"{path.name} contains labels that disagree with train.csv.")
    scores = np.array([float(by_id[row["id"]][column]) for row in validation])
    if not np.isfinite(scores).all():
        raise ValueError(f"{path.name} contains non-finite scores.")
    return scores


def run(train, validation, unlabeled, partners):
    y, z = targets(train), targets(validation)
    start = perf_counter()
    a, b = features(train), features(validation)
    scaler = StandardScaler().fit(a)
    x, v = scaler.transform(a), scaler.transform(b)
    metrics = {"feature_fit_s": perf_counter() - start, "knn_sweep": {}, "style_logistic_sweep": {}}
    metrics["training_feature_summary"] = {
        "sentence_length_sd_median": [float(np.median(a[y == label, 1])) for label in (0, 1)],
        "once_only_word_fraction_median": [float(np.median(a[y == label, 4])) for label in (0, 1)],
        "has_contraction_fraction": [float(np.mean(a[y == label, 5] > 0)) for label in (0, 1)],
    }
    predictions = {}
    for k in GRID:
        metrics["knn_sweep"][str(k)], predictions[f"knn_{k}"] = evaluate(knn(k), x, y, v, z)
    for c in (.1, 1, 10):
        model = LogisticRegression(C=c, solver="lbfgs", max_iter=1000)
        metrics["style_logistic_sweep"][str(c)], predictions[f"style_logistic_{c}"] = evaluate(model, x, y, v, z)
    baseline = predictions[f"knn_{K}"]
    selected = metrics["knn_sweep"][str(K)]
    metrics["selected"] = {"k": K, **selected}

    rng = np.random.default_rng(SEED)
    groups = [rng.permutation(np.flatnonzero(y == label)) for label in (0, 1)]
    metrics["learning_curve"] = []
    for size in SIZES:
        if size == len(train):
            record, p = selected, baseline
        else:
            ix = np.sort(np.concatenate([group[:size // 2] for group in groups]))
            scale = StandardScaler().fit(a[ix])
            record, p = evaluate(knn(K), scale.transform(a[ix]), y[ix], scale.transform(b), z)
        metrics["learning_curve"].append({"rows": size, **record})
        predictions[f"learning_{size}"] = p

    scale = StandardScaler().fit(np.vstack([a, features(unlabeled)]))
    metrics["extra_scaler"], predictions["extra_scaler"] = evaluate(
        knn(K), scale.transform(a), y, scale.transform(b), z, training_score=False
    )
    metrics["extra_scaler"]["unlabeled_rows"] = len(unlabeled)

    reduced = FEATURES_WITHOUT_LAYOUT_PUNCTUATION
    metrics["without_layout_punctuation"], predictions["without_layout_punctuation"] = evaluate(
        knn(K), x[:, reduced], y, v[:, reduced], z, training_score=False
    )
    metrics["feature_count"] = x.shape[1]
    metrics["reduced_feature_count"] = len(reduced)

    # Shift-robust variant (see improvement.md): the 10 scalar features only.
    # Standardization is per column, so slicing the 23-column scaled matrix is
    # identical to fitting a scaler on the 10 columns alone.
    scalar = FEATURES_SCALAR_ONLY
    metrics["scalar_features_knn"], predictions["scalar_features_knn"] = evaluate(
        knn(K), x[:, scalar], y, v[:, scalar], z, training_score=False
    )
    metrics["scalar_feature_count"] = len(scalar)

    lengths = np.array([len(row["text"].split()) for row in validation])
    metrics["length_bins"] = []
    for low, high in ((0, 150), (150, 250), (250, np.inf)):
        use = (lengths >= low) & (lengths < high)
        metrics["length_bins"].append([int(use.sum()), int(z[use].sum()), roc_auc_score(z[use], baseline[use])])
    metrics["errors"] = []
    metrics["thresholds"] = {}
    comparisons = {"knn": (baseline, .5)}
    for name, path, col, cutoff in partners:
        if path is not None:
            p = partner_scores(path, validation, col)
            predictions[name] = p
            comparisons[name] = (p, cutoff)
    knn_errors = (baseline >= .5) != z
    for name, (p, cutoff) in comparisons.items():
        pred = p >= cutoff
        mistakes = pred != z
        metrics["thresholds"][name] = {
            "validation_auc": roc_auc_score(z, p), "cutoff": cutoff,
            "fp": int((pred & (z == 0)).sum()), "fn": int((~pred & (z == 1)).sum()),
            "both_wrong_with_knn": int((mistakes & knn_errors).sum()),
            "only_knn_wrong": int((knn_errors & ~mistakes).sum()),
            "only_this_model_wrong": int((~knn_errors & mistakes).sum()),
        }
    for kind, mask in [("FP", (z == 0) & (baseline >= .5)), ("FN", (z == 1) & (baseline < .5))]:
        order = sorted(np.flatnonzero(mask), key=lambda i: (-abs(baseline[i] - .5), validation[i]["id"]))
        for i in order[:3] + order[-2:]:
            metrics["errors"].append({"id": validation[i]["id"], "type": kind,
                                      "score": float(baseline[i]), "style": b[i].tolist()})
    return metrics, predictions


def predict(train, test, output, columns=None):
    a, b = features(train), features(test)
    if columns is not None:
        a, b = a[:, columns], b[:, columns]
    scaler = StandardScaler().fit(a)
    model = knn(K).fit(scaler.transform(a), targets(train))
    scores = model.predict_proba(scaler.transform(b))[:, 1]
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "label"])
        writer.writerows((row["id"], score) for row, score in zip(test, scores))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=default_data_dir())
    parser.add_argument("--validation-ids", type=Path, default=config.VALIDATION_IDS)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--make-split", action="store_true")
    mode.add_argument("--out", type=Path, help="New directory for metrics and validation scores")
    mode.add_argument("--submission", type=Path, help="New path for a local Kaggle-format CSV")
    parser.add_argument("--partner-logistic", type=Path, help="Optional partner probability CSV")
    parser.add_argument("--partner-svm", type=Path, help="Optional partner decision-score CSV")
    parser.add_argument(
        "--feature-set", choices=("all", "scalar"), default="all",
        help="With --submission: 'all' = the 23 features (the reported model, "
             "default); 'scalar' = the 10 scalar features (shift-robust variant).")
    args = parser.parse_args()
    output = args.out or args.submission
    if output and output.exists():
        parser.error("The output path already exists. Choose a new path.")
    if (args.partner_logistic or args.partner_svm) and not args.out:
        parser.error("Partner scores are only used with --out.")
    if args.feature_set != "all" and not args.submission:
        parser.error("--feature-set only applies to --submission.")
    train = sorted(read_rows(args.data / "train.csv"), key=lambda row: row["id"])
    if len(train) != 10_000 or len({row["id"] for row in train}) != len(train):
        parser.error("Expected 10,000 uniquely identified labeled training rows.")
    with threadpool_limits(limits=1):
        if args.make_split:
            make_split(train, args.validation_ids)
        elif args.submission:
            columns = list(FEATURES_SCALAR_ONLY) if args.feature_set == "scalar" else None
            predict(train, read_rows(args.data / "test.csv"), args.submission, columns)
        else:
            train, validation = split_rows(train, args.validation_ids)
            partners = (("partner_logistic", args.partner_logistic, "pred_proba", .5),
                        ("partner_svm", args.partner_svm, "ai_score", 0.0))
            metrics, scores = run(train, validation, read_rows(args.data / "unlabeled.csv"), partners)
            write_report(args.out, validation, metrics, scores)
            print(f"Saved report metrics and {len(validation)} validation predictions to {args.out}.")


if __name__ == "__main__":
    main()
