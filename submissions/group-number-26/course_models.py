"""Run the slide-based text classifiers and controlled report experiments."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

# Aliased: this module already uses `config` as a local variable name inside
# main() and predict(), which would shadow a plain `import config`.
import config as group26_config


SEED = 20261023
SETTINGS = {"logistic": (.01, .1, 1.0), "knn": (15, 35, 75), "forest": (12, 24, None)}
SIZES = (500, 1000, 2000, 5000, 8000)


def default_data_dir():
    """Where to look for the competition CSVs when --data is not given.

    Resolves against the submission directory instead of the current working
    directory, so these scripts behave identically whether they are run from
    submissions/group-number-26/ or from the repository root. Falls back to
    group26_config.DATA_ROOT when the CSVs are absent, so --help and argument errors
    still work on a fresh clone with no data downloaded.

    This is the only change made while relocating Kevin's modules into the
    canonical directory; the model and experiment logic below is his, unedited.
    """
    try:
        return group26_config.find_data_dir()
    except FileNotFoundError:
        return group26_config.DATA_ROOT


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def targets(rows):
    return np.array([int(row["label"]) for row in rows])


def make_split(rows, path):
    _, validation = train_test_split(rows, test_size=2000, stratify=targets(rows), random_state=SEED)
    with path.open("x", encoding="utf-8") as out:
        out.writelines(f"{row_id}\n" for row_id in sorted(row["id"] for row in validation))


def split_rows(rows, path):
    ids = path.read_text(encoding="utf-8-sig").splitlines()
    holdout = set(ids)
    if len(ids) != 2000 or len(holdout) != 2000 or not holdout <= {row["id"] for row in rows}:
        raise ValueError("The split must contain exactly 2,000 distinct IDs from train.csv.")
    train = [row for row in rows if row["id"] not in holdout]
    validation = [row for row in rows if row["id"] in holdout]
    if np.bincount(targets(validation)).tolist() != [1000, 1000]:
        raise ValueError("The validation split must have 1,000 rows per label.")
    return train, validation


def counts(train, query):
    vectorizer = CountVectorizer(min_df=2, max_features=10_000)
    x = vectorizer.fit_transform(row["text"] for row in train)
    v = vectorizer.transform(row["text"] for row in query)
    return vectorizer, x, v


def idf(count_matrix, additional=None):
    # Session 8 defines unsmoothed log(N / df), without an offset.
    documents = count_matrix.shape[0]
    frequency = count_matrix.getnnz(axis=0)
    if additional is not None:
        documents += additional.shape[0]
        frequency += additional.getnnz(axis=0)
    return np.log(documents / frequency)


def tfidf(matrix, weights):
    return matrix.multiply(weights).tocsr()


def classifier(family, setting):
    if family == "logistic":
        return LogisticRegression(C=setting, solver="liblinear", max_iter=1000, random_state=SEED)
    if family == "knn":
        return KNeighborsClassifier(n_neighbors=setting, weights="uniform", metric="cosine", n_jobs=1)
    if family == "forest":
        return RandomForestClassifier(n_estimators=100, max_depth=setting, max_features="sqrt",
                                      random_state=SEED, n_jobs=1)
    raise ValueError(f"Unknown classifier: {family}")


def evaluate(family, setting, x, y, v, z, train_auc=True):
    start = perf_counter()
    model = classifier(family, setting).fit(x, y)
    fit = perf_counter() - start
    start = perf_counter()
    scores = model.predict_proba(v)[:, 1]
    elapsed = perf_counter() - start
    record = {"family": family, "setting": setting, "validation_auc": roc_auc_score(z, scores),
              "fit_s": fit, "score_s": elapsed}
    if train_auc:
        # None excludes self-neighbors for KNN. Other models use fitted training scores.
        record["train_auc"] = roc_auc_score(y, model.predict_proba(None if family == "knn" else x)[:, 1])
    return record, scores


def run(train, validation, unlabeled):
    start = perf_counter()
    vectorizer, train_counts, val_counts = counts(train, validation)
    weights = idf(train_counts)
    x, v = tfidf(train_counts, weights), tfidf(val_counts, weights)
    y, z = targets(train), targets(validation)
    metrics = {"vocabulary_size": x.shape[1], "feature_fit_s": perf_counter() - start, "sweep": []}
    scores = {}
    for family, options in SETTINGS.items():
        for setting in options:
            key = f"{family}_{setting}"
            record, scores[key] = evaluate(family, setting, x, y, v, z)
            record["key"] = key
            metrics["sweep"].append(record)
            print(f"{key}: validation AUROC {record['validation_auc']:.6f}", flush=True)

    selected = max(metrics["sweep"], key=lambda row: row["validation_auc"])
    metrics["selected"] = selected
    family, setting, key = selected["family"], selected["setting"], selected["key"]
    p = scores[key]

    rng = np.random.default_rng(SEED)
    classes = [rng.permutation(np.flatnonzero(y == label)) for label in (0, 1)]
    metrics["learning_curve"] = []
    for size in SIZES:
        if size == len(train):
            record, scores[f"learning_{size}"] = selected, p
        else:
            ix = np.sort(np.concatenate([group[:size // 2] for group in classes]))
            _, a, b = counts([train[i] for i in ix], validation)
            w = idf(a)
            record, scores[f"learning_{size}"] = evaluate(
                family, setting, tfidf(a, w), y[ix], tfidf(b, w), z
            )
        metrics["learning_curve"].append({"rows": size, "train_auc": record["train_auc"],
                                          "validation_auc": record["validation_auc"]})

    extra_counts = vectorizer.transform(row["text"] for row in unlabeled)
    w = idf(train_counts, extra_counts)
    metrics["extra_idf"], scores["extra_idf"] = evaluate(
        family, setting, tfidf(train_counts, w), y, tfidf(val_counts, w), z, train_auc=False
    )
    metrics["extra_idf"]["unlabeled_rows"] = len(unlabeled)
    metrics["raw_counts"], scores["raw_counts"] = evaluate(
        family, setting, train_counts, y, val_counts, z, train_auc=False
    )
    if family != "logistic":
        # Positive column scaling preserves tree split order in exact arithmetic.
        # A fixed global logistic model gives a comparison that can respond to the scale.
        best_logistic = max((r for r in metrics["sweep"] if r["family"] == "logistic"),
                            key=lambda r: r["validation_auc"])
        c = best_logistic["setting"]
        metrics["logistic_extra_idf"], scores["logistic_extra_idf"] = evaluate(
            "logistic", c, tfidf(train_counts, w), y, tfidf(val_counts, w), z, train_auc=False
        )
        metrics["logistic_raw_counts"], scores["logistic_raw_counts"] = evaluate(
            "logistic", c, train_counts, y, val_counts, z, train_auc=False
        )

    word_counts = np.array([len(row["text"].split()) for row in validation])
    metrics["length_bins"] = []
    for low, high in [(0, 150), (150, 250), (250, np.inf)]:
        ix = (word_counts >= low) & (word_counts < high)
        metrics["length_bins"].append([int(ix.sum()), int(z[ix].sum()), roc_auc_score(z[ix], p[ix])])
    metrics["errors"] = []
    for kind, mask in [("FP", (z == 0) & (p >= .5)), ("FN", (z == 1) & (p < .5))]:
        order = sorted(np.flatnonzero(mask), key=lambda i: (-abs(p[i] - .5), validation[i]["id"]))
        metrics[f"{kind}_count"] = len(order)
        for i in order[:3] + order[-2:]:
            metrics["errors"].append({"id": validation[i]["id"], "type": kind, "score": float(p[i])})
    return metrics, scores


def write_report(out, validation, metrics, scores):
    out.mkdir(parents=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (out / "validation_scores.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "label", *scores])
        writer.writerows([row["id"], row["label"], *(values[i] for values in scores.values())]
                         for i, row in enumerate(validation))


def predict(train, test, config, output):
    _, x, v = counts(train, test)
    weights = idf(x)
    model = classifier(config["family"], config["setting"]).fit(tfidf(x, weights), targets(train))
    scores = model.predict_proba(tfidf(v, weights))[:, 1]
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "label"])
        writer.writerows((row["id"], score) for row, score in zip(test, scores))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=default_data_dir())
    parser.add_argument("--validation-ids", type=Path, default=group26_config.VALIDATION_IDS)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--make-split", action="store_true")
    mode.add_argument("--out", type=Path, help="New directory for metrics and validation predictions")
    mode.add_argument("--submission", type=Path, help="New path for a local Kaggle-format CSV")
    parser.add_argument("--metrics", type=Path, help="Metrics JSON from the validation run")
    args = parser.parse_args()
    if args.submission and args.metrics is None:
        parser.error("--submission requires --metrics from the validation run")
    output = args.out or args.submission
    if output and output.exists():
        parser.error("The output path already exists. Choose a new path.")
    rows = sorted(read_rows(args.data / "train.csv"), key=lambda row: row["id"])
    if len(rows) != 10_000 or len({row["id"] for row in rows}) != len(rows):
        parser.error("Expected 10,000 uniquely identified labeled training rows.")
    with threadpool_limits(limits=1):
        if args.make_split:
            make_split(rows, args.validation_ids)
        elif args.submission:
            config = json.loads(args.metrics.read_text(encoding="utf-8"))["selected"]
            predict(rows, read_rows(args.data / "test.csv"), config, args.submission)
        else:
            train, validation = split_rows(rows, args.validation_ids)
            metrics, scores = run(train, validation, read_rows(args.data / "unlabeled.csv"))
            write_report(args.out, validation, metrics, scores)
            print(f"Saved report metrics and {len(validation)} predictions to {args.out}.")


if __name__ == "__main__":
    main()
