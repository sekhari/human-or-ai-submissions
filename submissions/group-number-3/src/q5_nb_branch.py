"""Q5: change only the 0.1-weight lexical branch of the Q1 five-model ensemble.

Run from submissions/group-number-3 after installing requirements.txt:
    python src/q5_nb_branch.py

Needs official data/train.csv, committed splits/splits.csv, and the four saved
results/val_probs_<model>.csv files. Generate missing validation predictions:
    python src/finetune_transformer.py --mode val --model distilroberta-base
    python src/finetune_transformer.py --mode val --model roberta-base --max_len 256 --batch 8
    python src/perplexity.py --mode val
    python src/perplexity2.py --mode val

The perplexity scripts use the committed feature caches. Outputs, including the
figure, stay in ignored results/q5_nb_branch/. No test prediction or submission.
"""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from time import perf_counter

from data import ROOT, get_split, load_splits, load_train

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".person_b_cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from models import tfidf_logreg
from nb_features import nb_weighted_logreg
from person_b import validate_split_data

COMPONENTS = ["distilroberta-base", "roberta-base", "ppl2_gbm", "ppl_gbm"]
WEIGHTS = [0.1, 0.5, 0.1, 0.2, 0.1]
CONFIGURATIONS = ["q1_five_model", "five_model_nb_branch"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_predictions(directory, validation):
    columns, hashes = [], {}
    for name in COMPONENTS:
        path = directory / f"val_probs_{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path.name}; see this script's validation commands.")
        pred = pd.read_csv(path).rename(columns={"p_ai": "p"})
        if not {"id", "label", "p"}.issubset(pred.columns):
            raise ValueError(f"{path.name}: require id,label,p columns.")
        if not pred.id.is_unique or len(pred) != len(validation) or set(pred.id) != set(validation.id):
            raise ValueError(f"{path.name}: IDs must match the fixed validation fold exactly.")
        aligned = validation[["id", "label"]].merge(pred, on="id", validate="one_to_one")
        if not aligned.label_x.eq(aligned.label_y).all():
            raise ValueError(f"{path.name}: labels disagree with official training labels.")
        if not np.isfinite(aligned.p).all() or not aligned.p.between(0, 1).all():
            raise ValueError(f"{path.name}: invalid probabilities.")
        columns.append(aligned.p.to_numpy())
        hashes[path.name] = digest(path)
    return np.column_stack(columns), hashes


def rank_average(probabilities):
    return sum(weight * (rankdata(probabilities[:, j]) / len(probabilities))
               for j, weight in enumerate(WEIGHTS))


def rank_auc(y, scores):
    n_positive = int(y.sum())
    n_negative = len(y) - n_positive
    return float((rankdata(scores)[y == 1].sum() - n_positive * (n_positive + 1) / 2)
                 / (n_positive * n_negative))


def paired_bootstrap(y, scores, repeats, seed):
    rng = np.random.default_rng(seed)
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    values = np.empty((repeats, 2))
    for i in range(repeats):
        indices = np.r_[rng.choice(positive, len(positive), replace=True),
                        rng.choice(negative, len(negative), replace=True)]
        for j, name in enumerate(CONFIGURATIONS):
            values[i, j] = rank_auc(y[indices], scores[name][indices])
    return values


def plot_results(metrics, destination):
    fig, ax = plt.subplots(figsize=(8, 5.1))
    fig.subplots_adjust(left=.12, right=.97, bottom=.27, top=.86)
    for j, row in enumerate(metrics):
        point = row["ensemble_val_auroc"]
        ax.errorbar(j, point, yerr=[[point - row["auc_ci95_low"]],
                                  [row["auc_ci95_high"] - point]],
                    fmt="o", capsize=5, markersize=9, color=["#2764a2", "#d86a22"][j])
        ax.annotate(f"{point:.7f}", (j, point), xytext=(13, 0),
                    textcoords="offset points", va="center", fontsize=12)
    ax.set_xticks([0, 1], ["Q1 five-model ensemble\nTF-IDF lexical branch (0.1)",
                          "Same five-model ensemble\nNB-weighted lexical branch (0.1)"])
    ax.set_xlim(-.5, 1.75)
    ax.set_ylim(min(row["auc_ci95_low"] for row in metrics) - .0007,
                max(row["auc_ci95_high"] for row in metrics) + .0007)
    ax.set_ylabel("Validation AUROC (zoomed scale)")
    ax.set_title("Changing only the lexical representation inside Q1")
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.5, .045, "Same 2,000 validation passages, other four components and all five weights.\n"
             "Conditional 95% bootstrap intervals; validation has been reused for model selection.",
             ha="center", va="bottom", fontsize=9)
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/q5_nb_branch")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.bootstrap < 100 or args.threads < 1:
        parser.error("Use at least 100 bootstrap resamples and one thread.")
    expected = json.loads((ROOT / "config/person_b.json").read_text())["expected_sha256"]
    input_hashes = {name: digest(ROOT / name) for name in ("data/train.csv", "splits/splits.csv")}
    if any(value != expected[name] for name, value in input_hashes.items()):
        raise ValueError("Official training data or committed split has changed.")
    train, splits = load_train(), load_splits()
    validate_split_data(train, splits)
    tr, va = get_split(train, splits, "random_fold")
    va = va.sort_values("id").reset_index(drop=True)
    if (len(tr), len(va)) != (8000, 2000) or va.label.value_counts().to_dict() != {0: 1000, 1: 1000}:
        raise ValueError("Expected the fixed stratified 8,000/2,000 split.")
    fixed, prediction_hashes = load_predictions(args.pred_dir, va)
    output = args.out_dir.resolve()
    if output in {args.pred_dir.resolve(), (ROOT / "data").resolve(), (ROOT / "splits").resolve()}:
        raise ValueError("Use a separate output directory.")
    output.mkdir(parents=True, exist_ok=True)
    branch_rows, lexical = [], {}
    for fold in ("random_fold", "ood_fold"):
        fit_rows, eval_rows = get_split(train, splits, fold)
        eval_rows = eval_rows.sort_values("id").reset_index(drop=True)
        for name, factory in (("tfidf_lr", tfidf_logreg), ("nb_lr", nb_weighted_logreg)):
            model = factory()
            with threadpool_limits(limits=args.threads):
                start = perf_counter()
                model.fit(fit_rows.text, fit_rows.label)
                fit_seconds = perf_counter() - start
                start = perf_counter()
                probability = model.predict_proba(eval_rows.text)[:, 1]
                prediction_seconds = perf_counter() - start
                train_auc = roc_auc_score(fit_rows.label, model.predict_proba(fit_rows.text)[:, 1])
            row = dict(fold=fold, branch=name, training_rows=len(fit_rows), validation_rows=len(eval_rows),
                       train_auroc=float(train_auc), val_auroc=float(roc_auc_score(eval_rows.label, probability)),
                       fit_seconds=fit_seconds, validation_prediction_seconds=prediction_seconds)
            branch_rows.append(row)
            eval_rows[["id", "label"]].assign(p=probability).to_csv(output / f"{fold}_{name}.csv", index=False)
            if fold == "random_fold":
                lexical[name] = probability
            print(json.dumps(row), flush=True)
    y = va.label.to_numpy()
    scores = {name: rank_average(np.column_stack([fixed, lexical[branch]]))
              for name, branch in zip(CONFIGURATIONS, ("tfidf_lr", "nb_lr"))}
    points = [float(roc_auc_score(y, scores[name])) for name in CONFIGURATIONS]
    if any(abs(rank_auc(y, scores[name]) - point) > 1e-12 for name, point in zip(CONFIGURATIONS, points)):
        raise ValueError("Independent AUROC formula disagrees.")
    bootstrap = paired_bootstrap(y, scores, args.bootstrap, args.seed)
    intervals = np.quantile(bootstrap, [.025, .975], axis=0)
    gains = bootstrap[:, 1] - bootstrap[:, 0]
    metrics = []
    for j, (name, branch) in enumerate(zip(CONFIGURATIONS, ("tfidf_lr", "nb_lr"))):
        branch_row = next(row for row in branch_rows if row["fold"] == "random_fold" and row["branch"] == branch)
        metrics.append(dict(configuration=name, ensemble_val_auroc=points[j], delta_vs_q1=points[j] - points[0],
                            auc_ci95_low=float(intervals[0, j]), auc_ci95_high=float(intervals[1, j]),
                            lexical_fit_and_prediction_seconds=branch_row["fit_seconds"] + branch_row["validation_prediction_seconds"]))
    for name, value in prediction_hashes.items():
        if digest(args.pred_dir / name) != value:
            raise ValueError("Input validation predictions changed during the experiment.")
    pd.DataFrame(branch_rows).to_csv(output / "branch_metrics.csv", index=False)
    pd.DataFrame(metrics).to_csv(output / "ensemble_metrics.csv", index=False)
    va[["id", "label"]].assign(**scores).to_csv(output / "validation_predictions.csv", index=False)
    pd.DataFrame(bootstrap, columns=CONFIGURATIONS).assign(gain=gains).to_csv(output / "paired_bootstrap.csv", index=False)
    summary = dict(metrics=metrics, branch_metrics=branch_rows, gain=points[1] - points[0],
                   paired_delta_ci95=np.quantile(gains, [.025, .975]).tolist(),
                   component_weights=dict(zip(COMPONENTS + ["lexical_branch"], WEIGHTS)),
                   input_sha256=input_hashes, prediction_sha256=prediction_hashes,
                   source_sha256={name: digest(Path(__file__).parent / name)
                                  for name in ("models.py", "nb_features.py", "q5_nb_branch.py")},
                   bootstrap_resamples=args.bootstrap, bootstrap_seed=args.seed,
                   python=platform.python_version(), sklearn=sklearn.__version__,
                   runtime_scope="Lexical branch fit and validation prediction only; other four models and LM feature extraction excluded.",
                   limitations="Exploratory reused validation fold. Conditional intervals exclude model/method selection and retraining variability. OOD scores describe the lexical branch only; no new Kaggle score.")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_results(metrics, output / "q5_nb_lexical_branch.png")
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)
    print("Paired conditional gain interval:", summary["paired_delta_ci95"], flush=True)


if __name__ == "__main__":
    main()
