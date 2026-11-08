"""Compare small MLPs with matched LR baselines and the existing gradient booster.

Run from the repository root:
    python -m experiments.neural_comparison --stage tune
    python -m experiments.neural_comparison --stage evaluate

Tuning sees only the 8,000-row training partition. Feature extraction is refitted
inside each fold; classifiers share those identical transformed arrays. Settings
are written to selected.json before the official validation set is evaluated.
No data is downloaded, test predictions generated, or existing results replaced.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import os
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from threadpoolctl import threadpool_limits

from src.features import build_features, strip_layout
from src.models import MODELS, build_tuned
from src.neural_models import SEED, candidates, classifier, fitted_pipeline, representation, text_column
from src.split import VAL_IDS_PATH, load_train_val

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "neural_comparison"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=list) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_record(model, X, y):
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, y)
    return time.perf_counter() - start, sorted({str(w.message) for w in caught})


def tune(args, out):
    train, _ = load_train_val()
    y = train.label.astype(int).to_numpy()
    folds = list(StratifiedKFold(args.folds, shuffle=True, random_state=SEED).split(train, y))
    fold_ids = train[["id", "label"]].copy()
    for f, (_, idx) in enumerate(folds):
        fold_ids.loc[idx, "fold"] = f
    fold_ids.to_csv(out / "training_cv_folds.csv", index=False)

    configs = candidates(args.profile)
    write_json(out / "search_plan.json", {
        "hypothesis": "Nonlinear combinations of word-topic and style signals can improve on LR and the existing gradient booster.",
        "representations": args.representations, "svd_components": args.components, "profile": args.profile,
        "folds": args.folds, "seed": SEED, "candidates": configs,
        "selection_metric": "mean training-partition CV AUROC, separately for each representation/classifier",
        "mlp_early_stopping": "Accuracy on an internal 15% of each fitting partition; never the official validation set.",
        "train_rows": len(train), "validation_ids_sha256": digest(VAL_IDS_PATH),
        "training_ids_sha256": hashlib.sha256(train.id.to_numpy().tobytes()).hexdigest(),
        "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in
                     ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "threadpoolctl", "tabulate")},
    })
    all_rows, baseline_rows = [], []
    settings = json.loads((ROOT / "results/model_comparison/best_params.json").read_text())
    total_start = time.perf_counter()
    for name in args.representations:
        for fold, (fit_idx, score_idx) in enumerate(folds):
            transform = representation(name, args.components)
            start = time.perf_counter()
            X_fit = transform.fit_transform(train.iloc[fit_idx])
            X_score = transform.transform(train.iloc[score_idx])
            feature_seconds = time.perf_counter() - start
            print(f"{name}: fold {fold + 1}/{args.folds}, {X_fit.shape[1]} features, preprocessing {feature_seconds:.1f}s", flush=True)
            for number, (kind, params) in enumerate(configs):
                model = classifier(kind, params)
                seconds, messages = fit_record(model, X_fit, y[fit_idx])
                auc = roc_auc_score(y[score_idx], model.predict_proba(X_score)[:, 1])
                all_rows.append({
                    "representation": name, "classifier": kind, "candidate": number,
                    "params": json.dumps(params), "fold": fold,
                    "cv_auroc": auc, "feature_fit_transform_s": feature_seconds,
                    "classifier_fit_s": seconds,
                    "n_iter": int(max(np.max(m.n_iter_) for m in model.estimators_)) if hasattr(model, "estimators_") else int(np.max(model.n_iter_)),
                    "warnings": json.dumps(messages),
                })
                pd.DataFrame(all_rows).to_csv(out / "cv_folds.csv", index=False)
                print(f"  {kind} {params}: CV AUROC {auc:.5f}, {seconds:.1f}s", flush=True)
            if name == "combined":
                spec = MODELS["style_hgb"]
                baseline = build_tuned("style_hgb", settings["style_hgb"])
                seconds, messages = fit_record(baseline, spec.inputs(train.iloc[fit_idx]), y[fit_idx])
                auc = roc_auc_score(y[score_idx], baseline.predict_proba(spec.inputs(train.iloc[score_idx]))[:, 1])
                baseline_rows.append({"fold": fold, "model": "style_hgb", "cv_auroc": auc,
                                      "fit_time_s": seconds, "warnings": json.dumps(messages)})
                pd.DataFrame(baseline_rows).to_csv(out / "baseline_cv.csv", index=False)
                print(f"  existing gradient booster: CV AUROC {auc:.5f}", flush=True)

    frame = pd.DataFrame(all_rows)
    table = frame.groupby(["representation", "classifier", "candidate", "params"], as_index=False).agg(
        mean_cv_auroc=("cv_auroc", "mean"), std_cv_auroc=("cv_auroc", "std"),
        mean_classifier_fit_s=("classifier_fit_s", "mean"),
    ).sort_values("mean_cv_auroc", ascending=False)
    table.to_csv(out / "tuning.csv", index=False)
    selected = table.groupby(["representation", "classifier"], sort=False).head(1)
    configs_out = []
    for row in selected.itertuples():
        configs_out.append({
            "name": f"{row.representation}_{row.classifier}",
            "representation": row.representation, "classifier": row.classifier,
            "params": json.loads(row.params), "mean_cv_auroc": row.mean_cv_auroc,
            "std_cv_auroc": row.std_cv_auroc,
        })
    write_json(out / "selected.json", {
        "models": configs_out, "components": args.components, "seed": SEED,
        "validation_ids_sha256": digest(VAL_IDS_PATH),
        "tuning_elapsed_s": time.perf_counter() - total_start,
    })
    print("\nSelected using training CV only:", flush=True)
    print(selected.to_string(index=False), flush=True)


def paired_ci(y, first, second, samples=2000):
    """Stratified paired bootstrap: identical examples for both models in each draw."""
    rng = np.random.default_rng(SEED)
    groups = [np.flatnonzero(y == k) for k in (0, 1)]
    deltas = []
    for _ in range(samples):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        deltas.append(roc_auc_score(y[idx], first[idx]) - roc_auc_score(y[idx], second[idx]))
    return np.percentile(deltas, [2.5, 97.5]).tolist()


def evaluate(args, out):
    if (out / "evaluation_metadata.json").exists():
        raise SystemExit("This run was already evaluated. Use a new output directory for a new experiment; preserve the existing evidence.")
    selected = json.loads((out / "selected.json").read_text())
    assert digest(VAL_IDS_PATH) == selected["validation_ids_sha256"], "Validation split changed."
    train, val = load_train_val()
    assert len(train) == 8000 and len(val) == 2000
    assert train.id.is_unique and val.id.is_unique and not set(train.id) & set(val.id)
    y, y_val = train.label.astype(int).to_numpy(), val.label.astype(int).to_numpy()
    stripped = val.copy()
    stripped["text"] = stripped.text.map(strip_layout)
    predictions = val[["id", "label"]].copy()
    stripped_predictions = val[["id", "label"]].copy()
    rows, losses, subgroup_rows = [], [], []
    settings = json.loads((ROOT / "results/model_comparison/best_params.json").read_text())
    models = []
    # Directly rerun the team's actual strongest model, not a weaker substitute.
    for name in ("style_hgb", "word_lr"):
        feature_transform = build_features if name == "style_hgb" else text_column
        baseline = Pipeline([("features", FunctionTransformer(feature_transform)),
                             ("classifier", build_tuned(name, settings[name]))])
        models.append((name, baseline, train, val, stripped,
                       {"source": "existing tuned baseline", "params": settings[name]}))
    for config in selected["models"]:
        models.append((config["name"], fitted_pipeline(
            config["representation"], config["classifier"], config["params"],
            selected["components"], selected["seed"]), train, val, stripped, config))

    for name, model, X_fit, X_val, X_strip, config in models:
        seconds, messages = fit_record(model, X_fit, y)
        p_train = model.predict_proba(X_fit)[:, 1]
        p_val = model.predict_proba(X_val)[:, 1]
        p_strip = model.predict_proba(X_strip)[:, 1]
        assert np.isfinite(p_val).all() and ((0 <= p_val) & (p_val <= 1)).all()
        predictions[name], stripped_predictions[name] = p_val, p_strip
        rows.append({
            "model": name, "train_auroc": roc_auc_score(y, p_train),
            "val_auroc": roc_auc_score(y_val, p_val),
            "layout_stripped_val_auroc": roc_auc_score(y_val, p_strip),
            "fit_time_s": seconds, "warnings": json.dumps(messages),
            "params": json.dumps(config),
        })
        estimator = model.named_steps["classifier"] if hasattr(model, "named_steps") and "classifier" in model.named_steps else model
        members = list(zip(estimator.estimators, estimator.estimators_)) if hasattr(estimator, "estimators_") and hasattr(estimator, "voting") else [(None, estimator)]
        for member_info, member in members:
            if hasattr(member, "loss_curve_"):
                member_name = f"{name}_{member_info[0]}" if member_info else name
                for epoch, loss in enumerate(member.loss_curve_, 1):
                    losses.append({"model": member_name, "epoch": epoch, "training_loss": loss,
                                   "internal_early_stopping_accuracy": member.validation_scores_[epoch - 1]})
        bins = pd.cut(val.text.str.split().str.len(), [0, 149, 250, float("inf")],
                      labels=["under_150_words", "150_to_250_words", "over_250_words"])
        for group in bins.cat.categories:
            mask = (bins == group).to_numpy()
            subgroup_rows.append({"model": name, "group": group, "count": int(mask.sum()),
                                  "human_count": int((y_val[mask] == 0).sum()),
                                  "ai_count": int((y_val[mask] == 1).sum()),
                                  "auroc": roc_auc_score(y_val[mask], p_val[mask]) if len(np.unique(y_val[mask])) == 2 else float("nan")})
        print(f"{name}: train {rows[-1]['train_auroc']:.5f}, validation {rows[-1]['val_auroc']:.5f}, formatting removed {rows[-1]['layout_stripped_val_auroc']:.5f} ({seconds:.1f}s)", flush=True)
        pd.DataFrame(rows).to_csv(out / "comparison.csv", index=False)
        predictions.to_csv(out / "val_predictions.csv", index=False)
        stripped_predictions.to_csv(out / "val_predictions_layout_stripped.csv", index=False)

    table = pd.DataFrame(rows).sort_values("val_auroc", ascending=False)
    table.to_csv(out / "comparison.csv", index=False)
    display = table.drop(columns=["warnings", "params"])
    (out / "comparison.md").write_text(display.round(5).to_markdown(index=False) + "\n")
    pd.DataFrame(losses).to_csv(out / "training_curves.csv", index=False)
    pd.DataFrame(subgroup_rows).to_csv(out / "length_subgroups.csv", index=False)

    contrasts = []
    for config in selected["models"]:
        if config["classifier"] != "mlp":
            continue
        name = config["name"]
        for reference in ("style_hgb", "word_lr", f"{config['representation']}_lr"):
            first, second = predictions[name].to_numpy(), predictions[reference].to_numpy()
            lo, hi = paired_ci(y_val, first, second)
            contrasts.append({"model": name, "reference": reference,
                              "val_auroc_difference": roc_auc_score(y_val, first) - roc_auc_score(y_val, second),
                              "paired_95ci_low": lo, "paired_95ci_high": hi})
    pd.DataFrame(contrasts).to_csv(out / "paired_comparisons.csv", index=False)
    write_json(out / "evaluation_metadata.json", {
        "selected_json_sha256": digest(out / "selected.json"),
        "validation_ids_sha256": digest(VAL_IDS_PATH),
        "note": "Bootstrap intervals condition on this fixed split and fitted models; they do not capture training or model-selection uncertainty. Formatting removal is a diagnostic, not the actual test distribution. No settings were selected on these validation metrics.",
    })

    os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "kaggle-neural-cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kaggle-neural-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(display.model, display.val_auroc, color=["#4c78a8" if "mlp" not in n else "#f58518" for n in display.model])
    axes[0].invert_yaxis()
    axes[0].set(xlim=(0.5, 1.0), xlabel="Validation AUROC", title="Original held-out passages")
    for i, score in enumerate(display.val_auroc):
        axes[0].text(score - 0.005, i, f"{score:.4f}", ha="right", va="center", color="white", fontsize=9)
    for name, frame in pd.DataFrame(losses).groupby("model"):
        axes[1].plot(frame.epoch, frame.training_loss, label=name)
    axes[1].set(xlabel="Training epoch", ylabel="Training loss", title="Neural-network fitting")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)
    print("\n" + display.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["tune", "evaluate", "all"], default="all")
    parser.add_argument("--representations", nargs="+", choices=["text", "style", "combined"], default=["text", "style", "combined"])
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--components", type=int, default=128)
    parser.add_argument("--profile", choices=["initial", "refined"], default="initial")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=args.threads):
        if args.stage in ("tune", "all"):
            if (args.output / "selected.json").exists():
                raise SystemExit("Tuning results already exist; use a new --output directory to preserve them.")
            tune(args, args.output)
        if args.stage in ("evaluate", "all"):
            evaluate(args, args.output)


if __name__ == "__main__":
    main()
