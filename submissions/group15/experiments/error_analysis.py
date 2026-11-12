"""Group 15 Q4: error analysis for the selected A + C rank ensemble.

The final score is EXACTLY (rankdata(A)/n + rankdata(C)/n)/2, with average
ranks for ties, as in ensemble.py. It is NOT a calibrated P(AI). Threshold
0.5 is a fixed diagnostic convention, not a validation-tuned operating point.
Q4.1's training AUROC/gap requires --refit: fit the development A/C models on
the same 8,000 rows, verify their validation predictions against the saved
ones, then score the training rows. Without it, the gap is explicitly missing.

CSV overrides require id plus score (or p_ai), optionally y_true. NPY files
prefer *_ids.npy companions. Without IDs, use the ACTUAL writer conventions:
A and ensemble.py use original train.csv validation row order; Model C uses
val_ids.txt line order. This fallback is recorded as an assumption, not proof.

Outputs: experiments/error_analysis_ac.{json,txt}, *_validation.csv,
*_model_comparison.csv, *_length_bins.csv, *_error_overlap.csv,
*_style_summary.csv, *_all_errors.csv, *_examples.csv, and three PNG figures.
Examples include original text for manual review; failure causes are not
invented from snippets. Validation was used for selection, so results are
descriptive development-set diagnostics, not an independent test estimate.
"""

from pathlib import Path
import argparse
import hashlib
import json
import re
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as TEAM_COMMON
from model_c_style import extract_style_features, load_csv, load_split, make_model

SEED = int(TEAM_COMMON.SEED)
MODELS = ("A", "C", "AC")
BIN_NAMES = ["short <=150", "medium 151-250", "long >250"]


def team_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rank_scores(scores):
    """Match ensemble.to_rank, including its treatment of ties."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("Ranking requires a nonempty finite one-dimensional vector")
    return rankdata(scores, method="average") / len(scores)


def ensemble_scores(a_scores, c_scores):
    if np.shape(a_scores) != np.shape(c_scores):
        raise ValueError("A and C score shapes differ")
    return (rank_scores(a_scores) + rank_scores(c_scores)) / 2.0


def read_predictions(path, target, fallback_ids, convention):
    """Attach source IDs before aligning scores to target's ID-file order."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; generate the development validation predictions first")
    if path.suffix.lower() == ".npy":
        values = np.load(path, allow_pickle=False)
        if values.shape != (len(target),):
            raise ValueError(f"{path}: expected shape {(len(target),)}, got {values.shape}")
        sidecar = path.with_name(path.stem + "_ids.npy")
        if sidecar.exists():
            ids = np.load(sidecar, allow_pickle=False)
            if ids.shape != values.shape:
                raise ValueError(f"{sidecar}: ID and prediction shapes differ")
            alignment = f"ID companion: {sidecar}"
        else:
            ids = np.asarray(fallback_ids)
            alignment = f"ASSUMED from writer code: {convention}; no ID companion"
            warnings.warn(f"{path.name}: {alignment}", stacklevel=2)
        pred = pd.DataFrame({"id": ids, "score": values})
    elif path.suffix.lower() == ".csv":
        pred = pd.read_csv(path, float_precision="round_trip")
        if "score" not in pred and "p_ai" in pred:
            pred = pred.rename(columns={"p_ai": "score"})
        alignment = "ID-keyed CSV"
    else:
        raise ValueError(f"{path}: use .npy or an ID-keyed .csv")
    if not {"id", "score"} <= set(pred.columns):
        raise ValueError(f"{path}: CSV needs id and score (or p_ai)")
    if pred.id.isna().any() or pred.id.duplicated().any():
        raise ValueError(f"{path}: IDs must be unique and nonmissing")
    if len(pred) != len(target) or set(pred.id) != set(target.id):
        raise ValueError(f"{path}: prediction IDs do not match the validation split")
    ordered = pred.set_index("id").loc[target.id.to_list()]
    if "y_true" in ordered and not np.array_equal(ordered.y_true, target.label):
        raise ValueError(f"{path}: saved labels disagree with train.csv")
    values = ordered.score.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not ((0 <= values) & (values <= 1)).all():
        raise ValueError(f"{path}: expected finite scores in [0,1]")
    return values, {"path": str(path), "sha256": sha256(path), "alignment": alignment}


def check_c_metadata(path, data_path, split_path):
    """Verify available provenance without requiring an uncommitted model."""
    if not path.exists():
        return {"status": "not available; model-fit provenance is not independently verified"}
    metadata = json.loads(path.read_text(encoding="utf-8"))
    expected_split = metadata.get("split", {}).get("val_id_sha256")
    expected_train = metadata.get("data_sha256", {}).get("train")
    if expected_split and expected_split != sha256(split_path):
        raise ValueError("Model C metadata: validation ID file changed; regenerate C and the A+C ensemble")
    if expected_train and expected_train != sha256(data_path / "train.csv"):
        raise ValueError("Model C metadata: train.csv changed; regenerate the saved development predictions")
    return {"status": "available hashes checked", "path": str(path),
            "sha256": sha256(path), "chosen": metadata.get("chosen", {}),
            "split_hash_checked": bool(expected_split), "train_hash_checked": bool(expected_train)}


def diagnostic_metrics(labels, scores, threshold):
    """AUROC/AP use scores; FP/FN and classification metrics use a fixed cut."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    if len(y) == 0:
        return {"count": 0, "human_count": 0, "ai_count": 0,
                **{k: None for k in ["auc", "average_precision", "accuracy", "precision",
                   "recall", "f1", "error_rate", "fpr", "fnr"]},
                "TN": 0, "FP": 0, "FN": 0, "TP": 0}
    pred = (s >= threshold).astype(int)
    tn, fp, fn, tp = map(int, confusion_matrix(y, pred, labels=[0, 1]).ravel())
    both = len(np.unique(y)) == 2
    return {
        "count": len(y), "human_count": tn + fp, "ai_count": fn + tp,
        "auc": float(roc_auc_score(y, s)) if both else None,
        "average_precision": float(average_precision_score(y, s)) if both else None,
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "error_rate": float((fp + fn) / len(y)),
        "fpr": fp / (tn + fp) if tn + fp else None,
        "fnr": fn / (fn + tp) if fn + tp else None,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


def refit_development(fit, val, saved_a, saved_c, metadata):
    """Refit only the 8,000 training rows; never use a full-data checkpoint."""
    from model_a import build_model
    chosen = metadata.get("chosen", {})
    model_a = build_model()  # selected A: case/punctuation, (1,2), C=10
    model_c = make_model(depth=chosen.get("max_depth", None),
                         leaf=chosen.get("min_samples_leaf", 2), seed=SEED)
    if chosen.get("n_estimators", 300) != 300:
        raise ValueError("C metadata differs from make_model's 300-tree setting")
    predictions = {}
    for name, model, saved in [("A", model_a, saved_a), ("C", model_c, saved_c)]:
        print(f"Refitting development Model {name} on {len(fit)} rows...", flush=True)
        model.fit(fit.text, fit.label)
        current = np.asarray(TEAM_COMMON.get_scores(model, val.text), dtype=float)
        if not np.allclose(current, saved, rtol=1e-7, atol=1e-9):
            delta = float(np.max(np.abs(current - saved)))
            raise ValueError(
                f"Refitted {name} does not reproduce saved validation scores "
                f"(max difference={delta:.6g}). Check settings, row order and package "
                "versions; training-gap/stress results cannot be attributed to the saved final model. "
                "Run without --refit to analyze saved validation scores only."
            )
        predictions[name] = TEAM_COMMON.get_scores(model, fit.text)
    rank_a, rank_c = rank_scores(predictions["A"]), rank_scores(predictions["C"])
    return {"A": rank_a, "C": rank_c, "AC": (rank_a + rank_c) / 2}, (model_a, model_c)


def layout_stress(val, models, baseline_scores, threshold):
    """Transform validation text only, hold both fitted models fixed, re-rank."""
    a_model, c_model = models
    rows = []
    transforms = {
        "collapse_blank_lines": lambda t: re.sub(r"\n(?:[ \t\r]*\n)+", "\n", t),
        "flatten_whitespace": lambda t: TEAM_COMMON.clean_text(t, mode="flatten"),
    }
    baseline_auc = float(roc_auc_score(val.label, baseline_scores))
    for name, transform in transforms.items():
        changed = val.text.map(transform)
        a_scores = TEAM_COMMON.get_scores(a_model, changed)
        c_scores = TEAM_COMMON.get_scores(c_model, changed)
        scores = ensemble_scores(a_scores, c_scores)
        rows.append({"operation": name, "changed_text_rows": int((changed != val.text).sum()),
                     "decision_flips": int(((scores >= threshold) !=
                                             (baseline_scores >= threshold)).sum()),
                     "auc_change": float(roc_auc_score(val.label, scores) - baseline_auc),
                     **diagnostic_metrics(val.label, scores, threshold)})
    return rows


def add_diagnostics(val, a_scores, c_scores, threshold):
    frame = val.copy()
    frame["a_raw_score"] = a_scores
    frame["c_raw_score"] = c_scores
    frame["score_A"] = rank_scores(a_scores)
    frame["score_C"] = rank_scores(c_scores)
    frame["score_AC"] = ensemble_scores(a_scores, c_scores)
    for name in MODELS:
        frame[f"prediction_{name}"] = (frame[f"score_{name}"] >= threshold).astype(int)
        frame[f"error_{name}"] = frame[f"prediction_{name}"] != frame.label
    frame["error_type"] = np.select(
        [(frame.label == 0) & frame.error_AC, (frame.label == 1) & frame.error_AC,
         (frame.label == 0)], ["FP", "FN", "TN"], default="TP")
    features = extract_style_features(frame.text)
    for name in ["word_count", "type_token_ratio", "repeated_trigram_fraction", "blank_line_fraction"]:
        frame[name] = features[name].to_numpy()
    frame["newline_count"] = frame.text.str.count("\n")
    frame["length_bin"] = pd.cut(frame.word_count, [-1, 150, 250, np.inf], labels=BIN_NAMES)
    frame["snippet"] = frame.text.str.replace(r"\s+", " ", regex=True).str.slice(0, 300)
    return frame


def overlap_table(frame):
    rows = []
    for wrong_a, wrong_c, label in [(False, False, "A correct / C correct"),
                                     (True, False, "A wrong / C correct"),
                                     (False, True, "A correct / C wrong"),
                                     (True, True, "A wrong / C wrong")]:
        g = frame[(frame.error_A == wrong_a) & (frame.error_C == wrong_c)]
        rows.append({"component_outcomes": label, "count": len(g),
                     "AC_correct": int((~g.error_AC).sum()), "AC_errors": int(g.error_AC.sum()),
                     "AC_FP": int(g.error_type.eq("FP").sum()), "AC_FN": int(g.error_type.eq("FN").sum())})
    return pd.DataFrame(rows)


def style_summary(frame):
    rows = []
    for name in ["TN", "FP", "FN", "TP"]:
        g = frame[frame.error_type == name]
        row = {"outcome": name, "count": len(g)}
        for col in ["word_count", "newline_count", "type_token_ratio", "repeated_trigram_fraction"]:
            row[f"mean_{col}"] = float(g[col].mean()) if len(g) else None
        row["blank_line_rate"] = float(g.blank_line_fraction.gt(0).mean()) if len(g) else None
        rows.append(row)
    return pd.DataFrame(rows)


def draw_figures(frame, comparison, bins, threshold, figures, prefix):
    colors = {"A": "#4477aa", "C": "#228833", "AC": "#aa3377"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in MODELS:
        fpr, tpr, _ = roc_curve(frame.label, frame[f"score_{name}"])
        auc = comparison.loc[comparison.model == name, "val_auc"].iloc[0]
        ax.plot(fpr, tpr, color=colors[name], label=f'{"A+C" if name == "AC" else name}: AUROC {auc:.5f}')
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Fixed validation: A, C and A+C")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(figures / f"{prefix}_roc.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    values = pd.to_numeric(bins.auc, errors="coerce").to_numpy(dtype=float)
    ax.bar(np.arange(len(bins)), values, color=colors["AC"])
    for i, row in bins.iterrows():
        value = values[i]
        text = f"{value:.4f}\nn={row['count']}" if np.isfinite(value) else f"AUC unavailable\nn={row['count']}"
        ax.text(i, min(value + .02, 1.06) if np.isfinite(value) else .04,
                text, ha="center", va="bottom", fontsize=9)
    ax.set(xticks=np.arange(len(bins)), xticklabels=bins.length_bin,
           ylim=(0, 1.16), ylabel="Validation AUROC", title="A+C by whitespace word count")
    fig.tight_layout(); fig.savefig(figures / f"{prefix}_length.png", dpi=180); plt.close(fig)

    cm = confusion_matrix(frame.label, frame.prediction_AC, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Human", "AI"], yticklabels=["Human", "AI"],
           xlabel="Predicted label", ylabel="True label", title=f"A+C rank score threshold {threshold:g}")
    fig.tight_layout(); fig.savefig(figures / f"{prefix}_confusion.png", dpi=180); plt.close(fig)


def report_text(summary, comparison, bins, overlap, styles, examples):
    show = lambda table: table.to_string(index=False, float_format=lambda x: f"{x:.6f}")
    brief = ["id", "label", "error_type", "score_A", "score_C", "score_AC", "word_count", "snippet"]
    gap_note = ("Training AUROC is unavailable. Rerun with --refit to fit only the 8,000 development "
                "rows and verify both models against the saved validation scores."
                if not summary["refit_verified"] else
                "Training AUROC is resubstitution performance of verified development models. "
                "Ranks are computed separately within train and validation, matching ensemble.py.")
    return (
        "Group 15 Q4: selected final model A+C\n"
        "Equal-weight normalized rank averaging, average ranks for ties. Scores are not probabilities.\n"
        f"Diagnostic threshold: {summary['threshold']}; 1=AI, 0=human; seed={SEED}\n"
        "A and C decisions in comparison tables use their normalized ranks at the same threshold.\n\n"
        "Q4.1 Train versus validation / comparison with components\n" + show(comparison) + "\n" + gap_note +
        "\nSmall holdout gaps do not establish robustness to unknown sources.\n\n"
        "Q4.2 Length groups (fixed before looking at errors)\n" + show(bins) +
        "\nAUC/AP are unavailable for an empty or single-class group. FPR/ FNR use the human/AI "
        "counts as denominators. Precision/recall/F1 use zero_division=0.\n\n"
        "Q4.3 Up to five false positives and five false negatives\n" + show(examples[brief]) +
        "\nFP: human classified as AI; FN: AI classified as human. Selected by the most extreme "
        "ensemble rank score in each direction, with ID tie-breaking. These are illustrative cases, "
        "not a representative sample or probability-based confidence estimates. Actual counts are "
        "used if fewer than five exist. Full text and review fields are in *_examples.csv.\n\n"
        "Q4.4 Complementarity and observable style patterns\n" + show(overlap) + "\n\n" + show(styles) +
        "\nCompare FP with TN (human) and FN with TP (AI). Review original passages before "
        "claiming causes such as recipe/list formatting, informal language, repetitive wording, "
        "or biography-like prose. Style averages are associations, not causal explanations. "
        "Domain/generator labels are unavailable; no domain accuracy is inferred from text.\n\n"
        "Layout sensitivity\n" + json.dumps(summary["layout_stress"], indent=2) +
        "\nStress tests hold both models fixed, transform validation text only, then re-rank each "
        "component over that transformed cohort. This is a layout-sensitivity proxy.\n\n"
        "Limitations\n" + "\n".join("- " + x for x in summary["limitations"]) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=TEAM_COMMON.DATA_DIR, help="Relative paths are resolved from the team folder")
    parser.add_argument("--val-ids", default=TEAM_COMMON.SPLIT_FILE)
    parser.add_argument("--a-val-preds", default="preds/model_a_val.npy")
    parser.add_argument("--c-val-preds", default="preds/model_c_val.npy")
    parser.add_argument("--ensemble-val-preds", default="preds/ensemble_ac_val.npy",
                        help="Check this saved ensemble against recomputed A+C if present")
    parser.add_argument("--threshold", type=float, default=0.5, help="Fixed rank-score diagnostic cutoff; no tuning")
    parser.add_argument("--output-prefix", default="error_analysis_ac")
    parser.add_argument("--refit", action="store_true", help="Verify development refits and compute train-validation gaps")
    parser.add_argument("--layout-stress", action="store_true", help="Requires --refit; stress-test the whole A+C pipeline")
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error("--threshold must be strictly between 0 and 1")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.output_prefix):
        parser.error("--output-prefix must be a simple filename prefix: letters, digits, underscores or hyphens")
    if args.layout_stress and not args.refit:
        parser.error("--layout-stress requires --refit to verify BOTH development models")

    data_path, split_path = team_path(args.data), team_path(args.val_ids)
    train = load_csv(data_path, "train.csv")
    if not {"text", "label"} <= set(train) or train.label.isna().any() or set(train.label.unique()) != {0, 1}:
        raise ValueError("train.csv must contain text and binary labels 0=human, 1=AI")
    if not train.text.map(lambda t: isinstance(t, str)).all():
        raise ValueError("train.csv text must be strings")
    fit, val, origin = load_split(train, split_path)
    # This is common.get_split's actual row order, even when cwd is elsewhere.
    # Do not call make_split: the real team split must never be reconstructed.
    csv_order_ids = train.loc[train.id.isin(val.id), "id"].to_numpy()
    sources = {}
    a_path, c_path = team_path(args.a_val_preds), team_path(args.c_val_preds)
    a_scores, sources["A"] = read_predictions(a_path, val, csv_order_ids, "Model A: train.csv validation row order")
    c_scores, sources["C"] = read_predictions(c_path, val, val.id.to_numpy(), "Model C: val_ids.txt line order")
    frame = add_diagnostics(val, a_scores, c_scores, args.threshold)
    ensemble_path = team_path(args.ensemble_val_preds)
    if ensemble_path.exists():
        saved, sources["AC"] = read_predictions(ensemble_path, val, csv_order_ids,
                                               "ensemble.py: train.csv validation row order")
        if not np.allclose(saved, frame.score_AC, rtol=0, atol=1e-12):
            raise ValueError("Saved ensemble_ac validation scores do not match rank_average(A,C). "
                             "Check stale files and A/C row-order alignment before rerunning ensemble.py --submit AC.")
        sources["AC"]["status"] = "matches recomputed equal-weight rank average"
    elif args.ensemble_val_preds != "preds/ensemble_ac_val.npy":
        raise FileNotFoundError(ensemble_path)
    else:
        sources["AC"] = {"status": "saved ensemble absent; computed exactly from saved A and C"}

    metadata = (check_c_metadata(ROOT / "experiments/model_c_metrics.json", data_path, split_path)
                if c_path.resolve() == (ROOT / "preds/model_c_val.npy").resolve()
                else {"status": "custom C predictions; default C metadata not applied"})
    train_scores, models = None, None
    if args.refit:
        train_scores, models = refit_development(fit, val, a_scores, c_scores, metadata)
    comparison_rows = []
    for name in MODELS:
        row = {"model": name, **diagnostic_metrics(frame.label, frame[f"score_{name}"], args.threshold)}
        row["val_auc"] = row.pop("auc")
        row["train_auc"] = float(roc_auc_score(fit.label, train_scores[name])) if train_scores is not None else None
        row["auc_gap"] = row["train_auc"] - row["val_auc"] if row["train_auc"] is not None else None
        comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows)
    bins = pd.DataFrame([{"length_bin": name, **diagnostic_metrics(
        frame.loc[frame.length_bin == name, "label"], frame.loc[frame.length_bin == name, "score_AC"],
        args.threshold)} for name in BIN_NAMES])
    overlap, styles = overlap_table(frame), style_summary(frame)
    errors = frame.loc[frame.error_AC].copy()
    errors["failure_hypothesis"] = "Manual source-text review required; no cause inferred automatically."
    errors["review_notes"] = ""
    fp = errors[errors.error_type == "FP"].sort_values(["score_AC", "id"], ascending=[False, True]).head(5)
    fn = errors[errors.error_type == "FN"].sort_values(["score_AC", "id"], ascending=[True, True]).head(5)
    examples = pd.concat([fp, fn], ignore_index=True)
    stress = layout_stress(frame, models, frame.score_AC.to_numpy(), args.threshold) if args.layout_stress else "not requested"
    final = comparison_rows[-1]
    summary = {
        "model": "A+C: equal-weight normalized rank average", "seed": SEED,
        "score_definition": "(rankdata(A, method='average')/n + rankdata(C, method='average')/n)/2",
        "score_is_probability": False, "threshold": args.threshold,
        "threshold_policy": "Fixed diagnostic cut; not chosen by maximizing validation classification metrics",
        "component_decision_policy": "Use normalized ranks for A and C at the same cutoff as AC",
        "train_rows": len(fit), "val_rows": len(val), "split": origin,
        "train_sha256": sha256(data_path / "train.csv"), "prediction_sources": sources,
        "model_c_metadata": metadata, "refit_verified": bool(args.refit),
        "train_auc": final["train_auc"], "val_auc": final["val_auc"], "auc_gap": final["auc_gap"],
        "TN": final["TN"], "FP": final["FP"], "FN": final["FN"], "TP": final["TP"],
        "A_C_spearman": (float(frame[["score_A", "score_C"]].corr(method="spearman").iloc[0, 1])
                         if frame.score_A.nunique() > 1 and frame.score_C.nunique() > 1 else None),
        "AC_rescues_A_errors": int((frame.error_A & ~frame.error_AC).sum()),
        "AC_harms_A_correct": int((~frame.error_A & frame.error_AC).sum()),
        "AC_rescues_C_errors": int((frame.error_C & ~frame.error_AC).sum()),
        "AC_harms_C_correct": int((~frame.error_C & frame.error_AC).sum()),
        "selected_FP": len(fp), "selected_FN": len(fn),
        "case_selection": "Up to 5 highest-score FP and 5 lowest-score FN, ties broken by ID",
        "layout_stress": stress,
        "limitations": [
            "The same validation set selected models/hyperparameters/ensemble; these diagnostics are not an independent final test.",
            "NPY arrays without ID sidecars rely on the documented writer-code order; hashes alone cannot prove row alignment or absence of training leakage.",
            "Rank scores depend on the cohort and are not calibrated probabilities; threshold errors are descriptive.",
            "Training AUROC is resubstitution AUROC when --refit is verified; otherwise it and the gap are unavailable.",
            "Length/style associations and illustrative extreme-score errors do not establish failure causes.",
            "No test labels, external domain/generator labels, or test/unlabeled passages are used.",
        ],
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": __import__("sklearn").__version__},
    }
    output, figures = ROOT / "experiments", ROOT / "figures"
    output.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    prefix = output / args.output_prefix
    for suffix, table in [("validation", frame.drop(columns="text")), ("model_comparison", comparison),
                          ("length_bins", bins), ("error_overlap", overlap), ("style_summary", styles),
                          ("all_errors", errors), ("examples", examples)]:
        table.to_csv(f"{prefix}_{suffix}.csv", index=False)
    Path(f"{prefix}.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    Path(f"{prefix}.txt").write_text(report_text(summary, comparison, bins, overlap, styles, examples), encoding="utf-8")
    draw_figures(frame, comparison, bins, args.threshold, figures, args.output_prefix)
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nA+C examples: {len(fp)} FP and {len(fn)} FN (up to five each).")
    print(f"Reports: {prefix}.txt and {prefix}.json; CSV tables alongside them.")
    print(f"Figures: {figures / (args.output_prefix + '_*.png')}")
    if not args.refit:
        print("Train AUROC/gap unavailable: add --refit to verify development models and compute them.")


if __name__ == "__main__":
    main()

