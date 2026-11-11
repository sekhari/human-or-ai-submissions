"""Test quote canonicalization as one controlled post-Q5 intervention."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from human_ai.check_predictions import check
from human_ai.data import ROOT, SPLIT_DIR, load_partitions, raw_path, read_csv, sha256
from . import run_q5 as q5
from .q5_sensitivity import CONDITIONS, apply_condition
from .quote_normalizer import QuoteNormalizer
from .run_q1 import clean_params, id_hash, predict_ai, score

HERE = Path(__file__).resolve().parent
REPRESENTATIONS = ("c60", "wc120")
EXTRA_SOURCES = ("experiments/supervised/q5_quotes.py", "experiments/supervised/q5_quote_plan.json",
                 "experiments/supervised/quote_normalizer.py", "experiments/supervised/q5_sensitivity.py",
                 "experiments/supervised/q5_quotes_job.py")


def read_json(path):
    return json.loads(Path(path).read_text())


def load_plan():
    return read_json(HERE / "q5_quote_plan.json")


def model_id(representation, normalize_quotes):
    return f"{representation}_{'norm' if normalize_quotes else 'raw'}"


def make_model(representation, normalize_quotes, feature_plan, calibrated=True):
    base = q5.make_model(representation, feature_plan, calibrated=False)
    if normalize_quotes:
        base = Pipeline([("quote_normalize", QuoteNormalizer()), *base.steps])
    if not calibrated:
        return base
    cv = StratifiedKFold(3, shuffle=True, random_state=feature_plan["seeds"]["calibration"])
    return CalibratedClassifierCV(base, method="sigmoid", cv=cv, ensemble=False, n_jobs=1)


def initialize(prefix, resume):
    if not prefix or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in prefix):
        raise ValueError("Invalid run prefix")
    plan = load_plan()
    fp = read_json(ROOT / plan["feature_plan"])
    base = ROOT / "results/supervised"
    reference = base / plan["reference_study"]
    preceding = base / plan["preceding_study"]
    if read_json(reference / "status.json")["stage"] != "complete" or read_json(preceding / "status.json")["stage"] != "complete":
        raise ValueError("Reference evidence is incomplete")
    for source, expected in read_json(reference / "provenance.json")["source_sha256"].items():
        if sha256(ROOT / source) != expected:
            raise ValueError(f"Changed Q5 implementation: {source}")
    guard_paths = [preceding / name for name in ("selection.json", "format_sensitivity.json", "cv_results.csv")]
    for rep in REPRESENTATIONS:
        folder = base / f"a_q5_20261108_v1_{rep}"
        for name, expected in read_json(folder / "completed.json")["artifact_sha256"].items():
            if sha256(folder / name) != expected:
                raise ValueError(f"Changed raw reference: {folder / name}")
        guard_paths += [folder / name for name in ("model.joblib", "validation_predictions.csv", "metrics.json")]
    incumbent_csv = ROOT / "submissions/a_q5_20261108_v1_wc120_full10000.csv"
    if incumbent_csv.exists():
        guard_paths.append(incumbent_csv)
    guards = {str(p.relative_to(ROOT)): sha256(p) for p in guard_paths}
    context = q5.provenance()
    context["source_sha256"].update({name: sha256(ROOT / name) for name in EXTRA_SOURCES})
    original_context = read_json(reference / "provenance.json")
    if any(context[k] != original_context[k] for k in ("python", "dependencies")):
        raise ValueError("Use the recorded Q5 environment")
    study = base / f"{prefix}_study"
    if study.exists():
        if not resume:
            raise FileExistsError("Choose a NEW prefix or resume identical settings")
        recorded = read_json(study / "provenance.json")
        for name, expected in recorded["source_sha256"].items():
            if sha256(ROOT / name) != expected:
                raise ValueError(f"Cannot resume changed source: {name}")
        if read_json(study / "q5_quote_plan.json") != plan or read_json(study / "source_guards.json") != guards:
            raise ValueError("Changed protocol or prior evidence")
        context = recorded
    else:
        if resume:
            raise FileNotFoundError("No existing run to resume")
        if any(base.glob(f"{prefix}_*_norm*")) or any((ROOT / "submissions").glob(f"{prefix}_*.csv")):
            raise FileExistsError("Candidate artifacts already exist without this study; use a new prefix")
        study.mkdir(parents=True)
        q5.atomic_json(study / "provenance.json", context)
        q5.atomic_json(study / "source_guards.json", guards)
        q5.atomic_json(study / "q5_quote_plan.json", plan)
        q5.atomic_json(study / "q5_feature_plan.json", fp)
        for name in context["source_sha256"]:
            target = study / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        for name in ("train_ids.csv", "validation_ids.csv", "manifest.json"):
            shutil.copy2(SPLIT_DIR / name, study / ("split_manifest.json" if name == "manifest.json" else name))
    if read_json(study / "split_manifest.json") != read_json(SPLIT_DIR / "manifest.json"):
        raise ValueError("Canonical split changed")
    return study, plan, fp, context, guards


def run_cv(study, plan, fp, train):
    text = [r["text"] for r in train]
    y = np.array([int(r["label"]) for r in train])
    folds = list(StratifiedKFold(3, shuffle=True, random_state=plan["seeds"]["inner_cv"]).split(text, y))
    assignment = np.empty(len(train), dtype=int)
    for fold, (_, held) in enumerate(folds):
        assignment[held] = fold
    q5.atomic_csv(study / "inner_folds.csv", ["id", "inner_validation_fold"],
                  ({"id": r["id"], "inner_validation_fold": int(assignment[i])} for i, r in enumerate(train)))
    path = study / "cv_progress.json"
    rows = read_json(path) if path.exists() else []
    done = {(r["model_id"], r["fold"]) for r in rows}
    if len(done) != len(rows):
        raise ValueError("Duplicate CV checkpoint")
    for row in rows:
        if (row["representation"] not in REPRESENTATIONS or row["fold"] not in range(3)
                or type(row["normalize_quotes"]) is not bool
                or row["model_id"] != model_id(row["representation"], row["normalize_quotes"])):
            raise ValueError("Invalid CV checkpoint")
        fit, held = folds[row["fold"]]
        if (row["fit_ids_sha256"] != id_hash([train[i]["id"] for i in fit]) or row["held_ids_sha256"] != id_hash([train[i]["id"] for i in held])):
            raise ValueError("Changed CV fitting IDs")
        if any(not np.isfinite(row[field]) or not 0 <= row[field] <= 1
               for field in ("original_auroc", "quote_auroc", "dash_auroc")):
            raise ValueError("Invalid saved CV score")
    for rep in REPRESENTATIONS:
        for normalized in (False, True):
            mid = model_id(rep, normalized)
            for fold, (fit, held) in enumerate(folds):
                if (mid, fold) in done:
                    continue
                q5.record_status(study, f"cv_{mid}_fold_{fold}", completed_cv_fits=len(rows))
                model = make_model(rep, normalized, fp, calibrated=False)
                seconds, notes = q5.fit_record(model, [text[i] for i in fit], y[fit])
                held_text = [text[i] for i in held]
                margins = {condition: model.decision_function([apply_condition(t, condition) for t in held_text]) for condition in CONDITIONS}
                error = float(np.max(abs(margins["original"] - margins["quote_typography"])))
                if normalized and error != 0:
                    raise ValueError("Canonical model violates idempotent quote mapping")
                rows.append({"model_id": mid, "representation": rep, "normalize_quotes": normalized,
                             "fold": fold, "original_auroc": float(roc_auc_score(y[held], margins["original"])),
                             "quote_auroc": float(roc_auc_score(y[held], margins["quote_typography"])),
                             "dash_auroc": float(roc_auc_score(y[held], margins["dash_typography"])),
                             "fit_seconds": seconds, "n_features": q5.feature_counts(model.named_steps["features"])["total"],
                             "fit_ids_sha256": id_hash([train[i]["id"] for i in fit]),
                             "held_ids_sha256": id_hash([train[i]["id"] for i in held]),
                             "warnings": notes, "quote_invariance_max_margin_error": error})
                q5.atomic_json(path, rows)
                print(f"CV {len(rows)}/12: {mid}, fold {fold}, clean={rows[-1]['original_auroc']:.6f}, quote={rows[-1]['quote_auroc']:.6f}", flush=True)
                del model
    summaries = []
    for rep in REPRESENTATIONS:
        for normalized in (False, True):
            mid = model_id(rep, normalized)
            observed = [next(r for r in rows if r["model_id"] == mid and r["fold"] == fold) for fold in range(3)]
            clean = [r["original_auroc"] for r in observed]
            summaries.append({"model_id": mid, "representation": rep, "normalize_quotes": normalized,
                              "mean_cv_auroc": float(np.mean(clean)), "std_cv_auroc": float(np.std(clean, ddof=1)),
                              "mean_quote_cv_auroc": float(np.mean([r["quote_auroc"] for r in observed])),
                              "mean_dash_cv_auroc": float(np.mean([r["dash_auroc"] for r in observed])),
                              **{f"fold_{i}_auroc": r["original_auroc"] for i, r in enumerate(observed)},
                              **{f"fold_{i}_quote_auroc": r["quote_auroc"] for i, r in enumerate(observed)}})
    old = read_json(study.parent / plan["reference_study"] / "cv_progress.json")
    for rep in REPRESENTATIONS:
        row = next(r for r in summaries if r["model_id"] == f"{rep}_raw")
        for fold in range(3):
            value = next(r["auroc"] for r in old if r["model"] == rep and r["fold"] == fold)
            if abs(row[f"fold_{fold}_auroc"] - value) > 1e-12:
                raise ValueError("Original raw CV failed to reproduce")
    q5.atomic_csv(study / "cv_results.csv", list(summaries[0]), summaries)
    return summaries


def persist_normalized_fit(folder, rep, fp, context, study, records):
    folder.mkdir(exist_ok=True)
    fit_path = folder / "model_fit.json"
    if fit_path.exists():
        fit = read_json(fit_path)
        cfg = read_json(folder / "config.json")
        if cfg["representation"] != rep or not cfg["normalize_quotes"] or cfg["n_train"] != len(records) or cfg["feature_plan"] != fp:
            raise ValueError("Changed normalized model checkpoint settings")
        if sha256(folder / "model.joblib") != fit["model_sha256"]:
            raise ValueError("Changed normalized model checkpoint")
        return fit
    q5.record_status(study, f"fitting_{folder.name}")
    model = make_model(rep, True, fp)
    q5.atomic_json(folder / "config.json", {"run_id": folder.name, "study_id": study.name,
                   "representation": rep, "normalize_quotes": True, "C": 1, "n_train": len(records),
                   "held_out_evaluation": "fixed development validation" if len(records) == 8000 else None,
                   "training_order": "canonical train IDs" if len(records) == 8000 else "canonical train IDs followed by validation IDs",
                   "feature_plan": fp, "provenance": context, "estimator_parameters": clean_params(model)})
    q5.atomic_csv(folder / "train_ids.csv", ["id"], ({"id": r["id"]} for r in records))
    seconds, notes = q5.fit_record(model, [r["text"] for r in records], np.array([int(r["label"]) for r in records]))
    temporary = folder / "model.joblib.tmp"
    joblib.dump(model, temporary)
    os.replace(temporary, folder / "model.joblib")
    fit = {"fit_seconds": seconds, "warnings": notes, "model_sha256": sha256(folder / "model.joblib"), "completed_at_utc": q5.utc_now()}
    q5.atomic_json(fit_path, fit)
    return fit


def evaluate_matrix(study, prefix, plan, fp, context, train, validation):
    text = [r["text"] for r in validation]
    labels = np.array([int(r["label"]) for r in validation])
    rows, probabilities = [], {}
    predictions_dir = study / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    for rep in REPRESENTATIONS:
        for normalized in (False, True):
            mid = model_id(rep, normalized)
            folder = study.parent / (f"{prefix}_{mid}" if normalized else f"a_q5_20261108_v1_{rep}")
            if normalized:
                fit = persist_normalized_fit(folder, rep, fp, context, study, train)
            else:
                fit = {"fit_seconds": read_json(folder / "metrics.json")["fit_seconds"]}
            if (folder / "completed.json").exists():
                for name, expected in read_json(folder / "completed.json")["artifact_sha256"].items():
                    if sha256(folder / name) != expected:
                        raise ValueError("Changed completed model artifact")
            model = joblib.load(folder / "model.joblib")
            by_condition = {}
            for condition in CONDITIONS:
                changed_text = [apply_condition(t, condition) for t in text]
                p = predict_ai(model, changed_text)
                by_condition[condition] = p
                probabilities[f"{mid}__{condition}"] = p
                n_changed = sum(before != after for before, after in zip(text, changed_text, strict=True))
                rows.append({"model_id": mid, "representation": rep, "train_preprocessing": "quote_normalized" if normalized else "raw",
                             "condition": condition, "validation_auroc": float(roc_auc_score(labels, p)),
                             "n_changed": n_changed, "n": len(validation), "fit_seconds": fit["fit_seconds"]})
                q5.atomic_csv(predictions_dir / f"{mid}_{condition}.csv", ["id", "label", "p_ai"],
                              ({"id": r["id"], "label": r["label"], "p_ai": format(float(v), ".17g")} for r, v in zip(validation, p, strict=True)))
            if normalized:
                np.testing.assert_array_equal(by_condition["original"], by_condition["quote_typography"])
                pt = predict_ai(model, [r["text"] for r in train])
                metrics = {"train": {"n": len(train), **score([int(r["label"]) for r in train], pt)},
                           "validation": {"n": len(validation), **score(labels, by_condition["original"])},
                           "quote_validation_auroc": float(roc_auc_score(labels, by_condition["quote_typography"])),
                           "dash_validation_auroc": float(roc_auc_score(labels, by_condition["dash_typography"])),
                           "fit_seconds": fit["fit_seconds"], "warnings": fit["warnings"],
                           "feature_counts": q5.feature_counts(model.calibrated_classifiers_[0].estimator.named_steps["features"]),
                           "quote_invariance_is_by_construction": True}
                if not (folder / "completed.json").exists():
                    q5.atomic_json(folder / "metrics.json", metrics)
                    shutil.copy2(predictions_dir / f"{mid}_original.csv", folder / "validation_predictions.csv")
                    check(folder / "validation_predictions.csv", "validation")
                    q5.atomic_json(folder / "completed.json", {"artifact_sha256": {name: sha256(folder / name) for name in
                                   ("config.json", "model.joblib", "model_fit.json", "metrics.json", "validation_predictions.csv", "train_ids.csv")}})
                else:
                    np.testing.assert_array_equal(by_condition["original"], np.array([float(r["p_ai"]) for r in read_csv(folder / "validation_predictions.csv")]))
            else:
                expected = np.array([float(r["p_ai"]) for r in read_csv(folder / "validation_predictions.csv")])
                np.testing.assert_array_equal(by_condition["original"], expected)
            print(f"MATRIX {mid}: clean={roc_auc_score(labels, by_condition['original']):.6f}, quote={roc_auc_score(labels, by_condition['quote_typography']):.6f}", flush=True)
            del model
    q5.atomic_csv(study / "comparison.csv", list(rows[0]), rows)
    return rows, probabilities


def compute_bootstrap(study, plan, labels, probabilities):
    q5.record_status(study, "paired_matrix_bootstrap")
    rng = np.random.default_rng(plan["seeds"]["bootstrap"])
    groups = [np.flatnonzero(labels == y) for y in (0, 1)]
    sampled = {key: [] for key in probabilities}
    for _ in range(plan["evaluation"]["bootstrap_replicates"]):
        ix = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        for key, p in probabilities.items():
            sampled[key].append(float(roc_auc_score(labels[ix], p[ix])))
    comparisons = {}
    for rep in REPRESENTATIONS:
        for label, condition in (("clean", "original"), ("quote", "quote_typography")):
            left, right = f"{rep}_norm__{condition}", f"{rep}_raw__{condition}"
            comparisons[f"{rep}_{label}"] = {"left": left, "right": right,
                "point_delta": float(roc_auc_score(labels, probabilities[left]) - roc_auc_score(labels, probabilities[right])),
                "interval95": np.quantile(np.array(sampled[left]) - sampled[right], [.025, .975]).tolist()}
        for label, condition in (("clean", "original"), ("quote", "quote_typography")):
            left, right = f"{rep}_norm__{condition}", f"wc120_raw__{condition}"
            comparisons[f"{rep}_{label}_vs_incumbent"] = {"left": left, "right": right,
                "point_delta": float(roc_auc_score(labels, probabilities[left]) - roc_auc_score(labels, probabilities[right])),
                "interval95": np.quantile(np.array(sampled[left]) - sampled[right], [.025, .975]).tolist()}
    result = {"seed": plan["seeds"]["bootstrap"], "replicates": plan["evaluation"]["bootstrap_replicates"],
              "method": "class-stratified paired percentile bootstrap", "primary_comparison_key": "wc120_quote",
              "comparisons": comparisons, "auroc_95_intervals": {key: np.quantile(v, [.025, .975]).tolist() for key, v in sampled.items()},
              "limitations": "conditional development intervals; no training variation or selection adjustment; invariance is constructed"}
    q5.atomic_json(study / "bootstrap.json", result)
    return result


def length_diagnostics(study, train, validation, probabilities):
    cuts = np.quantile([len(r["text"].split()) for r in train], [1/3, 2/3])
    bins = np.digitize([len(r["text"].split()) for r in validation], cuts, right=True)
    y = np.array([int(r["label"]) for r in validation])
    rows = []
    for mid in ("c60_raw", "c60_norm", "wc120_raw", "wc120_norm"):
        p = probabilities[f"{mid}__original"]
        for b in range(3):
            mask = bins == b
            rows.append({"model_id": mid, "length_bin": b, "n": int(mask.sum()), "n_human": int(np.sum(y[mask] == 0)),
                         "n_ai": int(np.sum(y[mask] == 1)), "validation_auroc": float(roc_auc_score(y[mask], p[mask]))})
    q5.atomic_csv(study / "length_subgroups.csv", list(rows[0]), rows)
    q5.atomic_json(study / "length_boundaries.json", {"training_word_count_tertiles": cuts.tolist()})
    return rows


def select_candidate(plan, cv, matrix, uncertainty, length_rows):
    rules = plan["export_rule"]
    by_cv = {r["model_id"]: r for r in cv}
    auc = {(r["model_id"], r["condition"]): r["validation_auroc"] for r in matrix}
    base_cv = by_cv["wc120_raw"]["mean_cv_auroc"]
    base_lengths = {r["length_bin"]: r["validation_auroc"] for r in length_rows if r["model_id"] == "wc120_raw"}
    base_dash_delta = auc[("wc120_raw", "dash_typography")] - auc[("wc120_raw", "original")]
    diagnostics = {}
    for rep in REPRESENTATIONS:
        mid = f"{rep}_norm"
        gains = [by_cv[mid][f"fold_{i}_quote_auroc"] - by_cv[f"{rep}_raw"][f"fold_{i}_quote_auroc"] for i in range(3)]
        clean_delta = auc[(mid, "original")] - auc[("wc120_raw", "original")]
        cv_delta = by_cv[mid]["mean_cv_auroc"] - base_cv
        length_loss = max(base_lengths[r["length_bin"]] - r["validation_auroc"] for r in length_rows if r["model_id"] == mid)
        extra_dash_loss = base_dash_delta - (auc[(mid, "dash_typography")] - auc[(mid, "original")])
        checks = {"quote_cv_mean_improves_own_raw": float(np.mean(gains)) > 0,
                  "quote_cv_positive_fold_count_passes": sum(g > 0 for g in gains) >= rules["minimum_positive_quote_cv_folds"],
                  "quote_validation_interval_lower_vs_incumbent_positive": uncertainty["comparisons"][f"{rep}_quote_vs_incumbent"]["interval95"][0] > 0,
                  "clean_point_within_budget_vs_incumbent": clean_delta >= -rules["maximum_clean_point_regression_vs_incumbent"],
                  "clean_cv_within_budget_vs_incumbent": cv_delta >= -rules["maximum_clean_cv_regression_vs_incumbent"],
                  "length_guard_passes_vs_incumbent": length_loss <= rules["maximum_length_bin_regression_vs_incumbent"],
                  "dash_guard_passes_vs_incumbent": extra_dash_loss <= rules["maximum_extra_dash_loss_vs_incumbent"]}
        diagnostics[mid] = {"checks": checks, "eligible": all(checks.values()), "quote_cv_fold_gains": gains,
                            "clean_delta_vs_incumbent": clean_delta, "clean_cv_delta_vs_incumbent": cv_delta,
                            "maximum_length_bin_regression_vs_incumbent": length_loss, "extra_dash_loss_vs_incumbent": extra_dash_loss}
    selected = next((mid for mid in rules["candidate_priority"] if diagnostics[mid]["eligible"]), None)
    category = None
    if selected:
        rep = selected.removesuffix("_norm")
        category = ("clean_and_quote_improvement" if uncertainty["comparisons"][f"{rep}_clean_vs_incumbent"]["interval95"][0] > 0
                    else "exploratory_robustness_tradeoff")
    return {"selected_model_id": selected, "export_eligible": selected is not None, "candidate_category": category,
            "eligibility": diagnostics, "retain_incumbent_until_external_check": True,
            "new_kaggle_upload": False, "primary_comparison_key": "wc120_quote",
            "interpretation": "A tradeoff candidate is exploratory; constructed quote invariance is not proof of hidden-test gains."}


def export_candidate(study, prefix, selection, fp, context, train, validation):
    mid = selection["selected_model_id"]
    rep = mid.removesuffix("_norm")
    folder = study.parent / f"{prefix}_{mid}_full10000"
    output = ROOT / "submissions" / f"{folder.name}.csv"
    metadata = output.with_suffix(".metadata.json")
    if (folder / "completed.json").exists():
        for relative, expected in read_json(folder / "completed.json")["artifact_sha256"].items():
            if sha256(ROOT / relative) != expected:
                raise ValueError("Changed completed export")
        return str(output.relative_to(ROOT))
    if not folder.exists() and (output.exists() or metadata.exists()):
        raise FileExistsError("Do not overwrite an existing CSV")
    records = train + validation
    fit = persist_normalized_fit(folder, rep, fp, context, study, records)
    q5.record_status(study, "predicting_selected_full_label_candidate")
    test_path = raw_path("test.csv")
    test = read_csv(test_path)
    model = joblib.load(folder / "model.joblib")
    p = predict_ai(model, [r["text"] for r in test])
    pt = predict_ai(model, [r["text"] for r in records])
    del model
    restored = predict_ai(joblib.load(folder / "model.joblib"), [r["text"] for r in test])
    np.testing.assert_array_equal(p, restored)
    q5.atomic_json(folder / "metrics.json", {"train": {"n": len(records), **score([int(r["label"]) for r in records], pt)},
                   "validation": None, "fit_seconds": fit["fit_seconds"], "warnings": fit["warnings"],
                   "note": "All supplied labels including the former validation rows; no held-out evaluation."})
    q5.atomic_csv(output, ["id", "label"], ({"id": r["id"], "label": format(float(v), ".17g")} for r, v in zip(test, p, strict=True)))
    count = check(output, "submission")
    q5.atomic_json(metadata, {"created_at_utc": q5.utc_now(), "model_run_id": folder.name, "source_study": study.name,
                   "representation": rep, "C": 1, "quote_normalization": True, "n_train": len(records), "n_test": count,
                   "full_refit": True, "full_refit_held_out_evaluation": None, "candidate_category": selection["candidate_category"],
                   "csv_sha256": sha256(output), "test_sha256": sha256(test_path), "model_sha256": fit["model_sha256"],
                   "model_reload_prediction_max_error": 0.0, "uploaded_to_kaggle": False,
                   "eligibility": selection["eligibility"][mid]})
    paths = [folder / n for n in ("config.json", "model_fit.json", "model.joblib", "metrics.json", "train_ids.csv")] + [output, metadata]
    q5.atomic_json(folder / "completed.json", {"artifact_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in paths}})
    return str(output.relative_to(ROOT))


def run(prefix, resume=False):
    study, plan, fp, context, guards = initialize(prefix, resume)
    if (study / "status.json").exists() and read_json(study / "status.json")["stage"] == "complete":
        print("Quote study already complete; no refit or overwrite", flush=True)
        return
    started = time.perf_counter()
    q5.record_status(study, "loading_canonical_data", started_at_utc=q5.utc_now())
    try:
        train, validation = load_partitions()
        with threadpool_limits(limits=1):
            cv = run_cv(study, plan, fp, train)
            matrix, probabilities = evaluate_matrix(study, prefix, plan, fp, context, train, validation)
            uncertainty = compute_bootstrap(study, plan, np.array([int(r["label"]) for r in validation]), probabilities)
            lengths = length_diagnostics(study, train, validation, probabilities)
            selection = select_candidate(plan, cv, matrix, uncertainty, lengths)
            q5.atomic_json(study / "selection.json", selection)
            if selection["export_eligible"]:
                selection["submission_csv"] = export_candidate(study, prefix, selection, fp, context, train, validation)
                q5.atomic_json(study / "selection.json", selection)
        for relative, expected in guards.items():
            if sha256(ROOT / relative) != expected:
                raise ValueError(f"Prior evidence changed: {relative}")
        q5.record_status(study, "complete", completed_at_utc=q5.utc_now(), completed_cv_fits=12,
                         exported_candidate=selection.get("submission_csv"), wall_seconds=time.perf_counter() - started)
    except BaseException as exc:
        q5.record_status(study, "failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    (ROOT / "tmp").mkdir(exist_ok=True)
    with (ROOT / "tmp/q5_quotes.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("A quote-canonicalization worker is already active") from exc
        run(args.prefix, args.resume)


if __name__ == "__main__":
    main()
