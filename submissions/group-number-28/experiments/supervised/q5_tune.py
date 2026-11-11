"""Run the bounded six-configuration Q5 follow-up and export at most one candidate."""
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from human_ai.check_predictions import check
from human_ai.data import ROOT, SPLIT_DIR, load_partitions, raw_path, read_csv, sha256
from . import run_q5 as q5
from .q5_sensitivity import evaluate_sensitivity
from .run_q1 import clean_params, id_hash, predict_ai, score

HERE = Path(__file__).resolve().parent
REPRESENTATIONS = ("c60", "wc120")
BASELINE = "baseline_wc120"
EXTRA_SOURCES = ("experiments/supervised/q5_tune.py", "experiments/supervised/q5_tuning_plan.json",
                 "experiments/supervised/q5_sensitivity.py", "experiments/supervised/q5_tune_job.py")


def load_plan():
    return json.loads((HERE / "q5_tuning_plan.json").read_text())


def candidate_id(representation, C):
    return f"{representation}_c{format(float(C), 'g').replace('.', 'p')}"


def make_model(representation, C, feature_plan, calibrated=True):
    actual = deepcopy(feature_plan)
    actual["classifier"]["C"] = C
    return q5.make_model(representation, actual, calibrated=calibrated)


def choose_parameters(cv_rows):
    preference = {1.0: 2, 0.1: 1, 10.0: 0}
    return {rep: max((r for r in cv_rows if r["representation"] == rep),
                     key=lambda r: (r["mean_cv_auroc"], -r["std_cv_auroc"], preference[float(r["C"])]))
            for rep in REPRESENTATIONS}


def initialize(prefix, resume):
    if not prefix or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in prefix):
        raise ValueError("Invalid prefix")
    plan = load_plan()
    feature_plan = json.loads((ROOT / plan["feature_plan"]).read_text())
    original_study = ROOT / "results/supervised" / plan["reference_study"]
    original_context = json.loads((original_study / "provenance.json").read_text())
    for relative, expected in original_context["source_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Original Q5 source changed: {relative}")
    if json.loads((original_study / "status.json").read_text())["stage"] != "complete":
        raise ValueError("Reference study is incomplete")
    guard_paths = [original_study / name for name in ("q5_plan.json", "selection.json", "cv_progress.json")]
    for rep in REPRESENTATIONS:
        origin = original_study.parent / f"a_q5_20261108_v1_{rep}"
        receipt = json.loads((origin / "completed.json").read_text())
        for name, expected in receipt["artifact_sha256"].items():
            if sha256(origin / name) != expected:
                raise ValueError(f"Changed reference artifact: {origin / name}")
        guard_paths += [origin / name for name in ("model.joblib", "validation_predictions.csv", "metrics.json")]
    previous_submission = ROOT / "submissions/a_q5_20261108_v1_wc120_full10000.csv"
    if previous_submission.exists():
        guard_paths.append(previous_submission)
    guards = {str(path.relative_to(ROOT)): sha256(path) for path in guard_paths}
    study = ROOT / "results/supervised" / f"{prefix}_study"
    context = q5.provenance()
    context["source_sha256"].update({path: sha256(ROOT / path) for path in EXTRA_SOURCES})
    if any(context[key] != original_context[key] for key in ("python", "dependencies")):
        raise ValueError("Use the recorded Q5 environment")
    if study.exists():
        if not resume:
            raise FileExistsError("Use a NEW prefix or resume the identical unfinished study")
        recorded = json.loads((study / "provenance.json").read_text())
        for relative, expected in recorded["source_sha256"].items():
            if sha256(ROOT / relative) != expected:
                raise ValueError(f"Cannot resume changed source: {relative}")
        if (json.loads((study / "q5_tuning_plan.json").read_text()) != plan or
                json.loads((study / "source_guards.json").read_text()) != guards):
            raise ValueError("Changed protocol or reference evidence")
        context = recorded
    else:
        if resume:
            raise FileNotFoundError("No existing study to resume")
        study.mkdir(parents=True)
        q5.atomic_json(study / "provenance.json", context)
        q5.atomic_json(study / "source_guards.json", guards)
        for relative in context["source_sha256"]:
            target = study / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        shutil.copy2(HERE / "q5_tuning_plan.json", study / "q5_tuning_plan.json")
        shutil.copy2(ROOT / plan["feature_plan"], study / "q5_feature_plan.json")
        for name in ("train_ids.csv", "validation_ids.csv", "manifest.json"):
            shutil.copy2(SPLIT_DIR / name, study / ("split_manifest.json" if name == "manifest.json" else name))
    if json.loads((study / "split_manifest.json").read_text()) != json.loads((SPLIT_DIR / "manifest.json").read_text()):
        raise ValueError("Canonical split changed")
    return study, plan, feature_plan, context, guards


def run_cv(study, plan, feature_plan, train):
    text = [r["text"] for r in train]
    labels = np.array([int(r["label"]) for r in train])
    folds = list(StratifiedKFold(3, shuffle=True, random_state=plan["seeds"]["inner_cv"]).split(text, labels))
    assignment = np.empty(len(train), dtype=int)
    for fold, (_, held) in enumerate(folds):
        assignment[held] = fold
    q5.atomic_csv(study / "inner_folds.csv", ["id", "inner_validation_fold"],
                  ({"id": r["id"], "inner_validation_fold": int(assignment[i])} for i, r in enumerate(train)))
    path = study / "cv_progress.json"
    observations = json.loads(path.read_text()) if path.exists() else []
    audits_path = study / "feature_fit_audit.json"
    audits = json.loads(audits_path.read_text()) if audits_path.exists() else []
    done = {(r["candidate_id"], r["fold"]) for r in observations}
    if len(done) != len(observations):
        raise ValueError("Duplicate CV checkpoints")
    for row in observations:
        if row["representation"] not in REPRESENTATIONS or row["C"] not in plan["C_candidates"] or row["fold"] not in range(3):
            raise ValueError("Invalid CV candidate")
        fit, held = folds[row["fold"]]
        if (row["candidate_id"] != candidate_id(row["representation"], row["C"]) or
                row["fit_ids_sha256"] != id_hash([train[i]["id"] for i in fit]) or
                row["held_ids_sha256"] != id_hash([train[i]["id"] for i in held]) or
                not np.isfinite(row["auroc"]) or not 0 <= row["auroc"] <= 1):
            raise ValueError("Corrupt CV checkpoint")
    for rep in REPRESENTATIONS:
        for fold, (fit, held) in enumerate(folds):
            todo = [C for C in plan["C_candidates"] if (candidate_id(rep, C), fold) not in done]
            if not todo:
                continue
            q5.record_status(study, f"preparing_{rep}_fold_{fold}", completed_cv_fits=len(observations))
            features = q5.feature_transformer(rep, feature_plan)
            started = time.perf_counter()
            x = features.fit_transform([text[i] for i in fit])
            z = features.transform([text[i] for i in held])
            seconds = time.perf_counter() - started
            audit = {"representation": rep, "fold": fold, "n_features": int(x.shape[1]),
                     "feature_fit_transform_seconds": seconds,
                     "fit_ids_sha256": id_hash([train[i]["id"] for i in fit]),
                     "held_ids_sha256": id_hash([train[i]["id"] for i in held])}
            audits = [r for r in audits if (r["representation"], r["fold"]) != (rep, fold)] + [audit]
            q5.atomic_json(audits_path, audits)
            for C in todo:
                cid = candidate_id(rep, C)
                q5.record_status(study, f"cv_{cid}_fold_{fold}", completed_cv_fits=len(observations))
                classifier = make_model(rep, C, feature_plan, calibrated=False).named_steps["classifier"]
                fit_seconds, notes = q5.fit_record(classifier, x, labels[fit])
                auc = float(roc_auc_score(labels[held], classifier.decision_function(z)))
                observations.append({"candidate_id": cid, "representation": rep, "C": C,
                                     "fold": fold, "auroc": auc, "classifier_fit_seconds": fit_seconds,
                                     "feature_fit_transform_seconds_shared": seconds,
                                     "n_features": int(x.shape[1]), "fit_ids_sha256": audit["fit_ids_sha256"],
                                     "held_ids_sha256": audit["held_ids_sha256"], "warnings": notes})
                q5.atomic_json(path, observations)
                print(f"CV {len(observations)}/18: {cid}, fold {fold}, AUROC={auc:.6f}", flush=True)
            del x, z, features
    rows = []
    for rep in REPRESENTATIONS:
        for C in plan["C_candidates"]:
            cid = candidate_id(rep, C)
            values = [next(r["auroc"] for r in observations if r["candidate_id"] == cid and r["fold"] == f) for f in range(3)]
            rows.append({"candidate_id": cid, "representation": rep, "C": C,
                         "mean_cv_auroc": float(np.mean(values)), "std_cv_auroc": float(np.std(values, ddof=1)),
                         **{f"fold_{i}_auroc": value for i, value in enumerate(values)}})
    q5.atomic_csv(study / "cv_results.csv", list(rows[0]), rows)
    reference = json.loads((study.parent / plan["reference_study"] / "cv_progress.json").read_text())
    gates = {}
    for rep in REPRESENTATIONS:
        new = next(r for r in rows if r["representation"] == rep and r["C"] == 1)
        errors = [abs(new[f"fold_{fold}_auroc"] - next(r["auroc"] for r in reference if r["model"] == rep and r["fold"] == fold)) for fold in range(3)]
        if max(errors) > 1e-12:
            raise ValueError(f"C=1 CV failed to reproduce Q5: {rep}")
        gates[rep] = {"max_cv_auroc_error": max(errors)}
    q5.atomic_json(study / "baseline_cv_gate.json", gates)
    chosen = choose_parameters(rows)
    q5.atomic_json(study / "chosen_parameters.json", {rep: row["C"] for rep, row in chosen.items()})
    q5.atomic_json(study / "finalists.json", chosen)
    q5.record_status(study, "training_only_selection_frozen", completed_cv_fits=18,
                     finalists=[row["candidate_id"] for row in chosen.values()])
    return rows, chosen


def persist_fit(folder, rep, C, feature_plan, context, study, records, held_out):
    folder.mkdir(exist_ok=True)
    fit_path = folder / "model_fit.json"
    if fit_path.exists():
        record = json.loads(fit_path.read_text())
        config = json.loads((folder / "config.json").read_text())
        if (config["representation"] != rep or config["C"] != C or config["n_train"] != len(records)
                or config["feature_plan"] != feature_plan):
            raise ValueError("Changed fitted-model settings")
        if sha256(folder / "model.joblib") != record["model_sha256"]:
            raise ValueError("Changed model checkpoint")
        return record
    q5.record_status(study, f"fitting_{folder.name}")
    model = make_model(rep, C, feature_plan)
    config = {"run_id": folder.name, "study_id": study.name, "representation": rep, "C": C,
              "feature_plan": feature_plan, "n_train": len(records), "held_out_evaluation": held_out,
              "training_order": "canonical training IDs" if len(records) == 8000 else "canonical training IDs followed by validation IDs",
              "provenance": context, "estimator_parameters": clean_params(model)}
    q5.atomic_json(folder / "config.json", config)
    q5.atomic_csv(folder / "train_ids.csv", ["id"], ({"id": r["id"]} for r in records))
    seconds, notes = q5.fit_record(model, [r["text"] for r in records], np.array([int(r["label"]) for r in records]))
    temporary = folder / "model.joblib.tmp"
    joblib.dump(model, temporary)
    os.replace(temporary, folder / "model.joblib")
    record = {"fit_seconds": seconds, "warnings": notes, "model_sha256": sha256(folder / "model.joblib"),
              "completed_at_utc": q5.utc_now()}
    q5.atomic_json(fit_path, record)
    return record


def evaluate_finalists(study, prefix, plan, feature_plan, context, train, validation, cv_rows, finalists):
    baseline_folder = study.parent / plan["baseline_run_id"]
    model = joblib.load(baseline_folder / "model.joblib")
    text = [r["text"] for r in validation]
    labels = np.array([int(r["label"]) for r in validation])
    saved = np.array([float(r["p_ai"]) for r in read_csv(baseline_folder / "validation_predictions.csv")])
    np.testing.assert_array_equal(predict_ai(model, text), saved)
    shutil.copy2(baseline_folder / "validation_predictions.csv", study / "baseline_validation_predictions.csv")
    metrics = json.loads((baseline_folder / "metrics.json").read_text())
    baseline_cv = next(r for r in cv_rows if r["representation"] == "wc120" and r["C"] == 1)
    rows = [{"candidate_id": BASELINE, "representation": "wc120", "C": 1,
             "cv_mean_auroc": baseline_cv["mean_cv_auroc"], "cv_std_auroc": baseline_cv["std_cv_auroc"],
             "train_auroc": metrics["train"]["auroc"], "validation_auroc": metrics["validation"]["auroc"],
             "fit_seconds": metrics["fit_seconds"], "timing_source": "historical Q5 fit"}]
    probabilities = {BASELINE: saved}
    models = {BASELINE: model}
    for rep, chosen in finalists.items():
        cid, C = chosen["candidate_id"], chosen["C"]
        folder = study.parent / f"{prefix}_{cid}"
        fit = persist_fit(folder, rep, C, feature_plan, context, study, train, "canonical 2000-row development validation")
        model = joblib.load(folder / "model.joblib")
        p = predict_ai(model, text)
        if (folder / "completed.json").exists():
            receipt = json.loads((folder / "completed.json").read_text())
            for name, expected in receipt["artifact_sha256"].items():
                if sha256(folder / name) != expected:
                    raise ValueError(f"Changed completed finalist artifact: {name}")
            values = json.loads((folder / "metrics.json").read_text())
            np.testing.assert_array_equal(p, np.array([float(r["p_ai"]) for r in read_csv(folder / "validation_predictions.csv")]))
        else:
            pt = predict_ai(model, [r["text"] for r in train])
            counts = q5.feature_counts(model.calibrated_classifiers_[0].estimator.named_steps["features"])
            values = {"train": {"n": len(train), **score([int(r["label"]) for r in train], pt)},
                      "validation": {"n": len(validation), **score(labels, p)}, "fit_seconds": fit["fit_seconds"],
                      "n_features": counts["total"], "feature_counts": counts, "warnings": fit["warnings"],
                      "timing_definition": "one CPU thread, full features and calibration, CV excluded"}
            q5.atomic_json(folder / "metrics.json", values)
            q5.atomic_csv(folder / "validation_predictions.csv", ["id", "label", "p_ai"],
                          ({"id": r["id"], "label": r["label"], "p_ai": format(float(v), ".17g")} for r, v in zip(validation, p, strict=True)))
            check(folder / "validation_predictions.csv", "validation")
            q5.atomic_json(folder / "completed.json", {"artifact_sha256": {name: sha256(folder / name) for name in
                           ("model.joblib", "model_fit.json", "config.json", "metrics.json", "validation_predictions.csv", "train_ids.csv")}})
        rows.append({"candidate_id": cid, "representation": rep, "C": C,
                     "cv_mean_auroc": chosen["mean_cv_auroc"], "cv_std_auroc": chosen["std_cv_auroc"],
                     "train_auroc": values["train"]["auroc"], "validation_auroc": values["validation"]["auroc"],
                     "fit_seconds": fit["fit_seconds"], "timing_source": "current finalist fit"})
        probabilities[cid], models[cid] = p, model
        print(f"FINALIST {cid}: validation AUROC={values['validation']['auroc']:.6f}", flush=True)
    q5.atomic_csv(study / "comparison.csv", list(rows[0]), rows)
    return rows, probabilities, models


def bootstrap(study, plan, labels, probabilities):
    q5.record_status(study, "paired_bootstrap")
    rng = np.random.default_rng(plan["seeds"]["bootstrap"])
    groups = [np.flatnonzero(labels == y) for y in (0, 1)]
    values = {key: [] for key in probabilities}
    for _ in range(plan["evaluation"]["bootstrap_replicates"]):
        ix = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        for key, p in probabilities.items():
            values[key].append(float(roc_auc_score(labels[ix], p[ix])))
    intervals = {key: np.quantile(v, [.025, .975]).tolist() for key, v in values.items()}
    deltas = {key: np.quantile(np.array(v) - values[BASELINE], [.025, .975]).tolist() for key, v in values.items()}
    result = {"seed": plan["seeds"]["bootstrap"], "replicates": plan["evaluation"]["bootstrap_replicates"],
              "method": "class-stratified paired percentile bootstrap", "auroc_95_intervals": intervals,
              "delta_vs_baseline_95_intervals": deltas,
              "point_deltas_vs_baseline": {key: float(roc_auc_score(labels, p) - roc_auc_score(labels, probabilities[BASELINE])) for key, p in probabilities.items()},
              "limitations": "development split, conditional on fitted models; training variation and selection correction omitted"}
    q5.atomic_json(study / "bootstrap.json", result)
    return result


def length_diagnostics(study, train, validation, probabilities):
    cuts = np.quantile([len(r["text"].split()) for r in train], [1/3, 2/3])
    bins = np.digitize([len(r["text"].split()) for r in validation], cuts, right=True)
    labels = np.array([int(r["label"]) for r in validation])
    rows = []
    for key, p in probabilities.items():
        for b in range(3):
            mask = bins == b
            rows.append({"candidate_id": key, "length_bin": b, "n": int(mask.sum()),
                         "n_human": int(np.sum(labels[mask] == 0)), "n_ai": int(np.sum(labels[mask] == 1)),
                         "validation_auroc": float(roc_auc_score(labels[mask], p[mask])) if len(set(labels[mask])) == 2 else None})
    q5.atomic_csv(study / "length_subgroups.csv", list(rows[0]), rows)
    q5.atomic_json(study / "length_boundaries.json", {"training_word_count_tertiles": cuts.tolist(),
                                                    "definition": "bin 0 <= cut0; bin 1 <= cut1; bin 2 > cut1"})
    return rows


def choose_export_candidate(plan, comparisons, uncertainty, length_rows, sensitivity):
    base = next(r for r in comparisons if r["candidate_id"] == BASELINE)
    base_lengths = {r["length_bin"]: r["validation_auroc"] for r in length_rows if r["candidate_id"] == BASELINE}
    formats = ("quote_typography", "dash_typography")
    diagnostics = {}
    for row in comparisons:
        key = row["candidate_id"]
        if key == BASELINE:
            continue
        regressions = [base_lengths[r["length_bin"]] - r["validation_auroc"] for r in length_rows if r["candidate_id"] == key]
        extra_format_loss = [sensitivity["models"][BASELINE][c]["delta_auroc"] - sensitivity["models"][key][c]["delta_auroc"] for c in formats]
        checks = {"different_from_current_settings": not (row["representation"] == "wc120" and row["C"] == 1),
                  "cv_mean_improves": row["cv_mean_auroc"] > base["cv_mean_auroc"],
                  "paired_interval_lower_positive": uncertainty["delta_vs_baseline_95_intervals"][key][0] > 0,
                  "length_guard_passes": max(regressions) <= plan["export_rule"]["maximum_length_bin_auroc_regression"],
                  "format_guard_passes": max(extra_format_loss) <= plan["export_rule"]["maximum_extra_format_auroc_loss_vs_baseline"]}
        diagnostics[key] = {"checks": checks, "eligible": all(checks.values()),
                            "maximum_length_bin_regression": max(regressions), "maximum_extra_format_loss": max(extra_format_loss)}
    available = [r for r in comparisons if r["candidate_id"] != BASELINE and diagnostics[r["candidate_id"]]["eligible"]]
    pool = available or [r for r in comparisons if r["candidate_id"] != BASELINE]
    selected = max(pool, key=lambda r: (r["validation_auroc"], r["cv_mean_auroc"], -r["cv_std_auroc"], r["representation"] == "c60"))
    return {"selected_candidate_id": selected["candidate_id"], "selected_representation": selected["representation"],
            "selected_C": selected["C"], "export_eligible": bool(available), "eligibility": diagnostics,
            "retain_current_public_baseline": not bool(available),
            "selection_scope": "local follow-up selection; no new Kaggle score or upload", "public_feedback_motivated_study": True}


def export_selected(study, prefix, feature_plan, context, selection, train, validation):
    cid = selection["selected_candidate_id"]
    folder = study.parent / f"{prefix}_{cid}_full10000"
    output = ROOT / "submissions" / f"{folder.name}.csv"
    metadata = output.with_suffix(".metadata.json")
    if (folder / "completed.json").exists():
        receipt = json.loads((folder / "completed.json").read_text())
        for relative, expected in receipt["artifact_sha256"].items():
            if sha256(ROOT / relative) != expected:
                raise ValueError("Changed full-label candidate")
        return str(output.relative_to(ROOT))
    if not folder.exists() and (output.exists() or metadata.exists()):
        raise FileExistsError("Do not overwrite candidate exports")
    records = train + validation
    rep, C = selection["selected_representation"], selection["selected_C"]
    fit = persist_fit(folder, rep, C, feature_plan, context, study, records, None)
    q5.record_status(study, "exporting_eligible_full_label_candidate")
    test_path = raw_path("test.csv")
    test = read_csv(test_path)
    model = joblib.load(folder / "model.joblib")
    p = predict_ai(model, [r["text"] for r in test])
    pt = predict_ai(model, [r["text"] for r in records])
    counts = q5.feature_counts(model.calibrated_classifiers_[0].estimator.named_steps["features"])
    del model
    restored = predict_ai(joblib.load(folder / "model.joblib"), [r["text"] for r in test])
    np.testing.assert_array_equal(p, restored)
    q5.atomic_json(folder / "metrics.json", {"train": {"n": len(records), **score([int(r["label"]) for r in records], pt)},
                   "validation": None, "fit_seconds": fit["fit_seconds"], "warnings": fit["warnings"],
                   "feature_counts": counts, "note": "Former validation labels are included; no held-out evaluation for this refit"})
    q5.atomic_csv(output, ["id", "label"], ({"id": r["id"], "label": format(float(v), ".17g")} for r, v in zip(test, p, strict=True)))
    count = check(output, "submission")
    q5.atomic_json(metadata, {"created_at_utc": q5.utc_now(), "source_study": study.name, "model_run_id": folder.name,
                   "representation": rep, "C": C, "n_train": len(records), "n_test": count, "full_refit": True,
                   "full_refit_held_out_evaluation": None, "csv_sha256": sha256(output), "test_sha256": sha256(test_path),
                   "model_sha256": fit["model_sha256"], "model_reload_prediction_max_error": 0.0,
                   "uploaded_to_kaggle": False, "eligibility_checks": selection["eligibility"][cid]})
    paths = [folder / n for n in ("model.joblib", "model_fit.json", "config.json", "metrics.json", "train_ids.csv")] + [output, metadata]
    q5.atomic_json(folder / "completed.json", {"artifact_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in paths}})
    print(f"Eligible candidate exported: {output}", flush=True)
    return str(output.relative_to(ROOT))


def run(prefix, resume=False):
    study, plan, feature_plan, context, guards = initialize(prefix, resume)
    if (study / "status.json").exists() and json.loads((study / "status.json").read_text())["stage"] == "complete":
        print("Tuning study already complete; no fits or overwrites", flush=True)
        return
    started = time.perf_counter()
    q5.record_status(study, "loading_canonical_data", started_at_utc=q5.utc_now())
    try:
        train, validation = load_partitions()
        with threadpool_limits(limits=1):
            cv_rows, finalists = run_cv(study, plan, feature_plan, train)
            comparisons, probabilities, models = evaluate_finalists(study, prefix, plan, feature_plan, context, train, validation, cv_rows, finalists)
            labels = np.array([int(r["label"]) for r in validation])
            uncertainty = bootstrap(study, plan, labels, probabilities)
            lengths = length_diagnostics(study, train, validation, probabilities)
            q5.record_status(study, "format_sensitivity_diagnostics")
            sensitivity = evaluate_sensitivity(study, models, validation, plan)
            del models
            selection = choose_export_candidate(plan, comparisons, uncertainty, lengths, sensitivity)
            q5.atomic_json(study / "selection.json", selection)
            if selection["export_eligible"]:
                selection["submission_csv"] = export_selected(study, prefix, feature_plan, context, selection, train, validation)
                q5.atomic_json(study / "selection.json", selection)
        for relative, expected in guards.items():
            if sha256(ROOT / relative) != expected:
                raise ValueError(f"Original evidence changed: {relative}")
        q5.record_status(study, "complete", completed_at_utc=q5.utc_now(), completed_cv_fits=18,
                         completed_finalists=list(finalists), exported_candidate=selection.get("submission_csv"),
                         wall_seconds=time.perf_counter() - started)
    except BaseException as exc:
        q5.record_status(study, "failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    (ROOT / "tmp").mkdir(exist_ok=True)
    with (ROOT / "tmp/q5_tune.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("A Q5 tuning worker is already active") from exc
        run(args.prefix, args.resume)


if __name__ == "__main__":
    main()
