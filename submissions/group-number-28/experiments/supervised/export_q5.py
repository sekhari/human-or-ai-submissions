"""Refit the selected Q5 candidate on all supplied labels and export a local CSV.

This is a separate post-study operation. It preserves the original 8,000-row
checkpoints and never treats the old validation rows as held out after refitting.
No Kaggle upload, parameter search, or test-set fitting is performed.
"""
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
from threadpoolctl import threadpool_limits

from human_ai.check_predictions import check
from human_ai.data import ROOT, load_partitions, raw_path, read_csv, sha256
from .run_q1 import clean_params, predict_ai, score
from .run_q5 import (MODEL_KEYS, atomic_csv, atomic_json, feature_counts,
                     fit_record, make_model, provenance, record_status, utc_now)


def read_json(path):
    return json.loads(Path(path).read_text())


def export(prefix, model_key, output, resume=False):
    if not prefix or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in prefix):
        raise ValueError("Invalid source-study prefix")
    base = ROOT / "results/supervised"
    study = base / f"{prefix}_study"
    source = base / f"{prefix}_{model_key}"
    selection = read_json(study / "selection.json")
    if read_json(study / "status.json")["stage"] != "complete":
        raise ValueError("Source Q5 study is not complete")
    if selection["nominal_highest_validation_model"] != model_key:
        raise ValueError("This exporter uses the recorded leading Q5 candidate only")
    verification_path = base / f"{prefix}_verification.json"
    if not read_json(verification_path)["all_checks_passed"]:
        raise ValueError("Source Q5 verification did not pass")
    source_config = read_json(source / "config.json")
    plan = read_json(study / "q5_plan.json")
    if source_config["plan"] != plan:
        raise ValueError("Source model differs from the frozen study plan")
    for name, expected in read_json(study / "provenance.json")["source_sha256"].items():
        if sha256(ROOT / name) != expected or sha256(study / "source" / name) != expected:
            raise ValueError(f"Changed study execution source: {name}")
    source_receipt = read_json(source / "completed.json")
    for name, expected in source_receipt["artifact_sha256"].items():
        if sha256(source / name) != expected:
            raise ValueError(f"Changed source artifact: {name}")
    source_guard_paths = [study / "selection.json", study / "q5_plan.json", source / "validation_predictions.csv", source / "model.joblib"]
    source_guard = {str(path.relative_to(ROOT)): sha256(path) for path in source_guard_paths}

    output = Path(output).resolve()
    output.relative_to(ROOT)
    if output.suffix.lower() != ".csv":
        raise ValueError("Use a CSV output path inside the repository")
    metadata = output.with_suffix(".metadata.json")
    folder = base / f"{source.name}_full10000"
    context = provenance()
    original_context = read_json(study / "provenance.json")
    if any(context[key] != original_context[key] for key in ("python", "dependencies")):
        raise ValueError("Use the Python and dependency versions recorded by the source study")
    relative_exporter = str(Path(__file__).resolve().relative_to(ROOT))
    context["source_sha256"][relative_exporter] = sha256(Path(__file__))
    if folder.exists():
        if not resume:
            raise FileExistsError("Candidate artifacts exist; do not overwrite them")
        config = read_json(folder / "config.json")
        if config["plan"] != plan or config["output_csv"] != str(output) or config["source_run_id"] != source.name:
            raise ValueError("Cannot resume different candidate settings or output")
        for name, expected in config["provenance"]["source_sha256"].items():
            if sha256(ROOT / name) != expected:
                raise ValueError(f"Changed export source: {name}")
        for key in ("python", "dependencies"):
            if context[key] != config["provenance"][key]:
                raise ValueError(f"Changed export environment: {key}")
        if (folder / "completed.json").exists():
            receipt = read_json(folder / "completed.json")
            for name, expected in receipt["artifact_sha256"].items():
                if sha256(ROOT / name) != expected:
                    raise ValueError(f"Changed completed export artifact: {name}")
            check(output, "submission")
            print(f"Candidate already complete: {output}; no refit or overwrite", flush=True)
            return
    else:
        if resume:
            raise FileNotFoundError("No candidate checkpoint to resume")
        if output.exists() or metadata.exists():
            raise FileExistsError("Output CSV or metadata already exists")
        folder.mkdir()
        model = make_model(model_key, plan)
        config = {
            "run_id": folder.name, "source_run_id": source.name, "study_id": study.name,
            "model": model_key, "n_train": 10000, "plan": plan,
            "training_order": "canonical training IDs followed by canonical validation IDs",
            "held_out_evaluation": None,
            "source_8000_model_validation_auroc": read_json(source / "metrics.json")["validation"]["auroc"],
            "scope": "Separate post-Q5 full-label candidate refit requested for manual Kaggle scoring; team final selection remains open",
            "output_csv": str(output), "provenance": context,
            "estimator_parameters": clean_params(model),
            "test_sha256": sha256(raw_path("test.csv")),
            "source_artifact_sha256": source_guard,
            "selection_verification_sha256": sha256(verification_path),
        }
        atomic_json(folder / "config.json", config)
        for relative in context["source_sha256"]:
            target = folder / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        shutil.copy2(__file__, folder / "export_q5.py")
    started = time.perf_counter()
    try:
        record_status(folder, "loading_all_supplied_labels")
        train, validation = load_partitions()
        records = train + validation
        if len(records) != 10000 or len({row["id"] for row in records}) != 10000:
            raise ValueError("Expected all 10,000 unique supplied labeled rows")
        if config["test_sha256"] != sha256(raw_path("test.csv")):
            raise ValueError("Test input changed since export initialization")
        text = [row["text"] for row in records]
        labels = np.array([int(row["label"]) for row in records])
        test = read_csv(raw_path("test.csv"))
        if len(test) != 3000 or len({row["id"] for row in test}) != 3000:
            raise ValueError("Expected 3,000 unique test IDs")
        if set(row["id"] for row in records) & set(row["id"] for row in test):
            raise ValueError("Labeled and test IDs overlap")
        atomic_csv(folder / "train_ids.csv", ["id"], ({"id": row["id"]} for row in records))
        with threadpool_limits(limits=1):
            fit_path = folder / "model_fit.json"
            if fit_path.exists():
                fitted = read_json(fit_path)
                if sha256(folder / "model.joblib") != fitted["model_sha256"]:
                    raise ValueError("Changed fitted-model checkpoint")
                model = joblib.load(folder / "model.joblib")
            else:
                record_status(folder, "fitting_wc120_on_10000_labels")
                model = make_model(model_key, plan)
                seconds, notes = fit_record(model, text, labels)
                temporary = folder / "model.joblib.tmp"
                joblib.dump(model, temporary)
                os.replace(temporary, folder / "model.joblib")
                fitted = {"fit_seconds": seconds, "warnings": notes,
                          "model_sha256": sha256(folder / "model.joblib"), "completed_at_utc": utc_now()}
                atomic_json(fit_path, fitted)
            record_status(folder, "predicting_and_reloading_saved_model")
            test_text = [row["text"] for row in test]
            probabilities = predict_ai(model, test_text)
            train_probabilities = predict_ai(model, text)
            counts = feature_counts(model.calibrated_classifiers_[0].estimator.named_steps["features"])
            del model
            restored = joblib.load(folder / "model.joblib")
            reloaded_probabilities = predict_ai(restored, test_text)
            np.testing.assert_array_equal(probabilities, reloaded_probabilities)
        metrics = {"train": {"n": len(records), **score(labels, train_probabilities)}, "validation": None,
                   "fit_seconds": fitted["fit_seconds"], "warnings": fitted["warnings"],
                   "feature_counts": counts, "n_features": counts["total"],
                   "timing_definition": "one CPU thread; full feature fitting and training-only sigmoid calibration included",
                   "note": "All supplied labels include the former validation rows; this refit has no held-out evaluation."}
        atomic_json(folder / "metrics.json", metrics)
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_csv(output, ["id", "label"],
                   ({"id": row["id"], "label": format(float(value), ".17g")}
                    for row, value in zip(test, probabilities, strict=True)))
        count = check(output, "submission")
        if [row["id"] for row in read_csv(output)] != [row["id"] for row in test]:
            raise ValueError("Export order differs from original test IDs")
        for relative, expected in source_guard.items():
            if sha256(ROOT / relative) != expected:
                raise ValueError(f"Source study changed during export: {relative}")
        atomic_json(metadata, {
            "created_at_utc": utc_now(), "model_run_id": folder.name,
            "source_run_id": source.name, "n_train": len(records), "n_test": count,
            "csv_sha256": sha256(output), "full_refit": True, "uploaded_to_kaggle": False,
            "model_sha256": fitted["model_sha256"], "test_sha256": config["test_sha256"],
            "export_source_sha256": sha256(Path(__file__)),
            "model_reload_prediction_max_error": float(np.max(abs(probabilities - reloaded_probabilities))),
            "source_8000_model_validation_auroc": config["source_8000_model_validation_auroc"],
            "full_refit_held_out_evaluation": None,
            "probability_summary": {"min": float(probabilities.min()), "max": float(probabilities.max()),
                                    "mean": float(probabilities.mean())},
        })
        atomic_json(folder / "verification.json", {
            "all_checks_passed": True, "n_train": len(records), "n_test": count,
            "canonical_all_label_ids_used_once": True, "test_ids_match_original_order": True,
            "test_used_for_fitting": False, "source_8000_artifacts_unchanged": True,
            "model_reload_max_probability_error": 0.0, "csv_sha256": sha256(output),
            "feature_counts": counts, "uploaded_to_kaggle": False,
        })
        tracked = [folder / name for name in ("config.json", "metrics.json", "model_fit.json", "model.joblib", "train_ids.csv", "verification.json")]
        tracked += [output, metadata]
        atomic_json(folder / "completed.json", {"completed_at_utc": utc_now(),
                    "artifact_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in tracked}})
        record_status(folder, "complete", completed_at_utc=utc_now(),
                      wall_seconds=time.perf_counter() - started, output_csv=str(output))
        print(f"Validated {count}-row Q5 candidate: {output}; no Kaggle upload", flush=True)
    except BaseException as exc:
        record_status(folder, "failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--model", choices=MODEL_KEYS, default="wc120")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    lock_folder = ROOT / "tmp"
    lock_folder.mkdir(exist_ok=True)
    with (lock_folder / "q5_export.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("A Q5 export is already running") from exc
        export(args.prefix, args.model, args.output, args.resume)


if __name__ == "__main__":
    main()
