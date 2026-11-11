"""Run a frozen four-group character/length/richness ablation with checkpoints."""
import argparse
import fcntl
import hashlib
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
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from human_ai.check_predictions import check
from human_ai.data import ROOT, SPLIT_DIR, load_partitions, raw_path, read_csv, sha256
from . import run_q5 as q5
from .q5_sensitivity import CONDITIONS, apply_condition
from .richness_features import LengthFeatures, RichnessFeatures
from .run_q1 import clean_params, id_hash, predict_ai, score

HERE = Path(__file__).resolve().parent
MODELS = ("c60", "c60_length", "c60_richness", "c60_length_richness")
INCUMBENT = "incumbent_wc120"
EXTRA_SOURCES = ("experiments/supervised/q5_richness.py", "experiments/supervised/q5_richness_plan.json",
                 "experiments/supervised/richness_features.py", "experiments/supervised/q5_sensitivity.py",
                 "experiments/supervised/q5_richness_job.py")


def read_json(path):
    return json.loads(Path(path).read_text())


def load_plan():
    return read_json(HERE / "q5_richness_plan.json")


def make_model(model_key, feature_plan, calibrated=True):
    if model_key not in MODELS:
        raise ValueError(model_key)
    base = q5.make_model("c60", feature_plan, calibrated=False)
    if model_key != "c60":
        blocks = [("char", base.named_steps["features"])]
        weights = {"char": 1.0}
        for name, extractor, dimension in (("length", LengthFeatures(), 2),
                                           ("richness", RichnessFeatures(window_size=50), 3)):
            if name in model_key.split("_"):
                blocks.append((name, Pipeline([("extract", extractor), ("scale", StandardScaler())])))
                weights[name] = 1 / np.sqrt(dimension)
        base.set_params(features=FeatureUnion(blocks, transformer_weights=weights, n_jobs=1))
    if not calibrated:
        return base
    folds = StratifiedKFold(3, shuffle=True, random_state=feature_plan["seeds"]["calibration"])
    return CalibratedClassifierCV(base, method="sigmoid", cv=folds, ensemble=False, n_jobs=1)


def feature_audit(features):
    blocks = dict(features.transformer_list) if isinstance(features, FeatureUnion) else {"char": features}
    vocabulary = blocks["char"].vocabulary_
    items = sorted((token, int(index)) for token, index in vocabulary.items())
    digest = hashlib.sha256(json.dumps(items, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    counts = {"char": len(vocabulary), "length": 0, "richness": 0}
    scalers = {}
    for name in ("length", "richness"):
        if name in blocks:
            scaler = blocks[name].named_steps["scale"]
            counts[name] = int(scaler.n_features_in_)
            scalers[name] = {"mean": scaler.mean_.tolist(), "var": scaler.var_.tolist(),
                             "scale": scaler.scale_.tolist(), "n_samples_seen": int(scaler.n_samples_seen_)}
    counts["total"] = sum(counts.values())
    return {"feature_counts": counts, "char_vocab_sha256": digest, "scalers": scalers}


def initialize(prefix, resume):
    if not prefix or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in prefix):
        raise ValueError("Invalid prefix")
    plan, fp = load_plan(), q5.load_plan()
    base = ROOT / "results/supervised"
    reference = base / plan["reference_study"]
    if read_json(reference / "status.json")["stage"] != "complete":
        raise ValueError("Historical reference is incomplete")
    old_context = read_json(reference / "provenance.json")
    for name, digest in old_context["source_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Changed historical source: {name}")
    paths = [reference / "cv_progress.json", reference / "q5_plan.json"]
    for mid in ("c60", "wc120"):
        folder = base / f"a_q5_20261108_v1_{mid}"
        for name, digest in read_json(folder / "completed.json")["artifact_sha256"].items():
            if sha256(folder / name) != digest:
                raise ValueError("Changed historical model")
        paths += [folder / n for n in ("model.joblib", "metrics.json", "validation_predictions.csv")]
    paths += [ROOT / "submissions/a_q5_20261108_v1_wc120_full10000.csv"]
    guards = {str(p.relative_to(ROOT)): sha256(p) for p in paths}
    context = q5.provenance()
    context["source_sha256"].update({name: sha256(ROOT / name) for name in EXTRA_SOURCES})
    if any(context[k] != old_context[k] for k in ("python", "dependencies")):
        raise ValueError("Use the pinned historical environment")
    study = base / f"{prefix}_study"
    if study.exists():
        if not resume:
            raise FileExistsError("Use a NEW prefix or resume identical settings")
        context = read_json(study / "provenance.json")
        for name, digest in context["source_sha256"].items():
            if sha256(ROOT / name) != digest:
                raise ValueError(f"Changed executing source: {name}")
        if read_json(study / "q5_richness_plan.json") != plan or read_json(study / "source_guards.json") != guards:
            raise ValueError("Changed plan or historical evidence")
    else:
        if resume:
            raise FileNotFoundError("No study to resume")
        if any(base.glob(f"{prefix}_*")) or any((ROOT / "submissions").glob(f"{prefix}_*")):
            raise FileExistsError("Orphan artifacts exist; choose a new prefix")
        study.mkdir(parents=True)
        for name, value in (("provenance.json", context), ("source_guards.json", guards),
                            ("q5_richness_plan.json", plan), ("q5_feature_plan.json", fp)):
            q5.atomic_json(study / name, value)
        for name in context["source_sha256"]:
            target = study / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        for name in ("train_ids.csv", "validation_ids.csv", "manifest.json"):
            shutil.copy2(SPLIT_DIR / name, study / ("split_manifest.json" if name == "manifest.json" else name))
    if read_json(study / "split_manifest.json") != read_json(SPLIT_DIR / "manifest.json"):
        raise ValueError("Changed canonical split")
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
    destination = study / "cv_predictions"
    destination.mkdir(exist_ok=True)
    path = study / "cv_progress.json"
    rows = read_json(path) if path.exists() else []
    done = {(r["model_id"], r["fold"]) for r in rows}
    if len(done) != len(rows):
        raise ValueError("Duplicate CV checkpoint")
    for row in rows:
        if row["model_id"] not in MODELS or row["fold"] not in range(3):
            raise ValueError("Unexpected CV checkpoint")
        fit, held = folds[row["fold"]]
        for name, indices in (("fit", fit), ("held", held)):
            if row[f"{name}_ids_sha256"] != id_hash([train[i]["id"] for i in indices]):
                raise ValueError("Changed fold IDs")
        saved = destination / f"{row['model_id']}_fold_{row['fold']}.csv"
        if sha256(saved) != row["predictions_sha256"]:
            raise ValueError("Changed CV margins")
        if any(not np.isfinite(row[k]) or not 0 <= row[k] <= 1 for k in ("original_auroc", "quote_auroc", "dash_auroc")):
            raise ValueError("Invalid CV score")
    for mid in MODELS:
        for fold, (fit, held) in enumerate(folds):
            if (mid, fold) in done:
                continue
            q5.record_status(study, f"cv_{mid}_fold_{fold}", completed_cv_fits=len(rows))
            model = make_model(mid, fp, calibrated=False)
            seconds, notes = q5.fit_record(model, [text[i] for i in fit], y[fit])
            margins = {condition: model.decision_function([apply_condition(text[i], condition) for i in held]) for condition in CONDITIONS}
            target = destination / f"{mid}_fold_{fold}.csv"
            q5.atomic_csv(target, ["id", "label", "original_margin", "quote_margin", "dash_margin"],
                          ({"id": train[i]["id"], "label": train[i]["label"],
                            **{f"{key}_margin": format(float(margins[condition][j]), ".17g")
                               for key, condition in zip(("original", "quote", "dash"), CONDITIONS, strict=True)}}
                           for j, i in enumerate(held)))
            audit = feature_audit(model.named_steps["features"])
            rows.append({"model_id": mid, "fold": fold, "fit_seconds": seconds, "warnings": notes,
                         **{f"{key}_auroc": float(roc_auc_score(y[held], margins[condition]))
                            for key, condition in zip(("original", "quote", "dash"), CONDITIONS, strict=True)},
                         "fit_ids_sha256": id_hash([train[i]["id"] for i in fit]),
                         "held_ids_sha256": id_hash([train[i]["id"] for i in held]),
                         "feature_audit": audit, "n_features": audit["feature_counts"]["total"],
                         "predictions_sha256": sha256(target)})
            q5.atomic_json(path, rows)
            print(f"CV {len(rows)}/12 {mid} fold {fold}: {rows[-1]['original_auroc']:.6f}", flush=True)
            del model
    summaries = []
    for mid in MODELS:
        observed = [next(r for r in rows if r["model_id"] == mid and r["fold"] == i) for i in range(3)]
        values = [r["original_auroc"] for r in observed]
        summaries.append({"model_id": mid, "mean_cv_auroc": float(np.mean(values)), "std_cv_auroc": float(np.std(values, ddof=1)),
                          "mean_quote_cv_auroc": float(np.mean([r["quote_auroc"] for r in observed])),
                          "mean_dash_cv_auroc": float(np.mean([r["dash_auroc"] for r in observed])),
                          **{f"fold_{i}{suffix}_auroc": r[f"{key}_auroc"] for i, r in enumerate(observed)
                             for suffix, key in (("", "original"), ("_quote", "quote"), ("_dash", "dash"))}})
    original = read_json(study.parent / plan["reference_study"] / "cv_progress.json")
    for fold in range(3):
        old = next(r["auroc"] for r in original if r["model"] == "c60" and r["fold"] == fold)
        if abs(summaries[0][f"fold_{fold}_auroc"] - old) > 1e-12:
            raise ValueError("C60 CV reproduction failed")
    q5.atomic_csv(study / "cv_results.csv", list(summaries[0]), summaries)
    # This choice uses training-only scores, before the current validation matrix.
    chosen = max(summaries[1:], key=lambda r: (r["mean_cv_auroc"], -MODELS.index(r["model_id"])))
    q5.atomic_json(study / "cv_candidate.json", {"model_id": chosen["model_id"], "mean_cv_auroc": chosen["mean_cv_auroc"],
                   "selection_uses_validation": False, "rule": plan["export_rule"]["selection"]})
    return summaries


def persist_fit(folder, mid, fp, context, study, records):
    folder.mkdir(exist_ok=True)
    target = folder / "model_fit.json"
    if target.exists():
        record = read_json(target)
        config = read_json(folder / "config.json")
        if config["model_id"] != mid or config["n_train"] != len(records) or config["feature_plan"] != fp:
            raise ValueError("Changed checkpoint settings")
        if sha256(folder / "model.joblib") != record["model_sha256"]:
            raise ValueError("Changed model checkpoint")
        return record
    q5.record_status(study, f"fitting_{folder.name}")
    model = make_model(mid, fp)
    q5.atomic_json(folder / "config.json", {"run_id": folder.name, "study_id": study.name, "model_id": mid,
                   "C": 1, "n_train": len(records), "held_out_evaluation": "fixed development validation" if len(records) == 8000 else None,
                   "feature_plan": fp, "provenance": context, "estimator_parameters": clean_params(model)})
    q5.atomic_csv(folder / "train_ids.csv", ["id"], ({"id": r["id"]} for r in records))
    seconds, notes = q5.fit_record(model, [r["text"] for r in records], np.array([int(r["label"]) for r in records]))
    temporary = folder / "model.joblib.tmp"
    joblib.dump(model, temporary)
    os.replace(temporary, folder / "model.joblib")
    record = {"fit_seconds": seconds, "warnings": notes, "model_sha256": sha256(folder / "model.joblib"), "completed_at_utc": q5.utc_now()}
    q5.atomic_json(target, record)
    return record


def evaluate_matrix(study, prefix, fp, context, train, validation):
    labels = np.array([int(r["label"]) for r in validation])
    text = [r["text"] for r in validation]
    predictions = study / "predictions"
    predictions.mkdir(exist_ok=True)
    rows, probabilities = [], {}
    for mid in (*MODELS, INCUMBENT):
        reused = mid in ("c60", INCUMBENT)
        folder = study.parent / (f"a_q5_20261108_v1_{'wc120' if mid == INCUMBENT else mid}" if reused else f"{prefix}_{mid}")
        if reused:
            fit = {"fit_seconds": read_json(folder / "metrics.json")["fit_seconds"]}
        else:
            fit = persist_fit(folder, mid, fp, context, study, train)
        if (folder / "completed.json").exists():
            for name, digest in read_json(folder / "completed.json")["artifact_sha256"].items():
                if sha256(folder / name) != digest:
                    raise ValueError("Changed completed model artifact")
        model = joblib.load(folder / "model.joblib")
        for condition in CONDITIONS:
            changed = [apply_condition(t, condition) for t in text]
            p = predict_ai(model, changed)
            probabilities[f"{mid}__{condition}"] = p
            rows.append({"model_id": mid, "condition": condition, "validation_auroc": float(roc_auc_score(labels, p)),
                         "n": len(validation), "n_changed": sum(a != b for a, b in zip(text, changed, strict=True)), "fit_seconds": fit["fit_seconds"]})
            q5.atomic_csv(predictions / f"{mid}_{condition}.csv", ["id", "label", "p_ai"],
                          ({"id": r["id"], "label": r["label"], "p_ai": format(float(v), ".17g")} for r, v in zip(validation, p, strict=True)))
        p = probabilities[f"{mid}__original"]
        if reused or (folder / "completed.json").exists():
            np.testing.assert_array_equal(p, np.array([float(r["p_ai"]) for r in read_csv(folder / "validation_predictions.csv")]))
        else:
            pt = predict_ai(model, [r["text"] for r in train])
            q5.atomic_json(folder / "metrics.json", {"train": {"n": len(train), **score([int(r["label"]) for r in train], pt)},
                           "validation": {"n": len(validation), **score(labels, p)}, "fit_seconds": fit["fit_seconds"], "warnings": fit["warnings"],
                           "feature_audit": feature_audit(model.calibrated_classifiers_[0].estimator.named_steps["features"])})
            shutil.copy2(predictions / f"{mid}_original.csv", folder / "validation_predictions.csv")
            check(folder / "validation_predictions.csv", "validation")
            q5.atomic_json(folder / "completed.json", {"artifact_sha256": {name: sha256(folder / name) for name in
                           ("config.json", "model.joblib", "model_fit.json", "metrics.json", "validation_predictions.csv", "train_ids.csv")}})
        print(f"MATRIX {mid}: original={roc_auc_score(labels, p):.6f}", flush=True)
        del model
    q5.atomic_csv(study / "comparison.csv", list(rows[0]), rows)
    return rows, probabilities


def contrast_definitions():
    contrasts = {"richness_given_length": [("c60_length_richness", 1), ("c60_length", -1)],
                 "length_only": [("c60_length", 1), ("c60", -1)],
                 "richness_only": [("c60_richness", 1), ("c60", -1)],
                 "length_given_richness": [("c60_length_richness", 1), ("c60_richness", -1)],
                 "interaction": [("c60_length_richness", 1), ("c60_length", -1), ("c60_richness", -1), ("c60", 1)]}
    for mid in MODELS[1:]:
        contrasts[f"{mid}_vs_incumbent"] = [(mid, 1), (INCUMBENT, -1)]
        contrasts[f"{mid}_vs_c60"] = [(mid, 1), ("c60", -1)]
    return contrasts


def compute_bootstrap(study, plan, labels, probabilities):
    q5.record_status(study, "paired_ablation_bootstrap")
    rng = np.random.default_rng(plan["seeds"]["bootstrap"])
    groups = [np.flatnonzero(labels == v) for v in (0, 1)]
    sampled = {key: [] for key in probabilities}
    points = {key: float(roc_auc_score(labels, p)) for key, p in probabilities.items()}
    for _ in range(plan["evaluation"]["bootstrap_replicates"]):
        indices = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        for key, p in probabilities.items():
            sampled[key].append(float(roc_auc_score(labels[indices], p[indices])))
    contrasts = {}
    for name, terms in contrast_definitions().items():
        keyed = [(f"{mid}__original", weight) for mid, weight in terms]
        draws = sum(weight * np.array(sampled[key]) for key, weight in keyed)
        contrasts[name] = {"terms": keyed, "point_delta": sum(weight * points[key] for key, weight in keyed),
                           "interval95": np.quantile(draws, [.025, .975]).tolist()}
        if len(keyed) == 2:
            contrasts[name].update(left=keyed[0][0], right=keyed[1][0])
    result = {"seed": plan["seeds"]["bootstrap"], "replicates": plan["evaluation"]["bootstrap_replicates"],
              "method": "class-stratified paired percentile bootstrap", "primary_comparison_key": "richness_given_length",
              "comparisons": contrasts, "auroc_95_intervals": {key: np.quantile(v, [.025, .975]).tolist() for key, v in sampled.items()},
              "limitations": "Conditional development intervals; omit training variation and selection adjustment; secondary contrasts descriptive."}
    q5.atomic_json(study / "bootstrap.json", result)
    return result


def length_diagnostics(study, train, validation, probabilities):
    cuts = np.quantile([len(r["text"].split()) for r in train], [1/3, 2/3])
    bins = np.digitize([len(r["text"].split()) for r in validation], cuts, right=True)
    y = np.array([int(r["label"]) for r in validation])
    rows = []
    for mid in (*MODELS, INCUMBENT):
        for b in range(3):
            mask = bins == b
            rows.append({"model_id": mid, "length_bin": b, "n": int(mask.sum()), "n_human": int(np.sum(y[mask] == 0)),
                         "n_ai": int(np.sum(y[mask] == 1)), "validation_auroc": float(roc_auc_score(y[mask], probabilities[f"{mid}__original"][mask]))})
    q5.atomic_csv(study / "length_subgroups.csv", list(rows[0]), rows)
    q5.atomic_json(study / "length_boundaries.json", {"training_word_count_tertiles": cuts.tolist()})
    return rows


def feature_diagnostics(study, train, validation):
    result = {}
    for name, rows in (("train", train), ("validation", validation)):
        text = [r["text"] for r in rows]
        length = LengthFeatures().fit_transform(text)
        richness = RichnessFeatures(window_size=50).fit_transform(text)
        values = np.column_stack([length, richness])
        result[name] = {"n": len(rows), "mean": values.mean(axis=0).tolist(), "std": values.std(axis=0).tolist(),
                        "min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist(),
                        "n_shorter_than_50_tokens": int(np.sum(np.expm1(length[:, 0]) < 49.5)),
                        "pearson_correlations": np.corrcoef(values, rowvar=False).tolist()}
    result["columns"] = list(LengthFeatures().get_feature_names_out()) + list(RichnessFeatures().get_feature_names_out())
    result["fitting_use"] = "Descriptive only. Validation summaries are never used by fitting or selection."
    q5.atomic_json(study / "numeric_feature_diagnostics.json", result)


def select_candidate(plan, cv, matrix, uncertainty, length_rows, original_cv):
    by_cv = {r["model_id"]: r for r in cv}
    chosen = max(cv[1:], key=lambda r: (r["mean_cv_auroc"], -MODELS.index(r["model_id"])))
    mid = chosen["model_id"]
    incumbent_folds = [next(r["auroc"] for r in original_cv if r["model"] == "wc120" and r["fold"] == i) for i in range(3)]
    gains = [chosen[f"fold_{i}_auroc"] - value for i, value in enumerate(incumbent_folds)]
    auc = {(r["model_id"], r["condition"]): r["validation_auroc"] for r in matrix}
    base_lengths = {r["length_bin"]: r["validation_auroc"] for r in length_rows if r["model_id"] == INCUMBENT}
    worst = max(base_lengths[r["length_bin"]] - r["validation_auroc"] for r in length_rows if r["model_id"] == mid)
    extra_loss = {condition: (auc[(INCUMBENT, condition)] - auc[(INCUMBENT, "original")]) - (auc[(mid, condition)] - auc[(mid, "original")])
                  for condition in CONDITIONS[1:]}
    rules = plan["export_rule"]
    checks = {"positive_validation_interval_lower_vs_incumbent": uncertainty["comparisons"][f"{mid}_vs_incumbent"]["interval95"][0] > 0,
              "positive_validation_interval_lower_vs_c60": uncertainty["comparisons"][f"{mid}_vs_c60"]["interval95"][0] > 0,
              "positive_mean_cv_delta_vs_incumbent": float(np.mean(gains)) > 0,
              "positive_cv_fold_count_vs_incumbent": sum(g > 0 for g in gains) >= rules["minimum_positive_cv_folds_vs_incumbent"],
              "length_guard": worst <= rules["maximum_length_bin_regression_vs_incumbent"],
              "typography_guard": max(extra_loss.values()) <= rules["maximum_extra_typography_loss_vs_incumbent"]}
    primary_gains = [by_cv["c60_length_richness"][f"fold_{i}_auroc"] - by_cv["c60_length"][f"fold_{i}_auroc"] for i in range(3)]
    primary_checks = {"positive_validation_interval_lower": uncertainty["comparisons"]["richness_given_length"]["interval95"][0] > 0,
                      "positive_mean_cv_delta": float(np.mean(primary_gains)) > 0,
                      "minimum_positive_cv_folds": sum(g > 0 for g in primary_gains) >= plan["primary_evidence_rule"]["minimum_positive_cv_folds"]}
    eligible = all(checks.values())
    return {"cv_selected_model_id": mid, "selected_model_id": mid if eligible else None, "export_eligible": eligible,
            "candidate_category": "development_improvement_candidate" if eligible else None,
            "eligibility": {mid: {"checks": checks, "eligible": eligible, "cv_fold_gains_vs_incumbent": gains,
                                 "maximum_length_bin_regression_vs_incumbent": worst, "extra_typography_loss_vs_incumbent": extra_loss}},
            "primary_comparison_key": "richness_given_length", "primary_evidence": {"checks": primary_checks, "all_passed": all(primary_checks.values()), "cv_fold_gains": primary_gains},
            "retain_incumbent_until_external_check": True, "new_kaggle_upload": False,
            "interpretation": "Incremental prediction conditional on the explicit length block and geometry; no causal or hidden-test claim."}


def export_candidate(study, prefix, selection, fp, context, train, validation):
    mid = selection["selected_model_id"]
    folder = study.parent / f"{prefix}_{mid}_full10000"
    output = ROOT / "submissions" / f"{folder.name}.csv"
    metadata = output.with_suffix(".metadata.json")
    if (folder / "completed.json").exists():
        for name, digest in read_json(folder / "completed.json")["artifact_sha256"].items():
            if sha256(ROOT / name) != digest:
                raise ValueError("Changed completed export")
        return str(output.relative_to(ROOT))
    if not folder.exists() and (output.exists() or metadata.exists()):
        raise FileExistsError("Existing output without checkpoint")
    records = train + validation
    fit = persist_fit(folder, mid, fp, context, study, records)
    model = joblib.load(folder / "model.joblib")
    test = read_csv(raw_path("test.csv"))
    text = [r["text"] for r in test]
    p = predict_ai(model, text)
    np.testing.assert_array_equal(p, predict_ai(joblib.load(folder / "model.joblib"), text))
    pt = predict_ai(model, [r["text"] for r in records])
    q5.atomic_json(folder / "metrics.json", {"train": {"n": len(records), **score([int(r["label"]) for r in records], pt)},
                   "validation": None, "fit_seconds": fit["fit_seconds"], "warnings": fit["warnings"],
                   "note": "All supplied labels including former validation rows; no held-out evaluation."})
    q5.atomic_csv(output, ["id", "label"], ({"id": r["id"], "label": format(float(v), ".17g")} for r, v in zip(test, p, strict=True)))
    count = check(output, "submission")
    q5.atomic_json(metadata, {"created_at_utc": q5.utc_now(), "model_run_id": folder.name, "source_study": study.name, "model_id": mid,
                   "C": 1, "n_train": len(records), "n_test": count, "full_refit": True, "full_refit_held_out_evaluation": None,
                   "candidate_category": selection["candidate_category"], "csv_sha256": sha256(output), "test_sha256": sha256(raw_path("test.csv")),
                   "model_sha256": fit["model_sha256"], "model_reload_prediction_max_error": 0.0,
                   "uploaded_to_kaggle": False, "eligibility": selection["eligibility"][mid]})
    paths = [folder / n for n in ("config.json", "model_fit.json", "model.joblib", "metrics.json", "train_ids.csv")] + [output, metadata]
    q5.atomic_json(folder / "completed.json", {"artifact_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in paths}})
    return str(output.relative_to(ROOT))


def run(prefix, resume=False):
    study, plan, fp, context, guards = initialize(prefix, resume)
    if (study / "status.json").exists() and read_json(study / "status.json")["stage"] == "complete":
        print("Richness study already complete; no refit or overwrite", flush=True)
        return
    started = time.perf_counter()
    q5.record_status(study, "loading_canonical_data", started_at_utc=q5.utc_now())
    try:
        train, validation = load_partitions()
        with threadpool_limits(limits=1):
            cv = run_cv(study, plan, fp, train)
            feature_diagnostics(study, train, validation)
            matrix, probabilities = evaluate_matrix(study, prefix, fp, context, train, validation)
            uncertainty = compute_bootstrap(study, plan, np.array([int(r["label"]) for r in validation]), probabilities)
            lengths = length_diagnostics(study, train, validation, probabilities)
            original_cv = read_json(study.parent / plan["reference_study"] / "cv_progress.json")
            selection = select_candidate(plan, cv, matrix, uncertainty, lengths, original_cv)
            q5.atomic_json(study / "selection.json", selection)
            if selection["export_eligible"]:
                selection["submission_csv"] = export_candidate(study, prefix, selection, fp, context, train, validation)
                q5.atomic_json(study / "selection.json", selection)
        for name, digest in guards.items():
            if sha256(ROOT / name) != digest:
                raise ValueError(f"Historical evidence changed: {name}")
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
    with (ROOT / "tmp/q5_richness.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("A richness worker is active") from exc
        run(args.prefix, args.resume)


if __name__ == "__main__":
    main()
