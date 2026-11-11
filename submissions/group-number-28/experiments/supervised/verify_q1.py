"""Audit recorded Q1 evidence, optionally retraining selected settings from scratch."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from human_ai.check_predictions import check
from human_ai.data import ROOT, load_partitions, read_csv, sha256
from .run_q1 import FAMILIES, make_model, predict_ai, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", required=True, help="NEW audit JSON; never overwrite")
    parser.add_argument("--refit", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    base = ROOT / "results/supervised"
    study = base / f"{args.prefix}_study"
    plan = json.loads((study / "q1_plan.json").read_text())
    context = json.loads((study / "provenance.json").read_text())
    # Verify both the current implementation and the frozen copy used in this run.
    for name in ("run_q1.py", "q1_plan.json", "requirements.txt"):
        expected = context["source_sha256"][f"experiments/supervised/{name}"]
        assert sha256(study / name) == expected, name
        assert sha256(Path(__file__).with_name(name)) == expected, name
    train, val = load_partitions()
    text = [r["text"] for r in train]
    val_text = [r["text"] for r in val]
    y = np.array([int(r["label"]) for r in train])
    val_y = np.array([int(r["label"]) for r in val])
    folds = read_csv(study / "inner_folds.csv")
    assert [r["id"] for r in folds] == [r["id"] for r in train]
    inner = StratifiedKFold(3, shuffle=True, random_state=plan["seeds"]["inner_cv"])
    for fold, (_, held) in enumerate(inner.split(text, y)):
        assert all(int(folds[i]["inner_validation_fold"]) == fold for i in held)
    chosen = json.loads((study / "chosen_parameters.json").read_text())
    cv = read_csv(study / "cv_results.csv")
    summary = read_csv(study / "comparison.csv")
    audit = {"inner_folds_match_training_only_split": True, "source_hashes_match": True,
             "reference_sizes": {"train": len(train), "validation": len(val)}, "models": {}}
    with threadpool_limits(limits=1):
        for family in FAMILIES:
            candidates = [r for r in cv if r["model"] == family]
            best = max(candidates, key=lambda r: float(r["mean_cv_auroc"]))
            assert float(best["value"]) == chosen[family]
            folder = base / f"{args.prefix}_{family}"
            count = check(folder / "validation_predictions.csv", "validation")
            rows = read_csv(folder / "validation_predictions.csv")
            assert [r["id"] for r in rows] == [r["id"] for r in val]
            recorded = np.array([float(r["p_ai"]) for r in rows])
            model = joblib.load(folder / "model.joblib")
            restored = predict_ai(model, val_text)
            np.testing.assert_allclose(restored, recorded, atol=1e-14, rtol=0)
            metrics = json.loads((folder / "metrics.json").read_text())
            auc = float(roc_auc_score(val_y, recorded))
            assert abs(auc - metrics["validation"]["auroc"]) < 1e-14
            assert abs(auc - float(next(r for r in summary if r["model"] == family)["validation_auroc"])) < 1e-14
            # Full-fit vocabulary must be learned from the canonical training rows.
            fitted = model.calibrated_classifiers_[0].estimator if family == "svm" else model
            tfidf = fitted.named_steps["tfidf"]
            analyze = tfidf.build_analyzer()
            train_features = {feature for passage in text for feature in analyze(passage)}
            assert set(tfidf.vocabulary_) <= train_features
            result = {"n_predictions": count, "validation_auroc": auc,
                      "selected_by_inner_cv": True, "saved_model_prediction_max_error": float(np.max(abs(restored - recorded))),
                      "vocabulary_features_present_in_training": True}
            if args.refit:
                start = time.perf_counter()
                fresh = make_model(plan, family, chosen[family]).fit(text, y)
                fresh_p = predict_ai(fresh, val_text)
                error = float(np.max(np.abs(fresh_p - recorded)))
                np.testing.assert_allclose(fresh_p, recorded, atol=1e-10, rtol=0)
                result.update({"fresh_refit_prediction_max_error": error,
                               "fresh_refit_plus_prediction_seconds": time.perf_counter() - start})
            audit["models"][family] = result
    audit["all_checks_passed"] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
