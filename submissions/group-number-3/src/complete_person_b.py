"""Run fresh Q2 learning curves and Q3 full-refit pseudo-label experiments.

    python src/complete_person_b.py
"""
import argparse
import json
from pathlib import Path

from threadpoolctl import threadpool_limits

from data import ROOT, get_split, load_splits, load_train
from person_b import sha256, validate_split_data
from learning_curve import MODEL_NAMES, SEEDS, SIZES, main as learning_curves
from pseudo_label import THRESHOLDS, main as pseudo_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--rerun-style", action="store_true", help="Also fit style-model learning curves.")
    args = parser.parse_args()
    config = json.loads((ROOT / "config/person_b.json").read_text())
    hashes = {name: sha256(ROOT / name) for name in config["expected_sha256"]}
    if hashes != config["expected_sha256"]:
        raise ValueError("Official input/shared split hash mismatch.")
    train, splits = load_train(), load_splits()
    validate_split_data(train, splits)
    tr, va = get_split(train, splits, "random_fold")
    if (len(tr), len(va)) != (8000, 2000):
        raise ValueError("Unexpected fixed split sizes.")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=config["threads"]):
        curves = learning_curves(output, rerun_style=args.rerun_style)
        pseudo = pseudo_labels(output)

    model_names = MODEL_NAMES if args.rerun_style else ["tfidf_logreg"]
    expected_curves = {(name, size, seed) for name in model_names for size in SIZES
                       for seed in (SEEDS if size < len(tr) else [0])}
    actual_curves = set(curves[["model", "n_train", "seed"]].itertuples(index=False, name=None))
    expected_pseudo = {(teacher, threshold) for teacher in ("self", "style_gbm")
                       for threshold in THRESHOLDS}
    actual_pseudo = set(pseudo.loc[pseudo.teacher.ne("none (control)"),
                                  ["teacher", "threshold"]].itertuples(index=False, name=None))
    controls = pseudo.loc[pseudo.teacher.eq("none (control)")]
    checks = {
        "inputs_unchanged": all(sha256(ROOT / name) == digest for name, digest in hashes.items()),
        "curve_settings_complete": actual_curves == expected_curves and len(curves) == len(expected_curves),
        "curve_scores_valid": curves[["train_auroc", "val_auroc"]].apply(
            lambda column: column.between(0, 1)).all().all(),
        "pseudo_settings_complete": actual_pseudo == expected_pseudo and len(pseudo) == 1 + len(expected_pseudo),
        "one_labeled_only_control": len(controls) == 1 and controls.n_added.eq(0).all(),
        "pseudo_scores_valid": pseudo.val_auroc.between(0, 1).all(),
    }
    checks = {name: bool(passed) for name, passed in checks.items()}
    (output / "person_b_completion_manifest.json").write_text(json.dumps({
        "input_sha256": hashes, "q2_models": model_names, "q2_seeds": SEEDS,
        "q3_feature_policy": "refit_full_tfidf_pipeline", "checks": checks,
        "all_checks_passed": all(checks.values()), "results_source": "fresh fits in this run"
    }, indent=2) + "\n")
    print(json.dumps(checks, indent=2), flush=True)
    if not all(checks.values()):
        raise SystemExit("Experiment output checks failed.")


if __name__ == "__main__":
    main()
