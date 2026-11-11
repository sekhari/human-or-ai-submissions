"""Generate a local Q1 Kaggle candidate after within-Q1 selection, without uploading.

Full-data refits get their own immutable artifacts and never overwrite the honest
8000-row checkpoints or evaluate on their former validation rows as held out.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from human_ai.check_predictions import check
from human_ai.data import ROOT, load_partitions, raw_path, read_csv, sha256, write_csv
from .run_q1 import make_model, predict_ai, provenance, save_json, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="An existing selected 8000-row Q1 run")
    parser.add_argument("--output", required=True)
    parser.add_argument("--refit-all", action="store_true", help="Fit selected settings on all 10000 labeled rows")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        raise ValueError("run-id must be a directory name")
    origin = ROOT / "results/supervised" / args.run_id
    config = json.loads((origin / "config.json").read_text())
    study = origin.parent / config["study_id"]
    selection = json.loads((study / "selection.json").read_text())
    if selection["winner_run_id"] != args.run_id:
        raise ValueError("This command exports the selected Q1 candidate only")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    receipt = output.with_suffix(".metadata.json")
    if receipt.exists():
        raise FileExistsError(receipt)
    if sha256(Path(__file__).with_name("run_q1.py")) != config["provenance"]["source_sha256"]["experiments/supervised/run_q1.py"]:
        raise ValueError("Model implementation changed; use the study snapshot or a new recorded experiment")
    # Read test passages only for prediction, never for feature fitting or selection.
    test = read_csv(raw_path("test.csv"))
    warnings.filterwarnings("error", category=ConvergenceWarning)
    with threadpool_limits(limits=1):
        if args.refit_all:
            artifact = origin.parent / f"{args.run_id}_full10000"
            if artifact.exists():
                raise FileExistsError(artifact)
            artifact.mkdir()
            train, validation = load_partitions()
            records = train + validation
            text = [r["text"] for r in records]
            y = np.array([int(r["label"]) for r in records])
            model = make_model(config["plan"], config["family"], config["chosen_value"])
            full_config = {
                "run_id": artifact.name, "selected_from": args.run_id,
                "status": "Q1 candidate; team final choice remains open",
                "plan": config["plan"], "family": config["family"], "chosen_value": config["chosen_value"],
                "n_train": len(records), "training_order": "canonical train IDs followed by canonical validation IDs",
                "supervised_labels": "supplied train.csv only; no new labels",
                "selection_source": f"../{study.name}/selection.json",
                "test_sha256": sha256(raw_path("test.csv")), "provenance": provenance(),
                "predict_source_sha256": sha256(Path(__file__)),
                "held_out_evaluation": None,
            }
            save_json(artifact / "config.json", full_config)
            write_csv(artifact / "train_ids.csv", ["id"], ({"id": r["id"]} for r in records))
            shutil.copy2(__file__, artifact / "predict.py")
            start = time.perf_counter()
            model.fit(text, y)
            seconds = time.perf_counter() - start
            p_train = predict_ai(model, text)
            save_json(artifact / "metrics.json", {
                "train": {"n": len(records), **score(y, p_train)},
                "validation": None, "fit_seconds": seconds,
                "timing_definition": "wall seconds, features and sigmoid calibration included",
                "note": "All 10000 labels used; no honest held-out score for this refit",
            })
            joblib.dump(model, artifact / "model.joblib")
        else:
            artifact = origin
            model = joblib.load(origin / "model.joblib")
        p_test = predict_ai(model, [r["text"] for r in test])
    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output, ["id", "label"],
              ({"id": r["id"], "label": format(float(p), ".17g")}
               for r, p in zip(test, p_test, strict=True)))
    check(output, "submission")
    save_json(receipt, {"model_run_id": artifact.name, "source_selection_run_id": args.run_id,
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "csv_sha256": sha256(output), "n_test": len(test),
                        "full_refit": args.refit_all, "uploaded_to_kaggle": False})
    print(f"Validated local candidate: {output} ({len(test)} IDs); no upload performed")


if __name__ == "__main__":
    main()
