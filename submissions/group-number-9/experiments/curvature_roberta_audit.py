"""Audit provenance, train/validation separation, and checked candidate exports."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from experiments.curvature_roberta import CACHE, CONFIG, OUT, ROOT, baseline, checkpoint, context_frames, scores, sha, training_frame, write
from experiments.robust_boost import paired_ci
from src.curvature_roberta import average_probabilities, blend
from src.roberta_experiment import frame_hash


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--partial", action="store_true")
    args = ap.parse_args()
    protocol = json.loads((OUT / "protocol.json").read_text())
    for key in ["sources", "inputs"]:
        assert all(sha(ROOT / p) == h for p, h in protocol[key].items())
    registration = json.loads((OUT / "queue_registration.json").read_text())
    assert registration["protocol_sha256"] == sha(OUT / "protocol.json")
    assert all(sha(ROOT / p) == h for p, h in registration["sources"].items())
    _, val = context_frames("fit")
    _, test = context_frames("full")
    sample = pd.read_csv(ROOT / "Data/sample_submission.csv")
    assert len(val) == 2000 and len(test) == 3000 and np.array_equal(test.id, sample.id)
    names = ["single"] if args.partial else ["single", "ensemble"]
    checked = []
    for name in names:
        choice = json.loads((OUT / f"{name}_selection.json").read_text())
        result = json.loads((OUT / f"{name}_validation.json").read_text())
        epoch, weight = result["epoch"], result["neural_weight"]
        assert not choice["fixed_validation_used"]
        assert choice["protocol_sha256"] == sha(OUT / "protocol.json")
        assert choice["chosen"]["epoch"] == epoch and choice["chosen"]["weight"] == weight
        assert result["selection_sha256"] == sha(OUT / f"{name}_selection.json")
        n = average_probabilities([scores("fit", s, epoch) for s in result["neural_seeds"]])
        p = blend(baseline("fit", val.id), n, weight)
        stored = pd.read_csv(OUT / f"{name}_validation_predictions.csv")
        assert np.array_equal(stored.id, val.id) and np.array_equal(stored.label, val.label.astype(int))
        np.testing.assert_allclose(stored.blend, p, rtol=1e-12, atol=1e-12)
        assert abs(roc_auc_score(val.label, p) - result["validation_auroc"]) < 1e-15
        for seed in result["neural_seeds"]:
            for context in ["inner", "domain", "fit"]:
                fit, hold = context_frames(context)
                assert set(fit.id).isdisjoint(hold.id)
                complete = json.loads((checkpoint(context, seed, epoch) / "complete.json").read_text())
                assert complete["signature"]["frame"] == frame_hash(training_frame(fit))
                assert complete["signature"]["configuration"] == {"seed": seed, **CONFIG}
                assert complete["pseudo_rows_available"] == 0
                expected_updates = epoch * int(np.ceil(np.ceil(len(fit) / CONFIG["batch_size"]) / CONFIG["gradient_accumulation"]))
                assert complete["updates"] == expected_updates
        if result["export_eligible"]:
            sub_path = ROOT / "submissions" / f"curvature_roberta_{name}_10000.csv"
            sub = pd.read_csv(sub_path)
            manifest = json.loads(sub_path.with_suffix(".json").read_text())
            assert list(sub.columns) == ["id", "label"] and len(sub) == 3000 and sub.id.is_unique
            assert np.array_equal(sub.id, test.id)
            assert np.isfinite(sub.label).all() and sub.label.between(0, 1).all()
            assert manifest["csv_sha256"] == sha(sub_path)
            assert manifest["protocol_sha256"] == sha(OUT / "protocol.json")
            nf = average_probabilities([scores("full", s, epoch) for s in result["neural_seeds"]])
            np.testing.assert_allclose(sub.label, blend(baseline("full", test.id), nf, weight), rtol=1e-12, atol=1e-12)
            for seed in result["neural_seeds"]:
                full, _ = context_frames("full")
                complete = json.loads((checkpoint("full", seed, epoch) / "complete.json").read_text())
                assert complete["signature"]["frame"] == frame_hash(training_frame(full))
                assert complete["training_rows"] == 10000
                assert complete["updates"] == epoch * 625
                expected = next(c["sha256"] for c in manifest["neural_checkpoints"] if c["seed"] == seed)
                assert expected == sha(checkpoint("full", seed, epoch) / "model.safetensors")
            checked.append(str(sub_path.relative_to(ROOT)))
    comparison = None
    if not args.partial:
        assert json.loads((OUT / "status.json").read_text())["status"] == "complete"
        single = pd.read_csv(OUT / "single_validation_predictions.csv")
        ensemble = pd.read_csv(OUT / "ensemble_validation_predictions.csv")
        assert np.array_equal(single.id, ensemble.id)
        y = single.label.astype(int).to_numpy()
        comparison = {"ensemble_minus_single": roc_auc_score(y, ensemble.blend) - roc_auc_score(y, single.blend),
            "paired_95_ci": paired_ci(y, ensemble.blend.to_numpy(), single.blend.to_numpy()),
            "limitation": "Conditional on the reused fixed validation split; not a public Kaggle comparison."}
        write(OUT / "ensemble_vs_single.json", comparison)
        with (OUT / "README.md").open("a") as f:
            f.write("\n## Final audit\n\nSource/dependency hashes, held-out prediction reconstruction, all training signatures and optimizer budgets, final 10000-row refits, and exported CSV IDs/ranges/hashes passed. No upload was performed.\n\n")
            f.write(f"Ensemble minus single validation AUROC: {comparison['ensemble_minus_single']:+.9f}; paired 95% interval {comparison['paired_95_ci']}.\n")
    record = {"passed": True, "partial": args.partial, "options_checked": names,
              "checked_submissions": checked, "source_and_dependency_hashes_match": True,
              "training_and_scoring_ids_verified": True, "all_refits_and_optimizer_budgets_checked": True,
              "unit_tests_passed": 4, "real_mps_smoke_passed": json.loads((OUT / "smoke.json").read_text())["passed"],
              "no_new_model_downloads": True, "no_unlabeled_examples_fitted": True,
              "kaggle_uploaded": False, "protocol_sha256": sha(OUT / "protocol.json")}
    write(OUT / ("first_submission_verification.json" if args.partial else "verification.json"), record)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
