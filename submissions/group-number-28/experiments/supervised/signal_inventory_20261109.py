"""Inventory already saved model signals; zero fits and no candidate export.

Fixed 50/50 blends with the two existing nonlinear SVD models are diagnostic
only. The reused development labels have already informed many model choices;
neither these blends nor rank averaging establish hidden-test improvement.
"""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

from human_ai.data import ROOT, load_partitions, read_csv, sha256, write_csv


RUNS = {
    **{name: f"a_q1_expanded_20261108_v2_{name}"
       for name in ("lr", "nb", "svm", "tree", "forest", "boost", "mlp", "lr_svd")},
    "c60": "a_q5_20261108_v1_c60",
    "wc120": "a_q5_20261108_v1_wc120",
    "w120": "a_q5_20261108_v1_w120",
    "quotes": "a_q5_quotes_20261108_v3_wc120_norm",
    "richness": "a_q5_richness_20261108_v4_c60_richness",
}


def run(output):
    output = ROOT / output
    if output.exists():
        raise FileExistsError(output)
    _, validation = load_partitions()
    ids = [r["id"] for r in validation]
    truth = {r["id"]: r["label"] for r in validation}
    y = np.array([int(truth[i]) for i in ids])
    predictions, input_hashes = {}, {}
    for name, run_id in RUNS.items():
        path = ROOT / "results/supervised" / run_id / "validation_predictions.csv"
        rows = read_csv(path)
        by_id = {r["id"]: r for r in rows}
        if len(rows) != len(ids) or len(by_id) != len(ids) or set(by_id) != set(ids):
            raise ValueError(f"ID contract failure: {name}")
        if any(by_id[i]["label"] != truth[i] for i in ids):
            raise ValueError(f"Label disagreement: {name}")
        p = np.array([float(by_id[i]["p_ai"]) for i in ids])
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError(f"Invalid probabilities: {name}")
        predictions[name] = p
        input_hashes[str(path.relative_to(ROOT))] = sha256(path)

    baseline = predictions["wc120"]
    baseline_auc = float(roc_auc_score(y, baseline))
    baseline_errors = (baseline >= .5) != y
    baseline_pair_correct = baseline[y == 1, None] > baseline[None, y == 0]
    baseline_pair_wrong = baseline[y == 1, None] < baseline[None, y == 0]
    pairs = int(np.sum(y == 1) * np.sum(y == 0))
    rows = []
    for name, p in predictions.items():
        errors = (p >= .5) != y
        correct_pairs = p[y == 1, None] > p[None, y == 0]
        wrong_pairs = p[y == 1, None] < p[None, y == 0]
        rows.append({
            "model": name, "source_run": RUNS[name],
            "validation_auroc": float(roc_auc_score(y, p)),
            "auroc_minus_wc120": float(roc_auc_score(y, p) - baseline_auc),
            "spearman_vs_wc120": float(spearmanr(baseline, p).statistic),
            "threshold_half_errors": int(errors.sum()),
            "threshold_half_wc120_errors_corrected": int(np.sum(baseline_errors & ~errors)),
            "threshold_half_wc120_correct_newly_wrong": int(np.sum(~baseline_errors & errors)),
            "strict_wrong_wc120_ai_human_pairs": int(baseline_pair_wrong.sum()),
            "strict_wrong_wc120_pairs_candidate_gets_right": int(np.sum(baseline_pair_wrong & correct_pairs)),
            "strict_correct_wc120_pairs_candidate_gets_wrong": int(np.sum(baseline_pair_correct & wrong_pairs)),
            "total_ai_human_pairs": pairs,
        })

    # Structural representatives fixed before their blend scores are computed:
    # the existing SVD128 MLP and SVD128 histogram boosting model. No weight grid.
    blends = []
    for name in ("mlp", "boost"):
        p = predictions[name]
        for mode, values in (
            ("probability_mean_50_50", (baseline + p) / 2),
            ("within_validation_percentile_rank_mean_50_50",
             (rankdata(baseline, method="average") + rankdata(p, method="average")) / (2 * len(y))),
        ):
            auc = float(roc_auc_score(y, values))
            blends.append({"baseline": "wc120", "second_model": name,
                           "rule": mode, "validation_auroc": auc,
                           "delta_vs_wc120": auc - baseline_auc,
                           "candidate_export_allowed": False})

    correlations = {a: {b: float(spearmanr(x, predictions[b]).statistic)
                        for b in predictions} for a, x in predictions.items()}
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": sha256(__file__), "input_sha256": input_hashes,
        "packages": {name: version(name) for name in ("numpy", "scipy", "scikit-learn")},
        "n_validation": len(ids), "n_human": int(np.sum(y == 0)), "n_ai": int(np.sum(y == 1)),
        "models_fitted": 0, "test_text_accessed": False, "exports": 0,
        "existing_mlp_representation": "Original word TF-IDF -> TruncatedSVD128 -> StandardScaler -> MLP128; not a pretrained contextual encoder.",
        "limitations": [
            "Reused development labels; all comparisons and blends are exploratory, with no selection correction.",
            "Threshold-half disagreement depends on probability calibration and is secondary to ranking evidence.",
            "Strict-pair counts ignore ties; they describe ranking complementarity, not an implementable oracle.",
            "Rank averaging uses the development score distribution; any production version needs a separately declared transform and evaluation.",
            "No training-only OOF blend selection, topic-separated blend evaluation or hidden-test gain has been established.",
            "The fixed blend scores are not new submission candidates and no weights are tuned here.",
        ],
    }
    output.mkdir(parents=True)
    write_csv(output / "model_inventory.csv", list(rows[0]), rows)
    write_csv(output / "fixed_blends.csv", list(blends[0]), blends)
    (output / "rank_correlations.json").write_text(json.dumps(correlations, indent=2, allow_nan=False) + "\n")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"baseline_auc": baseline_auc, "inventory": rows, "fixed_blends": blends}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/supervised/signal_inventory_20261109")
    run(parser.parse_args().output)
