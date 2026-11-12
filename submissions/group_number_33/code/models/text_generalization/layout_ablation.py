"""One fixed no-layout style model on the already frozen layout stress split."""

from __future__ import annotations

import json
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from .common import OUT, PROTOCOL_FILE, decision_scores, load_protocol, sha256
from .style_ablation import build_pipeline


DESTINATION = OUT / "layout_ablation"
SOURCE_LAYOUT_PLAN = OUT / "layout_stress/frozen_plan.json"
SOURCE_STYLE_PREREGISTRATION = OUT / "style_ablation/preregistered_feature_lists.json"
PLAN_FILE = DESTINATION / "frozen_plan.json"
CANDIDATE_NAME = "style_hgb31_no_layout"


def freeze_plan() -> dict:
    """Copy exactly the existing stress indices and preregistered no-layout config."""
    source = json.loads(SOURCE_LAYOUT_PLAN.read_text())
    style_plan = json.loads(SOURCE_STYLE_PREREGISTRATION.read_text())
    config = next(config for config in style_plan["configs"] if config["name"] == CANDIDATE_NAME)
    plan = {
        "diagnostic": "Post hoc layout-heldout extension with one previously fixed feature ablation",
        "source_layout_plan_sha256": sha256(SOURCE_LAYOUT_PLAN),
        "source_style_preregistration_sha256": sha256(SOURCE_STYLE_PREREGISTRATION),
        "source_protocol_sha256": source["source_protocol_sha256"],
        "train_sha256": source["train_sha256"],
        "train": source["train"],
        "validation": source["validation"],
        "selected_validation_group": source["selected_validation_group"],
        "config": config,
        "model_policy": "Exactly the prior 106-feature no_layout HGB31; no parameter or feature changes",
        "no_outer_or_test_data_used": True,
    }
    DESTINATION.mkdir(parents=True, exist_ok=True)
    if PLAN_FILE.exists():
        if json.loads(PLAN_FILE.read_text()) != plan:
            raise ValueError("Existing ablation plan differs from the source plans")
    else:
        PLAN_FILE.write_text(json.dumps(plan, indent=2))
    return plan


def run() -> dict:
    if (DESTINATION / "results.json").exists():
        raise FileExistsError("The single layout-ablation fit already has saved results")
    plan = freeze_plan()
    development, protocol = load_protocol()
    if plan["source_protocol_sha256"] != sha256(PROTOCOL_FILE):
        raise ValueError("The source protocol changed after the layout split was frozen")
    if plan["train_sha256"] != protocol["train_sha256"]:
        raise ValueError("The training dataset changed after the layout split was frozen")
    training = np.asarray(plan["train"])
    validation = np.asarray(plan["validation"])
    if set(training) & set(validation) or set(training) | set(validation) != set(range(len(development))):
        raise ValueError("Invalid frozen layout partition")
    config = plan["config"]
    model = build_pipeline(config)
    print(json.dumps({"candidate": config["name"], "kept_features": len(config["kept_features"]),
                      "fit_rows": len(training), "validation_rows": len(validation)}), flush=True)
    with threadpool_limits(limits=2):
        started = perf_counter()
        model.fit(development.text.iloc[training], development.label.iloc[training])
        fit_seconds = perf_counter() - started
        train_scores = decision_scores(model, development.text.iloc[training])
        scores = decision_scores(model, development.text.iloc[validation])
    mean, std = float(train_scores.mean()), float(train_scores.std())
    if std <= 0:
        raise ValueError("Constant fit scores")
    labels = development.label.iloc[validation].to_numpy()
    record = {
        "candidate": config["name"],
        "family": "style_ablation",
        "config": config,
        "layout_auroc": float(roc_auc_score(labels, scores)),
        "layout_fit_seconds": fit_seconds,
        "layout_train_score_mean": mean,
        "layout_train_score_std": std,
        "validation_group": plan["selected_validation_group"],
        "validation_rows": len(validation),
        "fit_rows": len(training),
        "source_layout_plan_sha256": plan["source_layout_plan_sha256"],
    }
    np.savez_compressed(
        DESTINATION / f'{config["name"]}_predictions.npz',
        layout_ids=development.id.iloc[validation].to_numpy(),
        layout_labels=labels,
        layout_scores=scores,
        layout_z=(scores - mean) / std,
        layout_development_indices=validation,
        layout_original_indices=np.asarray(protocol["development_indices"])[validation],
    )
    (DESTINATION / "results.json").write_text(json.dumps([record], indent=2, allow_nan=False))
    pd.DataFrame([{key: value for key, value in record.items() if key != "config"}]).to_csv(
        DESTINATION / "scores.csv", index=False
    )
    print(f'{config["name"]}: layout AUROC={record["layout_auroc"]:.6f}, fit_seconds={fit_seconds:.2f}', flush=True)
    return record


if __name__ == "__main__":
    run()
