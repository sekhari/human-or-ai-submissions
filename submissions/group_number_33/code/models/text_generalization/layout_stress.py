"""One fixed, post hoc layout-heldout diagnostic on development rows only."""
from __future__ import annotations

import argparse
import json
import re
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from . import char_search, style_search, word_search
from .common import OUT, PROTOCOL_FILE, decision_scores, load_protocol, sha256


DESTINATION = OUT / "layout_stress"
PLAN_FILE = DESTINATION / "frozen_plan.json"


def layout_group(text):
    """Assign one mutually exclusive layout group using the original text."""
    if "\n" not in text:
        return "no_newlines"
    if re.search(r"\n\s*\n", text):
        return "blank_paragraph_multiblock"
    return "multiline_no_blank_paragraphs"


def fixed_candidates():
    requested = [("style", style_search, "style_hgb31"),
                 ("word", word_search, "raw_punct_word12_lr_C4"),
                 ("char", char_search, "char_min_36_lr8")]
    return [dict(family=family, config=next(
        config for config in module.candidate_configs() if config["name"] == name
    )) for family, module, name in requested]


def create_plan():
    if PLAN_FILE.exists():
        raise FileExistsError("The diagnostic plan is already frozen")
    development, protocol = load_protocol()
    groups = development.text.map(layout_group)
    counts = []
    for name in sorted(groups.unique()):
        labels = development.label[groups == name]
        counts.append(dict(group=name, rows=len(labels),
                           class_0=int((labels == 0).sum()),
                           class_1=int((labels == 1).sum())))
    eligible = [row for row in counts if row["rows"] >= 1000
                and row["class_0"] >= 200 and row["class_1"] >= 200]
    if not eligible:
        raise ValueError("No layout group satisfies the predeclared eligibility")
    chosen = sorted(eligible, key=lambda row: (-row["rows"], row["group"]))[0]["group"]
    validation = np.flatnonzero(groups.to_numpy() == chosen)
    training = np.flatnonzero(groups.to_numpy() != chosen)
    if set(development.label.iloc[training]) != {0, 1}:
        raise ValueError("The complementary training group needs both classes")
    plan = dict(
        diagnostic="Post hoc layout-heldout stress diagnostic; not an independent unseen estimate",
        group_rule="no newline; otherwise blank paragraph if regex \\n\\s*\\n matches; otherwise multiline without blank paragraphs",
        selection_rule="Largest group with >=1000 rows and >=200 examples of each class; lexical group name breaks ties",
        group_counts=counts,
        selected_validation_group=chosen,
        train=training.tolist(),
        validation=validation.tolist(),
        train_class_counts={str(k): int(v) for k, v in development.label.iloc[training].value_counts().items()},
        train_sha256=protocol["train_sha256"],
        source_protocol_sha256=sha256(PROTOCOL_FILE),
        configs=fixed_candidates(),
        plan_created_before_any_model_fit=True,
        no_outer_or_test_data_used=True,
    )
    DESTINATION.mkdir(parents=True, exist_ok=True)
    assignments = pd.DataFrame(dict(
        development_index=np.arange(len(development)),
        original_row_index=protocol["development_indices"],
        id=development.id,
        label=development.label,
        layout_group=groups,
        split=np.where(groups == chosen, "validation", "train"),
    ))
    assignments.to_csv(DESTINATION / "group_ids.csv", index=False)
    PLAN_FILE.write_text(json.dumps(plan, indent=2))
    print(json.dumps({key: plan[key] for key in ["diagnostic", "group_counts", "selected_validation_group", "train_class_counts", "configs"]}, indent=2), flush=True)


def run():
    plan = json.loads(PLAN_FILE.read_text())
    development, protocol = load_protocol()
    if plan["source_protocol_sha256"] != sha256(PROTOCOL_FILE):
        raise ValueError("The source protocol changed after diagnostic freeze")
    if plan["train_sha256"] != protocol["train_sha256"]:
        raise ValueError("The dataset changed after diagnostic freeze")
    training, validation = np.asarray(plan["train"]), np.asarray(plan["validation"])
    if set(training) & set(validation) or set(training) | set(validation) != set(range(len(development))):
        raise ValueError("Invalid diagnostic partition")
    builders = dict(style=style_search.build_pipeline,
                    word=word_search.build_pipeline, char=char_search.build_pipeline)
    records = []
    with threadpool_limits(limits=2):
        for entry in plan["configs"]:
            config, family = entry["config"], entry["family"]
            model = builders[family](config)
            started = perf_counter()
            model.fit(development.text.iloc[training], development.label.iloc[training])
            fit_seconds = perf_counter() - started
            train_scores = decision_scores(model, development.text.iloc[training])
            scores = decision_scores(model, development.text.iloc[validation])
            mean, std = float(train_scores.mean()), float(train_scores.std())
            if std <= 0:
                raise ValueError("Constant fit scores")
            labels = development.label.iloc[validation].to_numpy()
            record = dict(candidate=config["name"], family=family, config=config,
                          layout_auroc=float(roc_auc_score(labels, scores)),
                          layout_fit_seconds=fit_seconds,
                          layout_train_score_mean=mean, layout_train_score_std=std,
                          validation_group=plan["selected_validation_group"],
                          validation_rows=len(validation), fit_rows=len(training))
            np.savez_compressed(
                DESTINATION / f'{config["name"]}_predictions.npz',
                layout_ids=development.id.iloc[validation].to_numpy(),
                layout_labels=labels, layout_scores=scores,
                layout_z=(scores - mean) / std,
                layout_development_indices=validation,
                layout_original_indices=np.asarray(protocol["development_indices"])[validation],
            )
            records.append(record)
            (DESTINATION / "results.json").write_text(json.dumps(records, indent=2, allow_nan=False))
            pd.DataFrame([{key: value for key, value in row.items() if key != "config"}
                          for row in records]).to_csv(DESTINATION / "scores.csv", index=False)
            print(f'{config["name"]}: layout AUROC={record["layout_auroc"]:.6f}, fit_seconds={fit_seconds:.2f}', flush=True)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    create_plan() if args.plan_only else run()
