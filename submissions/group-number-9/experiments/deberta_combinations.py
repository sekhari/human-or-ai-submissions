"""Train-only selection for DeBERTa/boosting combinations and a text/style MLP.

    python -m experiments.deberta_combinations --stage all

Each GPU extraction runs in a separate FP32 process. Cached representations and
head weights stay ignored. Existing experiment files are preserved.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.deberta import device_for, load_model, load_tokenizer
from src.deberta_hybrid import extract, head_probabilities, prepare_inputs, train_head
from src.features import build_extended_features
from src.split import VAL_IDS_PATH, load_train_val

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/deberta_combinations"
CACHE = ROOT / ".cache/deberta_combinations"
PREVIOUS = ROOT / "results/deberta_v3_mac_isolated"
WEIGHTS = ROOT / ".cache/deberta_v3_mac_isolated"
SEEDS = [42, 43, 44]
EPOCHS = [5, 10, 20, 30]
HEAD_GRID = [{"width": width, "weight_decay": wd, "lr": 0.001, "dropout": 0.2}
             for width in [64, 128] for wd in [0.001, 0.01]]
BLEND_WEIGHTS = [0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 1]


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def partitions():
    train, val = load_train_val()
    ids = pd.read_csv(PREVIOUS / "inner_split_ids.csv").set_index("id")
    assert set(ids.index) == set(train.id) and ids.index.is_unique
    fitting = train.id.map(ids.partition).eq("inner_fit").to_numpy()
    assert fitting.sum() == 6400 and (~fitting).sum() == 1600
    assert set(train.id).isdisjoint(val.id)
    return train, val, fitting


def protocol():
    value = {"validation_ids_sha256": sha(VAL_IDS_PATH), "train_csv_sha256": sha(ROOT / "Data/train.csv"),
             "inner_split_sha256": sha(PREVIOUS / "inner_split_ids.csv"),
             "reference": "v2_word_masked", "reference_validation_auroc": 0.999288,
             "head_grid": HEAD_GRID, "head_epoch_grid": EPOCHS, "head_seeds": SEEDS,
             "blend_weights": BLEND_WEIGHTS,
             "stack_models": "probability/logit blends; 2-score logistic C=0.01,0.1,1,10; style+score HGB depth=2,3 and L2=1,10",
             "selection": "Only original 8,000 training rows. Base models fit 6,400; their held-out scores on 1,600 select blends and train/select combiners (4-fold CV). The original encoder epoch selection also used these 1,600 labels, so combiner CV is conditional on that earlier selection, not an independent estimate.",
             "hybrid": "Frozen, already fine-tuned DeBERTa pooled vector (768) plus 58 engineered features. Separate train-only normalizers and learned projections feed a fusion MLP. Text/style ablations receive the same search. This stage does not jointly fine-tune the encoder.",
             "final_fit": "Reused 8,000-row encoder and boosting reference. Selected fusion heads refit on 8,000 rows. Stack learned from held-out 1,600 scores is applied after refitting the base models on 8,000, as in holdout blending; score-distribution changes remain a limitation.",
             "evaluation": "Fixed 2,000-row validation used after selection, with paired bootstrap differences. This split has been viewed in earlier experiments; it is not a fresh final test.",
             "submission_gate": "Generate candidate files only for methods whose fixed-validation AUROC exceeds the frozen boosting reference. No Kaggle upload or Git push."}
    path = OUT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == value, "Protocol changed: use a new output directory."
    else:
        write(path, value)


def cache_features(kind):
    train, val, fitting = partitions()
    checkpoint = WEIGHTS / ("tune/best" if kind == "inner" else "fit/final")
    if kind == "test":
        from src.data import load_split
        frame = load_split("test")
    else:
        frame = train if kind == "inner" else pd.concat([train, val], ignore_index=True)
    path = CACHE / f"{kind}_features.npz"
    signature = {"checkpoint_sha256": sha(checkpoint / "model.safetensors"),
                 "ids_sha256": hashlib.sha256(frame.id.to_numpy().tobytes()).hexdigest(),
                 "train_csv_sha256": sha(ROOT / "Data/train.csv"), "max_length": 512,
                 "feature_source_sha256": sha(ROOT / "src/features.py"),
                 "extract_source_sha256": sha(ROOT / "src/deberta_hybrid.py")}
    metadata = OUT / f"{kind}_features.json"
    if path.exists():
        assert json.loads(metadata.read_text())["signature"] == signature
        print(f"Reusing {kind} encoder cache.", flush=True)
        return
    device = device_for("mps")
    model = load_model(42, checkpoint).to(device)
    started = time.perf_counter()
    def progress(row):
        write(OUT / f"{kind}_extraction_progress.json", row)
        print(f"{kind} encoder: {row['rows']}/{row['total_rows']} rows, {row['elapsed_s']:.0f}s", flush=True)
    vectors, scores = extract(model, frame, load_tokenizer(), device, progress=progress)
    style = build_extended_features(frame)
    assert style.shape[1] == 58
    temporary = path.with_name(path.stem + "_temporary.npz")
    np.savez_compressed(temporary, ids=frame.id.to_numpy(), vectors=vectors, logits=scores,
                        style=style.to_numpy(dtype=np.float32),
                        labels=frame.label.to_numpy(dtype=np.int64) if "label" in frame else np.full(len(frame), -1))
    temporary.replace(path)
    write(metadata, {"signature": signature, "rows": len(frame), "embedding_dim": vectors.shape[1],
                     "style_columns": list(style), "elapsed_s": time.perf_counter() - started})


def inner_boost():
    from experiments.deberta_finetune import baseline_prediction
    train, _, fitting = partitions()
    path = CACHE / "inner_boosting.csv"
    if path.exists():
        cached = pd.read_csv(path)
        assert np.array_equal(cached.id, train.loc[~fitting, "id"])
        return cached
    with threadpool_limits(limits=2):
        p = baseline_prediction(train.loc[fitting], train.loc[~fitting])
    frame = train.loc[~fitting, ["id", "label"]].copy()
    frame["boosting"] = p
    frame.to_csv(path, index=False)
    write(OUT / "inner_boosting.json", {"training_rows": 6400, "heldout_rows": 1600,
                                        "heldout_auroc": roc_auc_score(frame.label, p),
                                        "training_ids_sha256": hashlib.sha256(train.loc[fitting, "id"].to_numpy().tobytes()).hexdigest()})
    return frame


def stack_candidates():
    yield {"kind": "probability_blend", "weight": 0}, None
    for kind in ["probability_blend", "logit_blend"]:
        for weight in BLEND_WEIGHTS[1:]:
            yield {"kind": kind, "weight": weight}, None
    for c in [0.01, 0.1, 1, 10]:
        yield {"kind": "logistic_stack", "C": c}, make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
    for depth in [2, 3]:
        for l2 in [1, 10]:
            yield {"kind": "style_score_stack", "max_depth": depth, "l2": l2}, HistGradientBoostingClassifier(max_depth=depth, max_iter=100, learning_rate=0.05, l2_regularization=l2, min_samples_leaf=30, early_stopping=False, random_state=42)


def stack_inputs(config, boosting, neural_logits, style):
    scores = np.column_stack([logit(np.clip(boosting, 1e-12, 1 - 1e-12)), neural_logits])
    return np.column_stack([scores, np.log1p(style)]) if config["kind"] == "style_score_stack" else scores


def blend(config, boosting, neural_logits):
    weight = config["weight"]
    if weight == 0:
        return boosting.copy()
    if config["kind"] == "probability_blend":
        return (1 - weight) * boosting + weight * expit(neural_logits)
    return expit((1 - weight) * logit(np.clip(boosting, 1e-12, 1 - 1e-12)) + weight * neural_logits)


def stack_tune():
    import joblib
    if (OUT / "stack_selection.json").exists():
        return
    train, _, fitting = partitions()
    cache = np.load(CACHE / "inner_features.npz")
    base = inner_boost()
    assert np.array_equal(cache["ids"], train.id) and np.array_equal(base.id, train.loc[~fitting, "id"])
    b, n, style, y = base.boosting.to_numpy(), cache["logits"][~fitting], cache["style"][~fitting], cache["labels"][~fitting]
    rows, best_score, best = [], -np.inf, None
    cv = StratifiedKFold(4, shuffle=True, random_state=91)
    for config, estimator in stack_candidates():
        x = stack_inputs(config, b, n, style)
        with threadpool_limits(limits=2):
            p = blend(config, b, n) if estimator is None else cross_val_predict(estimator, x, y, cv=cv, method="predict_proba")[:, 1]
        score = roc_auc_score(y, p)
        rows.append({**config, "inner_oof_auroc": score})
        if score > best_score:
            best_score, best = score, (config, estimator)
    pd.DataFrame(rows).to_csv(OUT / "stack_search.csv", index=False)
    config, estimator = best
    if estimator is not None:
        with threadpool_limits(limits=2):
            estimator.fit(stack_inputs(config, b, n, style), y)
        joblib.dump(estimator, CACHE / "stack_combiner.joblib")
    write(OUT / "stack_selection.json", {"configuration": config, "inner_oof_auroc": best_score,
                                         "boosting_inner_auroc": roc_auc_score(y, b),
                                         "deberta_inner_auroc": roc_auc_score(y, n),
                                         "note": "Training-only selection; CV conditional on the prior encoder epoch selection. Final base models are refit on 8,000 rows."})
    print("Selected stack:", config, best_score, flush=True)


def hybrid_tune():
    if (OUT / "hybrid_selection.json").exists():
        return
    train, _, fitting = partitions()
    d = np.load(CACHE / "inner_features.npz")
    assert np.array_equal(d["ids"], train.id)
    rows, chosen, curves = [], {}, []
    for mode in ["hybrid", "text_only", "style_only"]:
        xf, xe, text_dim, _, _ = prepare_inputs(d["vectors"][fitting], d["style"][fitting], d["vectors"][~fitting], d["style"][~fitting], mode)
        best_score = -np.inf
        for index, params in enumerate(HEAD_GRID):
            config = {**params, "text_dim": text_dim, "mode": mode}
            predictions = {epoch: [] for epoch in EPOCHS}
            for seed in SEEDS:
                _, scores, history = train_head(xf, d["labels"][fitting], xe, config, seed, max(EPOCHS), EPOCHS)
                for epoch, p in scores.items():
                    predictions[epoch].append(p)
                curves.extend({"mode": mode, "configuration_index": index, "seed": seed, **r} for r in history)
            for epoch, scores in predictions.items():
                auc = roc_auc_score(d["labels"][~fitting], np.mean(scores, axis=0))
                rows.append({"mode": mode, "configuration_index": index, **params, "epochs": epoch, "inner_auroc": auc})
                if auc > best_score:
                    best_score = auc
                    chosen[mode] = {"configuration": config, "epochs": epoch, "seeds": SEEDS, "inner_auroc": auc}
            pd.DataFrame(rows).to_csv(OUT / "hybrid_search.csv", index=False)
            pd.DataFrame(curves).to_csv(OUT / "head_training_curves.csv", index=False)
            print(f"Head search: {mode}, configuration {index + 1}/{len(HEAD_GRID)}, best inner AUROC={best_score:.6f}", flush=True)
    write(OUT / "hybrid_selection.json", chosen)


def fit_evaluate():
    import joblib
    from experiments.neural_comparison import paired_ci
    if (OUT / "evaluation.json").exists():
        return
    train, val, fitting = partitions()
    d = np.load(CACHE / "final_features.npz")
    all_ids = pd.concat([train.id, val.id], ignore_index=True)
    assert np.array_equal(d["ids"], all_ids)
    k = len(train)
    bframe = pd.read_csv(PREVIOUS / "boosting_val_predictions.csv")
    assert np.array_equal(bframe.id, val.id) and np.array_equal(bframe.label, val.label)
    b = bframe.stack_v2_word_masked.to_numpy()
    predictions = val[["id", "label"]].copy()
    predictions["boosting_reference"] = b
    predictions["deberta_text_reference"] = expit(d["logits"][k:].astype(np.float64))
    selection = json.loads((OUT / "stack_selection.json").read_text())
    config = selection["configuration"]
    if "blend" in config["kind"]:
        stack = blend(config, b, d["logits"][k:])
    else:
        model = joblib.load(CACHE / "stack_combiner.joblib")
        stack = model.predict_proba(stack_inputs(config, b, d["logits"][k:], d["style"][k:]))[:, 1]
    predictions["deberta_boosting_stack"] = stack
    selected = json.loads((OUT / "hybrid_selection.json").read_text())
    for mode, choice in selected.items():
        xf, xe, dim, tscale, sscale = prepare_inputs(d["vectors"][:k], d["style"][:k], d["vectors"][k:], d["style"][k:], mode)
        config = {**choice["configuration"], "text_dim": dim}
        scores = []
        for seed in choice["seeds"]:
            model, _, history = train_head(xf, d["labels"][:k], xe, config, seed, choice["epochs"])
            scores.append(head_probabilities(model, xe))
            torch.save(model.state_dict(), CACHE / f"{mode}_seed{seed}.pt")
        joblib.dump({"text_scaler": tscale, "style_scaler": sscale}, CACHE / f"{mode}_scalers.joblib")
        predictions[f"fusion_{mode}"] = np.mean(scores, axis=0)
        print(f"Refitted {mode} heads on 8,000 rows.", flush=True)
    baseline_auc = roc_auc_score(val.label, b)
    rows = []
    for name in predictions.columns[2:]:
        p = predictions[name].to_numpy()
        assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
        auc = roc_auc_score(val.label, p)
        lo, hi = (0.0, 0.0) if name == "boosting_reference" else paired_ci(val.label.astype(int).to_numpy(), p, b)
        rows.append({"model": name, "validation_auroc": auc, "difference_from_boosting": auc - baseline_auc,
                     "paired_95ci_low": lo, "paired_95ci_high": hi, "beats_reference_numerically": auc > baseline_auc})
    predictions.to_csv(OUT / "val_predictions.csv", index=False)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "comparison.csv", index=False)
    write(OUT / "evaluation.json", {"training_rows": 8000, "validation_rows": 2000,
                                    "validation_ids_sha256": sha(VAL_IDS_PATH),
                                    "protocol_sha256": sha(OUT / "protocol.json"),
                                    "stack_selection_sha256": sha(OUT / "stack_selection.json"),
                                    "hybrid_selection_sha256": sha(OUT / "hybrid_selection.json"),
                                    "results": rows, "kaggle_uploaded": False,
                                    "note": "Heads use frozen fine-tuned representations. No joint encoder/head fine-tuning in this experiment. Validation has been reused across project experiments; intervals condition on fitted models and this split."})
    print(comparison.to_string(index=False), flush=True)


def winners():
    comparison = pd.read_csv(OUT / "comparison.csv").set_index("model")
    return [name for name in ["deberta_boosting_stack", "fusion_hybrid"]
            if bool(comparison.loc[name, "beats_reference_numerically"])]


def make_submissions():
    import joblib
    from src.data import load_split
    from experiments.deberta_finetune import baseline_prediction
    from experiments.make_submission import check_submission
    chosen = winners()
    if not chosen:
        print("Neither proposed method exceeded the validation reference; no candidate generated.", flush=True)
        return
    d = np.load(CACHE / "test_features.npz")
    test = load_split("test")
    assert np.array_equal(d["ids"], test.id)
    comparison = pd.read_csv(OUT / "comparison.csv").set_index("model")
    for name in chosen:
        filename = "deberta_style_hybrid_8000.csv" if name == "fusion_hybrid" else "deberta_boosting_stack_8000.csv"
        output = ROOT / "submissions" / filename
        if output.exists():
            continue
        if name == "fusion_hybrid":
            selection = json.loads((OUT / "hybrid_selection.json").read_text())["hybrid"]
            config = selection["configuration"]
            scales = joblib.load(CACHE / "hybrid_scalers.joblib")
            t = np.clip(scales["text_scaler"].transform(d["vectors"]), -10, 10)
            s = np.clip(scales["style_scaler"].transform(np.log1p(d["style"])), -10, 10)
            inputs = np.concatenate([t, s], axis=1).astype(np.float32)
            from src.deberta_hybrid import FusionHead
            scores = []
            for seed in selection["seeds"]:
                model = FusionHead(config["text_dim"], s.shape[1], config["width"], config["dropout"])
                model.load_state_dict(torch.load(CACHE / f"hybrid_seed{seed}.pt", weights_only=True))
                scores.append(head_probabilities(model, inputs))
            p = np.mean(scores, axis=0)
        else:
            selection = json.loads((OUT / "stack_selection.json").read_text())
            config = selection["configuration"]
            train, _, _ = partitions()
            with threadpool_limits(limits=2):
                b = baseline_prediction(train, test)
            if "blend" in config["kind"]:
                p = blend(config, b, d["logits"])
            else:
                estimator = joblib.load(CACHE / "stack_combiner.joblib")
                p = estimator.predict_proba(stack_inputs(config, b, d["logits"], d["style"]))[:, 1]
        submission = pd.DataFrame({"id": test.id, "label": p})
        check_submission(submission, test)
        submission.to_csv(output, index=False)
        write(output.with_suffix(".json"), {"model": name, "training_rows": 8000,
                                            "prediction_rows": len(test), "configuration": selection,
                                            "validation_auroc": float(comparison.loc[name, "validation_auroc"]),
                                            "csv_sha256": sha(output), "uploaded_to_kaggle": False,
                                            "note": "Stack combiner uses held-out base scores within training data; hybrid uses frozen fine-tuned encoder. See experiment protocol."})
        print(f"Wrote and checked {output}; no upload performed.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "inner", "stack", "heads", "final", "evaluate", "test", "submit"], default="all")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    protocol()
    if args.stage == "all":
        for stage in ["inner", "stack", "heads", "final", "evaluate"]:
            subprocess.run([sys.executable, "-u", "-m", "experiments.deberta_combinations", "--stage", stage],
                           cwd=ROOT, env=dict(os.environ, TOKENIZERS_PARALLELISM="false", PYTORCH_ENABLE_MPS_FALLBACK="1"), check=True)
        if winners():
            for stage in ["test", "submit"]:
                subprocess.run([sys.executable, "-u", "-m", "experiments.deberta_combinations", "--stage", stage], cwd=ROOT, check=True)
    elif args.stage in {"inner", "final", "test"}:
        cache_features(args.stage)
    elif args.stage == "stack":
        stack_tune()
    elif args.stage == "heads":
        hybrid_tune()
    elif args.stage == "submit":
        make_submissions()
    else:
        fit_evaluate()


if __name__ == "__main__":
    main()
