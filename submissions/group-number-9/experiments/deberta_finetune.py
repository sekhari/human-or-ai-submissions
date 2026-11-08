"""Fine-tune DeBERTa-v3-base, selecting epochs within training data only.

    python -m experiments.deberta_finetune --stage prepare
    python -m experiments.deberta_finetune --stage tune
    python -m experiments.deberta_finetune --stage fit
    python -m experiments.deberta_finetune --stage evaluate
    python -m experiments.deberta_finetune --stage submit

Weights/checkpoints remain under ignored .cache/. Results contain metrics/configs.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.data import load_split
from src.deberta import MODEL_ID, REVISION, autocast, device_for, load_model, load_tokenizer, optimizer_update, precision_for, predict, sync, train_model
from src.features import strip_layout
from src.split import VAL_IDS_PATH, load_train_val

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def callbacks(out, stage):
    def progress(row):
        write_json(out / f"{stage}_progress.json", row)
        print(f"{stage} epoch {row['epoch']}, batch {row['batch']}/{row['batches_in_epoch']}: loss {row['mean_training_loss']:.4f}, {row['elapsed_s']:.0f}s", flush=True)
    def epoch(history):
        pd.DataFrame(history).to_csv(out / f"{stage}_epochs.csv", index=False)
    return epoch, progress


def prepare(args, out, weights, device):
    from huggingface_hub import snapshot_download
    options = {"repo_id": MODEL_ID, "revision": REVISION,
               "cache_dir": ROOT / ".cache/huggingface", "token": False,
               "allow_patterns": ["config.json", "pytorch_model.bin", "spm.model",
                                  "tokenizer_config.json", "special_tokens_map.json"]}
    try:
        snapshot = Path(snapshot_download(**options, local_files_only=True))
        if not all((snapshot / p).is_file() for p in ("config.json", "pytorch_model.bin", "spm.model")):
            raise FileNotFoundError("Partial pretrained snapshot.")
    except (FileNotFoundError, OSError):
        snapshot = Path(snapshot_download(**options, max_workers=2))
    write_json(out / "pretrained_model.json", {
        "model_id": MODEL_ID, "revision": REVISION, "license": "MIT",
        "weights_sha256": digest(snapshot / "pytorch_model.bin"),
        "tokenizer_sha256": digest(snapshot / "spm.model"),
        "source": "https://huggingface.co/microsoft/deberta-v3-base",
    })
    print(f"Pretrained model ready: {MODEL_ID}, revision {REVISION}", flush=True)


def tune(args, out, weights, device):
    if (out / "selection.json").exists():
        raise SystemExit("Selection already exists; preserve it and use a new --output for a new experiment.")
    train, _ = load_train_val()
    fit_idx, inner_idx = train_test_split(np.arange(len(train)), test_size=1600,
                                         stratify=train.label, random_state=args.seed)
    ids = train[["id", "label"]].copy()
    ids["partition"] = "inner_fit"
    ids.loc[inner_idx, "partition"] = "inner_validation"
    ids.to_csv(out / "inner_split_ids.csv", index=False)
    config = {"model_id": MODEL_ID, "revision": REVISION, "seed": args.seed,
              "encoder_lr": args.encoder_lr, "head_lr": args.head_lr,
              "weight_decay": 0.01, "warmup_ratio": 0.1,
              "batch_size": args.batch_size, "gradient_accumulation": args.gradient_accumulation,
              "eval_batch_size": args.eval_batch_size, "epochs": args.epochs,
              "max_length": args.max_length, "precision": args.precision,
              "resume": args.resume, "one_epoch_per_process": args.one_epoch_per_process}
    write_json(out / "protocol.json", {
        "configuration": config, "device": str(device), "python": platform.python_version(),
        "runtime_settings": {"mps_padding_multiple": 128, "other_padding_multiple": 8,
                             "mps_memory_fraction": 0.65, "clear_unused_mps_buffers_every_batches": 25,
                             "optimizer_implementation": "fused AdamW on MPS; default AdamW elsewhere",
                             "one_epoch_per_process": args.one_epoch_per_process,
                             "save_complete_epoch_state": args.resume},
        "packages": {p: importlib.metadata.version(p) for p in
                     ("torch", "transformers", "sentencepiece", "numpy", "pandas", "scikit-learn")},
        "selection": "Highest AUROC on 1,600 inner holdout rows drawn only from the 8,000 training rows. Encoder/head learning rates fixed in advance. Official validation does not choose epochs.",
        "validation_ids_sha256": digest(VAL_IDS_PATH),
        "train_csv_sha256": digest(ROOT / "Data/train.csv"),
        "kaggle_public_target_approx": 0.976,
        "gradient_boosting_reference": "v2_word_masked; recorded fixed-validation AUROC 0.999288",
        "sources": ["https://huggingface.co/microsoft/deberta-v3-base",
                    "https://huggingface.co/docs/transformers/model_doc/deberta-v2"],
    })
    epoch, progress = callbacks(out, "tune")
    selected_epoch, history = train_model(train.iloc[fit_idx], train.iloc[inner_idx], config,
                                         device, weights / "tune", epoch, progress)
    if len(history) < config["epochs"]:
        print("Epoch checkpoint saved. Restart this stage to continue in a fresh GPU process.", flush=True)
        raise SystemExit(75)
    write_json(out / "selection.json", {
        "configuration": {**config, "epochs": selected_epoch},
        "selected_epoch": selected_epoch, "inner_auroc": max(r["inner_auroc"] for r in history),
        "validation_ids_sha256": digest(VAL_IDS_PATH),
        "selection_source": "inner holdout only; fixed official validation untouched",
    })


def smoke(args, out, weights, device):
    """Short real-model memory/gradient check; discard these updated weights."""
    import gc
    train, _ = load_train_val()
    batch_frame = train.groupby("label", group_keys=False).head(args.batch_size)
    batch_frame = batch_frame.sample(args.batch_size, random_state=args.seed)
    tokenizer = load_tokenizer()
    model = load_model(args.seed).to(device).train()
    encoded = tokenizer(batch_frame.text.tolist(), truncation=True, max_length=args.max_length,
                        padding="max_length", return_tensors="pt")
    batch = {key: value.to(device) for key, value in encoded.items()}
    batch["labels"] = torch.tensor(batch_frame.label.astype(int).to_numpy(), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.encoder_lr, fused=True if device.type == "mps" else None)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler(device.type, enabled=args.precision == "fp16")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    applied = 0
    started = time.perf_counter()
    for step in range(8):
        with autocast(device, args.precision):
            loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite smoke-test loss; do not start full training. Try --precision fp32.")
        scaler.scale(loss).backward()
        norm, updated = optimizer_update(model, optimizer, scheduler, scaler)
        applied += int(updated)
        print(f"Smoke step {step + 1}: loss={loss.item():.5f}, update_applied={updated}, scale={scaler.get_scale()}", flush=True)
    sync(device)
    if applied == 0:
        raise RuntimeError("All smoke-test updates overflowed; do not start full training. Try --precision fp32.")
    result = {"device": str(device), "precision": args.precision, "micro_batch_size": args.batch_size,
              "sequence_length": args.max_length, "optimizer_updates_applied": applied,
              "elapsed_s": time.perf_counter() - started,
              "gpu_name": torch.cuda.get_device_name() if device.type == "cuda" else str(device),
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None,
              "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30 if device.type == "cuda" else None,
              "note": "Only training-partition inputs were used; the modified smoke-test weights are discarded."}
    write_json(out / "smoke_test.json", result)
    print(json.dumps(result, indent=2), flush=True)
    del model, optimizer, scheduler, scaler, batch, loss, norm
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def fit(args, out, weights, device):
    if (weights / "fit/final/config.json").exists():
        raise SystemExit("Full-training checkpoint already exists; preserving it.")
    selection = json.loads((out / "selection.json").read_text())
    assert digest(VAL_IDS_PATH) == selection["validation_ids_sha256"]
    train, _ = load_train_val()
    epoch, progress = callbacks(out, "fit")
    _, history = train_model(train, None, selection["configuration"], device, weights / "fit", epoch, progress)
    if len(history) < selection["configuration"]["epochs"]:
        print("Epoch checkpoint saved. Restart this stage to continue in a fresh GPU process.", flush=True)
        raise SystemExit(75)


def baseline_prediction(train, val):
    # Reuse the team's strongest frozen stack, including OOF text scores for fitting.
    from experiments.stacking import text_scores, stack_matrix, fit_booster
    from src.features import build_extended_features
    from threadpoolctl import threadpool_limits
    row = pd.read_csv(ROOT / "results/stacking/comparison.csv").set_index("variant").loc["v2_word_masked"]
    text = {}
    with threadpool_limits(limits=2):
        for name in ("word_lr", "masked_lr"):
            text[name] = text_scores(name, train, val)
        Xf = stack_matrix(build_extended_features(train), {m: v[0] for m, v in text.items()}, list(text))
        Xe = stack_matrix(build_extended_features(val), {m: v[1] for m, v in text.items()}, list(text))
        p = fit_booster(json.loads(row.params), Xf, train.label.astype(int).to_numpy(), Xe)
    return p


def baseline(args, out, weights, device):
    train, val = load_train_val()
    reference = out / "boosting_val_predictions.csv"
    signature = {str(path): digest(ROOT / path) for path in
                 ("Data/train.csv", "splits/val_ids.csv", "src/features.py", "src/models.py",
                  "results/stacking/comparison.csv", "results/model_comparison/best_params.json")}
    if reference.exists():
        metadata = json.loads((out / "boosting_reference.json").read_text())
        assert metadata["input_sha256"] == signature, "Boosting reference inputs changed."
        return
    p = baseline_prediction(train, val)
    frame = val[["id", "label"]].copy()
    frame["stack_v2_word_masked"] = p
    frame.to_csv(reference, index=False)
    write_json(out / "boosting_reference.json", {
        "model": "v2_word_masked", "validation_auroc": roc_auc_score(val.label, p),
        "input_sha256": signature, "training_rows": len(train), "validation_rows": len(val),
    })
    print(f"Strongest boosting reference AUROC: {roc_auc_score(val.label, p):.6f}", flush=True)


def evaluate(args, out, weights, device):
    if (out / "evaluation.json").exists():
        raise SystemExit("This experiment was already evaluated; preserving the original result.")
    selection = json.loads((out / "selection.json").read_text())
    assert digest(VAL_IDS_PATH) == selection["validation_ids_sha256"]
    config = selection["configuration"]
    train, val = load_train_val()
    model = load_model(config["seed"], weights / "fit/final").to(device)
    tokenizer = load_tokenizer()
    start = time.perf_counter()
    p = predict(model, val, tokenizer, device, config["max_length"], config["eval_batch_size"], config["precision"])
    predictions = val[["id", "label"]].copy()
    predictions["deberta"] = p
    predictions.to_csv(out / "val_predictions.csv", index=False)
    auc = roc_auc_score(val.label, p)
    print(f"DeBERTa fixed-validation AUROC: {auc:.6f}", flush=True)
    stripped = val.copy()
    stripped["text"] = stripped.text.map(strip_layout)
    raw_tokens = tokenizer(val.text.tolist(), truncation=True, max_length=config["max_length"], return_attention_mask=True)
    stripped_tokens = tokenizer(stripped.text.tolist(), truncation=True, max_length=config["max_length"], return_attention_mask=True)
    identical_inputs = raw_tokens == stripped_tokens
    if identical_inputs:
        # Exact equality of all model inputs is stronger evidence than a second
        # floating-point forward pass over the same inputs.
        stripped_p = p.copy()
    else:
        stripped_p = predict(model, stripped, tokenizer, device, config["max_length"], config["eval_batch_size"], config["precision"])
    predictions["deberta_layout_stripped"] = stripped_p
    predictions.to_csv(out / "val_predictions.csv", index=False)
    # Release the large GPU model before running the classical-model comparison.
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    globals()["baseline"](args, out, weights, device)
    reference = pd.read_csv(out / "boosting_val_predictions.csv")
    assert np.array_equal(reference.id, val.id) and np.array_equal(reference.label, val.label)
    baseline_p = reference.stack_v2_word_masked.to_numpy()
    predictions["stack_v2_word_masked"] = baseline_p
    predictions.to_csv(out / "val_predictions.csv", index=False)
    baseline_auc = roc_auc_score(val.label, baseline_p)
    bins = pd.cut(val.text.str.split().str.len(), [0, 149, 250, float("inf")],
                  labels=["under_150_words", "150_to_250_words", "over_250_words"])
    subgroup_rows = []
    for name, scores in (("deberta", p), ("stack_v2_word_masked", baseline_p)):
        for group in bins.cat.categories:
            mask = (bins == group).to_numpy()
            labels = val.label.to_numpy()[mask]
            subgroup_rows.append({"model": name, "group": str(group), "count": int(mask.sum()),
                                  "human_count": int((labels == 0).sum()), "ai_count": int((labels == 1).sum()),
                                  "auroc": roc_auc_score(labels, scores[mask]) if len(np.unique(labels)) == 2 else None})
    pd.DataFrame(subgroup_rows).to_csv(out / "length_subgroups.csv", index=False)
    from experiments.neural_comparison import paired_ci
    lo, hi = paired_ci(val.label.astype(int).to_numpy(), p, baseline_p)
    result = {"model": MODEL_ID, "validation_auroc": auc,
              "layout_stripping_gives_identical_model_inputs": identical_inputs,
              "layout_stripped_validation_auroc": roc_auc_score(val.label, stripped_p),
              "strongest_boosting_validation_auroc": baseline_auc,
              "difference": auc - baseline_auc, "difference_paired_95ci": [lo, hi],
              "evaluation_elapsed_s": time.perf_counter() - start,
              "training_rows": len(train), "validation_rows": len(val),
              "selection_json_sha256": digest(out / "selection.json"),
              "kaggle_public_target_approx": 0.976, "kaggle_public_score": None,
              "note": "Validation and Kaggle public AUROC are different quantities. No public target can be verified without a Kaggle submission. The paired interval conditions on this split and fitted models."}
    write_json(out / "evaluation.json", result)
    pd.DataFrame([{"model": "deberta_v3_base", "validation_auroc": auc},
                  {"model": "stack_v2_word_masked", "validation_auroc": baseline_auc}]).to_csv(out / "comparison.csv", index=False)
    print(json.dumps(result, indent=2), flush=True)


def submit(args, out, weights, device):
    # First candidate retains the independently evaluated 8,000-row model. No hidden
    # test labels or validation-based refitting are used; manifest states this clearly.
    selection = json.loads((out / "selection.json").read_text())
    config = selection["configuration"]
    output = args.submission_path
    if output.exists():
        raise SystemExit(f"Preserving existing {output}.")
    model = load_model(config["seed"], weights / "fit/final").to(device)
    test = load_split("test")
    p = predict(model, test, load_tokenizer(), device, config["max_length"], config["eval_batch_size"], config["precision"])
    sub = pd.DataFrame({"id": test.id, "label": p})
    from experiments.make_submission import check_submission
    check_submission(sub, test)
    output.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output, index=False)
    write_json(output.with_suffix(".json"), {
        "model": MODEL_ID, "revision": REVISION, "configuration": config,
        "training_rows": 8000, "checkpoint": str((weights / "fit/final").resolve()),
        "csv_sha256": digest(output), "prediction_rows": len(sub),
        "uploaded_by_script": False,
    })
    print(f"Wrote and checked {output}; no upload performed.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "smoke", "tune", "fit", "baseline", "evaluate", "submit", "all"], default="all")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Save/restore complete state at epoch boundaries.")
    parser.add_argument("--one-epoch-per-process", action="store_true", help="Requires --resume; exit 75 between epochs to release GPU caches.")
    parser.add_argument("--output", type=Path, default=ROOT / "results/deberta_v3")
    parser.add_argument("--weights", type=Path, default=ROOT / ".cache/deberta_v3")
    parser.add_argument("--submission-path", type=Path, default=ROOT / "submissions/deberta_v3_base_8000.csv")
    args = parser.parse_args()
    if args.one_epoch_per_process and not args.resume:
        parser.error("--one-epoch-per-process requires --resume")
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    device = device_for(args.device)
    args.precision = precision_for(device, args.precision)
    print(f"Device: {device}; precision: {args.precision}; checkpoints: {args.weights}", flush=True)
    stages = ["prepare", "smoke", "tune", "fit", "evaluate", "submit"] if args.stage == "all" else [args.stage]
    for stage in stages:
        globals()[stage](args, args.output, args.weights, device)


if __name__ == "__main__":
    main()
