"""Train the raw-text RoBERTa recipe on the committed split or all 10,000 labels."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import time

from human_ai.data import load_partitions, sha256
from submission_model import (
    MODEL_NAME, MODEL_REVISION, PACKAGE_ROOT, RECIPE, atomic_json, auroc, epoch_order,
    id_digest, package_versions, predict_records, resolve_device, seed_training,
    tokenize, update_groups, write_csv_exclusive,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=PACKAGE_ROOT / "data/raw")
    parser.add_argument("--full-data", action="store_true", help="Fit all 10,000 labels; no held-out evaluation")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true", help="Use an already downloaded public model")
    parser.add_argument("--epochs", type=int, default=RECIPE["epochs"], help="Default 2; changes are recorded as a smoke run")
    parser.add_argument("--max-train-rows", type=int, default=None, help="Smoke testing only")
    parser.add_argument("--max-validation-rows", type=int, default=None, help="Smoke testing only")
    args = parser.parse_args()
    for name in ("epochs", "max_train_rows", "max_validation_rows"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("max_train_rows", "max_validation_rows"):
        value = getattr(args, name)
        if value is not None and value < 2:
            parser.error(f"--{name.replace('_', '-')} needs at least two rows for a balanced smoke test")
    if args.full_data and args.max_validation_rows is not None:
        parser.error("--max-validation-rows does not apply to --full-data")
    return args


def smoke_subset(records, limit):
    """Take a deterministic, class-balanced subset while retaining canonical order."""
    if limit is None or limit >= len(records):
        return records
    remaining = {"0": limit // 2, "1": limit - limit // 2}
    selected = []
    for row in records:
        if remaining[row["label"]] > 0:
            selected.append(row)
            remaining[row["label"]] -= 1
    if len(selected) != limit:
        raise ValueError("The requested smoke subset cannot be balanced across both labels")
    return selected


def run(args):
    if args.output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output_dir}")
    fitting, validation = load_partitions(raw_dir=args.data_dir)
    training = fitting + validation if args.full_data else fitting
    held = [] if args.full_data else validation
    training = smoke_subset(training, args.max_train_rows)
    held = smoke_subset(held, args.max_validation_rows)
    smoke = args.epochs != RECIPE["epochs"] or args.max_train_rows is not None or args.max_validation_rows is not None
    recipe = {**RECIPE, "epochs": args.epochs}
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = resolve_device(torch, args.device)
    torch.set_num_threads(recipe["cpu_threads"])
    seed_training(torch, device, recipe["seed"])
    # Reserve this run so concurrent or retried invocations cannot overwrite it.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "format_version": 1,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "recipe": recipe,
        "device": device,
        "full_data": args.full_data,
        "smoke_run": smoke,
        "n_train": len(training),
        "n_validation": len(held),
        "training_id_order_sha256": id_digest(training),
        "validation_id_order_sha256": id_digest(held),
        "source_train_sha256": sha256(args.data_dir / "train.csv"),
        "split_manifest_sha256": sha256(PACKAGE_ROOT / "data/splits/manifest.json"),
        "python_version": platform.python_version(),
        "packages": package_versions(),
        "raw_whitespace_preserved": True,
        "test_used_for_fitting": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(args.output_dir / "config.json", config)
    started = time.perf_counter()
    load_options = {"revision": MODEL_REVISION, "cache_dir": args.cache_dir,
                    "local_files_only": args.local_files_only}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, **load_options)
    encoded = tokenize(tokenizer, training, recipe["max_length"])
    encoded["labels"] = torch.tensor([int(row["label"]) for row in training], dtype=torch.long)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, dtype=torch.float32, attn_implementation="sdpa",
        use_safetensors=True, **load_options,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"],
        foreach=False, fused=False,
    )
    steps = []
    training_started = time.perf_counter()
    for epoch in range(1, recipe["epochs"] + 1):
        model.train()
        order = epoch_order(len(training), recipe["seed"], epoch)
        for update, (indices, micros) in enumerate(update_groups(
            order, recipe["microbatch"], recipe["gradient_accumulation"]
        ), start=1):
            optimizer.zero_grad(set_to_none=True)
            update_loss = 0.0
            for micro in micros:
                index = torch.tensor(micro, dtype=torch.long)
                batch = {name: values.index_select(0, index).to(device) for name, values in encoded.items()}
                result = model(**batch)
                if not torch.isfinite(result.loss).item():
                    raise RuntimeError("Nonfinite training loss")
                weight = len(micro) / len(indices)
                (result.loss * weight).backward()
                update_loss += result.loss.detach().item() * weight
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), recipe["max_grad_norm"], foreach=False, error_if_nonfinite=True,
            )
            optimizer.step()
            row = {"epoch": epoch, "update": update, "documents": len(indices),
                   "loss": update_loss, "preclip_gradient_norm": norm.item()}
            steps.append(row)
            total = math.ceil(len(training) / (recipe["microbatch"] * recipe["gradient_accumulation"]))
            if update == 1 or update % 25 == 0 or update == total:
                print(json.dumps({**row, "updates_per_epoch": total}), flush=True)
    training_seconds = time.perf_counter() - training_started
    write_csv_exclusive(args.output_dir / "training_log.csv", steps, list(steps[0]))
    metrics = {"n_train": len(training), "n_validation": len(held), "smoke_run": smoke,
               "epochs": recipe["epochs"], "optimizer_updates": len(steps),
               "training_seconds": training_seconds, "validation_auroc": None,
               "validation_auroc_logit_margin": None}
    if held:
        probabilities, margins = predict_records(
            torch, model, tokenizer, held, device, recipe["max_length"], recipe["microbatch"],
        )
        labels = [int(row["label"]) for row in held]
        metrics["validation_auroc"] = auroc(labels, probabilities)
        metrics["validation_auroc_logit_margin"] = auroc(labels, margins)
        rows = [{"id": row["id"], "label": row["label"], "p_ai": format(probability, ".17g")}
                for row, probability in zip(held, probabilities, strict=True)]
        write_csv_exclusive(args.output_dir / "validation_predictions.csv", rows, ["id", "label", "p_ai"])
    # A completion marker is published only after the entire model directory is ready.
    model_directory = args.output_dir / ".model-incomplete"
    model.save_pretrained(model_directory, safe_serialization=True)
    tokenizer.save_pretrained(model_directory)
    model_directory.rename(args.output_dir / "model")
    metrics["process_seconds"] = time.perf_counter() - started
    atomic_json(args.output_dir / "metrics.json", metrics)
    atomic_json(args.output_dir / "completion.json", {"status": "complete", "model_directory": "model"})
    print(json.dumps({"status": "complete", "output_dir": str(args.output_dir), **metrics}), flush=True)


if __name__ == "__main__":
    run(parse_args())
