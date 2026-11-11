"""Load a completed portable RoBERTa run and write Kaggle AI probabilities."""
import argparse
import json
import os
from pathlib import Path

from human_ai.data import read_csv
from submission_model import PACKAGE_ROOT, predict_records, resolve_device, write_csv_exclusive


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="The completed train.py --output-dir")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=PACKAGE_ROOT / "data/raw")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-test-rows", type=int, default=None, help="Smoke testing only; produces a partial CSV")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.max_test_rows is not None and args.max_test_rows < 1):
        parser.error("Batch size and any smoke-test row limit must be positive")
    return args


def run(args):
    if args.output.exists():
        raise FileExistsError(f"Choose a new prediction output: {args.output}")
    completion = json.loads((args.model_dir / "completion.json").read_text())
    if completion.get("status") != "complete" or completion.get("model_directory") != "model":
        raise ValueError("Model training has not completed successfully")
    config = json.loads((args.model_dir / "config.json").read_text())
    records = read_csv(args.data_dir / "test.csv")
    ids = [row["id"] for row in records]
    if len(records) != 3000 or len(set(ids)) != 3000 or any("text" not in row for row in records):
        raise ValueError("Expected the 3,000 original test rows with unique IDs and text")
    if args.max_test_rows is not None:
        records = records[:args.max_test_rows]
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = resolve_device(torch, args.device)
    torch.set_num_threads(config["recipe"]["cpu_threads"])
    model_path = args.model_dir / "model"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float32, attn_implementation="sdpa",
    ).to(device)
    probabilities, _ = predict_records(
        torch, model, tokenizer, records, device, config["recipe"]["max_length"], args.batch_size,
    )
    rows = [{"id": row["id"], "label": format(probability, ".17g")}
            for row, probability in zip(records, probabilities, strict=True)]
    write_csv_exclusive(args.output, rows, ["id", "label"])
    print(json.dumps({"status": "complete", "output": str(args.output), "n_test": len(records),
                      "full_data_model": config["full_data"],
                      "smoke_run": config["smoke_run"] or args.max_test_rows is not None,
                      "device": device, "class1_is_ai": True}), flush=True)


if __name__ == "__main__":
    run(parse_args())
