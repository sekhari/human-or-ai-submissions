import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train import MAX_LENGTH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument(
        "--model-dir",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")

    test = pd.read_csv(args.test_csv)

    if not {"id", "text"}.issubset(test.columns):
        raise ValueError("Test CSV must contain id and text columns.")

    if len(test) != 3000 or not test["id"].is_unique:
        raise ValueError("Expected 3,000 test rows with unique IDs.")

    if test[["id", "text"]].isna().any().any():
        raise ValueError("Test data contains missing values.")

    if not test["text"].map(lambda value: isinstance(value, str)).all():
        raise ValueError("All text values must be strings.")

    model_dirs = [path.expanduser().resolve() for path in args.model_dir]

    if len(set(model_dirs)) != len(model_dirs):
        raise ValueError("Each model directory must be supplied only once.")

    for model_dir in model_dirs:
        if not (model_dir / "config.json").is_file():
            raise FileNotFoundError(
                f"Checkpoint config.json not found in {model_dir}"
            )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    passages = test["text"].tolist()
    probability_sum = torch.zeros(len(passages), dtype=torch.float64)

    for model_dir in model_dirs:
        print(f"Loading checkpoint: {model_dir}")

        # Training checkpoints may use the tokenizer saved in best_model.
        tokenizer_dir = model_dir
        if not (tokenizer_dir / "tokenizer_config.json").is_file():
            tokenizer_dir = model_dir.parent.parent / "best_model"

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            use_fast=False,
        )
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.to(device)
        model.eval()

        probabilities = []

        with torch.inference_mode():
            for start in range(0, len(passages), args.batch_size):
                batch = tokenizer(
                    passages[start : start + args.batch_size],
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                ).to(device)

                logits = model(**batch).logits
                ai_scores = logits.float().softmax(dim=-1)[:, 1]
                probabilities.extend(ai_scores.cpu().tolist())

        probability_sum += torch.tensor(
            probabilities,
            dtype=torch.float64,
        )

        del model, batch, logits, ai_scores

        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    averaged_probabilities = probability_sum / len(model_dirs)

    submission = pd.DataFrame({
        "id": test["id"].to_numpy(),
        "label": averaged_probabilities.numpy(),
    })

    if (
        submission["label"].isna().any()
        or not submission["label"].between(0, 1).all()
    ):
        raise ValueError("Predicted probabilities are invalid.")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output_csv, index=False)

    print(f"Averaged {len(model_dirs)} checkpoint(s)")
    print(f"Wrote {len(submission)} predictions to {args.output_csv}")


if __name__ == "__main__":
    main()