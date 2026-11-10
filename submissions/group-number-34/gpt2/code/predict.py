"""Write GPT-2 classification probabilities for a text CSV file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from transformers import AutoModelForSequenceClassification

from common import (
    choose_device, encode_frame, make_loader, make_tokenizer,
    predict_probabilities, read_text_csv, recover_model_directory,
)


def run_prediction(
    checkpoint, test_csv, output, sample_submission=None,
    device="auto", batch_size=4, max_length=512,
) -> Path:
    """Load a classifier and write probabilities in the required ID order."""
    checkpoint = Path(checkpoint)
    if checkpoint.is_file():
        reference = json.loads(checkpoint.read_text())
        directory = reference['directory']
        if Path(directory).name != directory:
            raise ValueError('The checkpoint reference must contain a directory name.')
        checkpoint = checkpoint.parent / directory
    checkpoint = recover_model_directory(checkpoint)
    output = Path(output)
    metadata_path = checkpoint / "training_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("max_length") != max_length:
            raise ValueError("Prediction max_length must match the saved training metadata.")
    frame = read_text_csv(test_csv)
    if sample_submission is not None:
        sample = pd.read_csv(sample_submission, dtype={"id": str}, keep_default_na=False)
        if "id" not in sample or sample["id"].duplicated().any():
            raise ValueError("The sample submission must contain unique IDs.")
        if len(sample) != len(frame) or set(sample["id"]) != set(frame["id"]):
            raise ValueError("Test and sample-submission IDs must match exactly.")
        frame = frame.set_index("id").loc[sample["id"].tolist()].reset_index()

    tokenizer = make_tokenizer(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    if model.config.model_type != "gpt2" or model.config.num_labels != 1:
        raise ValueError("Expected a GPT-2 checkpoint with one classification logit.")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    device = choose_device(str(device))
    model.float().to(device)
    dataset = encode_frame(frame, tokenizer, max_length)
    loader = make_loader(dataset, tokenizer, batch_size)
    probabilities = predict_probabilities(model, loader, device)
    if len(probabilities) != len(frame):
        raise RuntimeError("Prediction count does not match the input row count.")
    result = pd.DataFrame({"id": frame["id"], "label": probabilities})
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    result.to_csv(temporary, index=False)
    os.replace(temporary, output)
    print(f"Saved {len(result)} predictions to {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test_csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample_submission", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()
    run_prediction(**vars(args))


if __name__ == "__main__":
    main()
