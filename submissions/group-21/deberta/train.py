import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256


class PassageDataset(torch.utils.data.Dataset):
    def __init__(self, frame, tokenizer):
        self.inputs = tokenizer(
            frame["text"].tolist(),
            truncation=True,
            max_length=MAX_LENGTH,
        )
        self.labels = frame["label"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: values[index] for key, values in self.inputs.items()}
        item["labels"] = self.labels[index]
        return item


def ai_probabilities(logits):
    return torch.as_tensor(logits).float().softmax(dim=-1)[:, 1].numpy()


def compute_metrics(prediction):
    scores = ai_probabilities(prediction.predictions)
    return {"auroc": roc_auc_score(prediction.label_ids, scores)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Score the existing best_model checkpoint on both splits.",
    )
    args = parser.parse_args()

    model_dir = args.output_dir / "best_model"
    if args.evaluate_only and not model_dir.is_dir():
        parser.error(f"Saved checkpoint directory does not exist: {model_dir}")

    set_seed(args.seed)
    data = pd.read_csv(args.train_csv)
    train_frame, validation_frame = train_test_split(
        data,
        test_size=0.20,
        stratify=data["label"],
        random_state=args.seed,
    )

    if args.evaluate_only:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
            id2label={0: "HUMAN", 1: "AI"},
            label2id={"HUMAN": 0, "AI": 1},
        )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        learning_rate=2e-5,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        weight_decay=0.01,
        warmup_ratio=0.10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="auroc",
        greater_is_better=True,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        seed=args.seed,
        report_to="none",
    )

    train_dataset = PassageDataset(train_frame, tokenizer)
    validation_dataset = PassageDataset(validation_frame, tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    if not args.evaluate_only:
        trainer.train()
        trainer.save_model(str(model_dir))
        tokenizer.save_pretrained(model_dir)
        print(f"Saved best model to {model_dir}")

    validation_result = trainer.predict(
        validation_dataset,
        metric_key_prefix="validation",
    )
    validation_output = validation_frame[["id", "label"]].copy()
    validation_output["ai_probability"] = ai_probabilities(
        validation_result.predictions
    )
    validation_path = args.output_dir / "validation_predictions.csv"
    validation_output.to_csv(validation_path, index=False)

    print(f"Validation AUROC: {validation_result.metrics['validation_auroc']:.6f}")
    print(f"Saved validation predictions to {validation_path}")

    train_result = trainer.predict(train_dataset, metric_key_prefix="train")
    print(f"Training AUROC: {train_result.metrics['train_auroc']:.6f}")


if __name__ == "__main__":
    main()