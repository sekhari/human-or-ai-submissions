"""Score raw test passages with the saved v2 and DistilRoBERTa models."""
import argparse
import json
from pathlib import Path


def predict(data_dir, out_dir):
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    from scipy.special import expit
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments)
    from distilroberta.features import lm_features, ranking_margins, stylo

    test = pd.read_csv(data_dir / "test.csv", keep_default_na=False)
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    if not {"id", "text"}.issubset(test) or len(test) != 3000:
        raise ValueError("Expected 3,000 test rows with id and text columns.")
    if (list(sample.columns) != ["id", "label"] or not test.id.is_unique or
            not sample.id.is_unique or set(test.id) != set(sample.id)):
        raise ValueError("The sample submission must contain each test ID exactly once.")
    bundle_dir = out_dir / "distilroberta_v1"
    bundle = joblib.load(bundle_dir / "v2.joblib")
    info = json.loads((bundle_dir / "final_checkpoint.json").read_text())
    checkpoint = bundle_dir / info["checkpoint"]
    texts = test.text.tolist()
    style = np.asarray([stylo(text) for text in texts], dtype=np.float32)
    lm = lm_features(texts, model_name=bundle["lm_source"], revision=bundle.get("lm_revision"))
    matrix = np.column_stack([style, lm])[:, bundle["ci_style"]]
    style_margin = bundle["f_style"].decision_function(matrix)
    word_margin = bundle["f_word"].decision_function(bundle["word_vec"].transform(texts))

    def standardize(values, epsilon=0.0):
        scale = values.std() + epsilon
        if not np.isfinite(values).all() or scale == 0:
            raise ValueError("A component has nonfinite scores or zero normalization scale.")
        return (values - values.mean()) / scale

    v2 = 0.5 * standardize(style_margin, 1e-9) + 0.5 * standardize(word_margin, 1e-9)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    args = TrainingArguments(output_dir=str(bundle_dir / "predict_tmp"),
                             per_device_eval_batch_size=32, fp16=torch.cuda.is_available(),
                             use_cpu=not torch.cuda.is_available(), report_to="none")
    trainer = Trainer(model=model, args=args, data_collator=DataCollatorWithPadding(tokenizer))
    transformer = ranking_margins(trainer, tokenizer, texts)
    probability = expit(0.75 * standardize(v2) + 0.25 * standardize(transformer))
    scored = pd.Series(probability, index=test.id)
    sample["label"] = sample.id.map(scored)
    path = out_dir / "submission.csv"
    pending = path.with_suffix(".csv.tmp")
    sample.to_csv(pending, index=False)
    pending.replace(path)
    print(f"Saved {len(sample)} predictions to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    predict(args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
