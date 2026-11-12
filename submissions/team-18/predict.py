"""Write AI probabilities in the competition sample submission format."""

import argparse
import json
from pathlib import Path

import joblib

from data_io import read_texts, submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("submission.csv"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    config = json.loads((args.run_dir / "run.json").read_text())["config"]
    id_column, text_column = config["id_col"], config["text_col"]
    test = read_texts(args.test, id_column, text_column)
    if config["model"] == "deberta":
        import neural

        scores = neural.predict(args.run_dir / "transformer", test[text_column],
                                config["max_length"], config["batch_size"], args.device)
    else:
        model = joblib.load(args.run_dir / "model.joblib")
        scores = model.predict_proba(test[text_column])[:, 1]
    result = submission(test[id_column], scores, args.sample_submission, id_column)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"Wrote {len(result)} predictions to {args.out}")


if __name__ == "__main__":
    main()
