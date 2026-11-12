# Party in Thee Code: handoff

Use the shared split files for all controlled comparisons. Label 0 is human;
label 1 is AI. The probability column always means P(AI).

## Load the baseline

These examples run from the project folder, with the supplied data in `data/`.

```python
import joblib
import pandas as pd
from baseline import load_data, load_split, make_model, fit_and_evaluate

df = load_data("data/train.csv")
train, val = load_split(df)
model = joblib.load("models/baseline.joblib")
```

The model checkpoint is in the private handoff ZIP, not the course repository.
Run `python baseline.py --train data/train.csv` to generate it locally.
The saved model uses the versions in requirements.txt. A fresh baseline can be
fit with `make_model()` or the baseline script if the environment differs.

## Q3: unlabeled data

```python
unlabeled = pd.read_csv("data/unlabeled.csv", dtype={"id": str})
p_ai = model.predict_proba(unlabeled["text"])[:, 1]
```

These are model predictions, not true labels. For a pseudo-labeling experiment,
choose a confidence rule, add only the selected unlabeled rows to `train`, and
fit a fresh pipeline. Evaluate on `val` and compare with supervised-only using
the same settings. Record the threshold, number selected and label counts.
Validation rows must stay out of training, vocabulary fitting and pseudo-label
selection. Check overlaps between unlabeled and validation texts before using
unlabeled rows. Never add the validation labels to the training set.

## Q4: error analysis

```python
pred = pd.read_csv("results/validation_predictions.csv", dtype={"id": str})
cases = pred.merge(df[["id", "text"]], on="id", validate="one_to_one")
false_positives = cases[cases.error_type == "FP"]
false_negatives = cases[cases.error_type == "FN"]
```

The decision threshold is 0.5; probabilities equal to 0.5 are predicted AI.
Word counts use whitespace-separated tokens, not the TF-IDF tokenizer.
Pick at least five FP and five FN, read the texts and write short failure
hypotheses. Length-bin AUROC can be computed directly from these predictions.
The file is for this baseline. If we select another model later, repeat Q4
using that model's validation predictions.

## Q5: improvement

Reuse `train` and `val` above. Compare a specific change against a fresh
supervised baseline. Keep other settings fixed where possible and record what
changed. Runtime comparisons should be rerun on one machine. We have not
tested an improvement or concluded which failure pattern to target yet.

## Q6: code and development record

The training entrypoint is `python baseline.py --train data/train.csv`.
The notebook calls the same functions. Seed, config, environment and metrics
are saved in the package. There is no final test-prediction command yet;
that comes after model selection. The PR URL and later experiment commits
still need to come from our actual team repository.

Keep commits as the models and experiments change. ChatGPT helped draft the baseline code and handoff
notes, execute the baseline, and check the exported files. Later team changes
and checks should be reflected in the final AI-use disclosure.
