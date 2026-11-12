# Party in Thee Code

Run every command from `submissions/party-in-thee-code/`.
Team branch: `submission/party-in-thee-code`.

## Team

Kaggle team name: **Party in Thee Code**.

- Skye Li (yl4534): shared split, baseline, Q1 model comparison, Q2 learning curves.
- Candice (yx772): Q3 pseudo-labeling, and the character 2-5 gram plus blank-line submission.
- James Chen (jzc28): Q4 error analysis, Q5 improvement, and the newline pipeline in `predict.py`.

## Python version and dependencies

Python 3.11 or newer. The recorded baseline environment is Python 3.12.14
(`results/environment.json`).

```bash
python -m pip install -r requirements.txt
```

Q2 also needs matplotlib:

```bash
python -m pip install -r requirements_q2.txt
```

Put `train.csv`, `unlabeled.csv`, and `test.csv` in `data/`. A download named
`train(1).csv` can be renamed to `train.csv`. Raw CSVs and model checkpoints
are not committed.

## Train and evaluate on the labeled validation split

The shared split is 8,000 training rows and 2,000 validation rows, stratified
with seed 42. Training has 4,000 examples of each label. Validation has 1,000
of each. Use `splits/train_ids.csv` and `splits/validation_ids.csv`. Do not
make another split. The IDs were split after sorting them as strings.

Current model, fit only on the 8,000 training rows and scored on the 2,000
validation rows:

```bash
python q5_char_blank.py --train data/train.csv
```

This writes `results/q5_char_blank.csv` and `results/q5_char_blank.json`.
Clean validation AUROC is 0.9937. The same run reports the newline model at
0.9917. With whitespace collapsed, the scores are 0.9261 and 0.8887.

Earlier experiments, same split and seed:

```bash
python baseline.py --train data/train.csv
python q1.py --train data/train.csv
python q2.py --train data/train.csv
python q3_pseudo_label.py --train data/train.csv
python q4_error_analysis.py --train data/train.csv
python q5_improvement.py --train data/train.csv
python q5_round2.py --train data/train.csv
python predict.py --train data/train.csv --validate
```

## Command that writes submission.csv

```bash
python q5_char_blank.py --train data/train.csv --test data/test.csv --out submission.csv
```

This fits on all 10,000 labeled rows and writes `submission.csv` with columns
`id` and `label`. `label` is P(AI). The public leaderboard score for this file
is 0.95836.

`predict.py` is the earlier newline pipeline. It is not the current submission.

## Seeds and config

- Seed 42 for the split, the SVM, the logistic regression, and the 3-fold split.
- Shared split files above. Label 0 is human and label 1 is AI.
- Current model: character TF-IDF, n-grams 2-5, `min_df=2`, 50,000 features,
  sublinear TF, smoothed IDF, L2 normalization, `LinearSVC` with `C=1`.
- A logistic regression (`C=1`, `liblinear`) combines the SVM margin with a
  blank-line indicator. Its weights are fit on 3-fold out-of-fold scores from
  the rows used for that fit, not on the validation labels.
- In `train.csv`, 0.06% of human passages contain a blank line, against 66% of
  AI passages. 31.9% of `test.csv` passages contain one (33.0% of `train.csv`).
- Q1 selected character 3-5 grams plus a linear SVM, `C=1`, validation AUROC
  0.970704. Q5 then kept newlines by replacing each one with "¶". That model
  reaches 0.9917 on clean validation and 0.889 after whitespace is collapsed.

## AI-tool disclosure

ChatGPT helped draft and debug the baseline and Q1/Q2 code and notes, execute
those experiments, and check the exported outputs. Claude (Anthropic) helped
write and run the Q4, Q5, and `predict.py` code and draft the corresponding
report text. The team reviewed the code, reran the commands, and checked the
numbers against the saved outputs. No AI tool or external API labeled or scored
the competition passages.

## Result index

- Q1: character TF-IDF + linear SVM (`C=1`), validation AUROC 0.970704 (`results/q1/`).
- Q2: learning curves (`Q2_README.md`, `results/q2/`).
- Q3: confidence pseudo-labeling on the word logistic regression, no gain (`results/q3_pseudo_metrics.json`).
- Q4: error analysis of the Q1 model (`results/q4/`).
- Q5: newline-aware character SVM, validation AUROC 0.991701 (`results/q5/`). `predict.py` still runs this model.
- Current submission: `q5_char_blank.py`. Shared-split AUROC 0.9937 clean and 0.9261 with whitespace collapsed. Public leaderboard 0.95836.
