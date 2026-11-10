# Zhitong Liu (zl2358) — Team 35 contribution

Kaggle team name: `team 35` (repository folder and branch use `team-35`).

Contributor: Zhitong Liu (zl2358). Other teammates' work belongs in their own NetID folders before team-level integration.

## Report draft

The version-controlled LaTeX source is [`writeup/team_35_report.tex`](writeup/team_35_report.tex). It is Zhitong's working draft for later team integration. Q1 contains measured character-model results and marked spaces for additional models. `writeup/seed42_results.json` records the Q1 search and wording-cue ablation measurements without competition passages. The final team report and course PR will be prepared after combining contributions.

## Setup

Use Python 3.14. The scripts use the CSV files in this folder's `data/` directory by default. Open `submissions/team-35/zl2358` as the VS Code folder and select your regular Python 3.14 interpreter. Install dependencies into that Python installation if needed:

```powershell
py -m pip install -r requirements.txt
```

Place the Kaggle `train.csv` and `test.csv` in `data/`. The repository ignores `data/` and CSV files, so the competition data stays local.

On macOS, from this team directory, the equivalent setup is `python3 -m venv .venv`, followed by `.venv/bin/python -m pip install -r requirements.txt`. Run scripts with `.venv/bin/python` in place of `py` in the commands below.

## Q1: our character TF-IDF logistic model

```powershell
py train.py
```

The first measured model is character 3–5-gram TF-IDF with logistic regression. The script reserves one stratified 2,000-row validation set (seed `42`) and keeps it fixed while searching `C` over `{0.1, 1, 10}`. It selects `C` by validation AUROC. `data/q1/q1_search.csv` records every fit, including training and validation AUROC, error counts at a 0.5 threshold, and fit time. `q1_summary.json` records the split and selected setting. The separate comparison script adds two more approaches, pending a run on the course data.

### Measured Q1 results

| Setting | Train AUROC | Validation AUROC | Train errors / 8,000 | Validation errors / 2,000 |
| --- | ---: | ---: | ---: | ---: |
| C=0.1 | 0.963456 | 0.933322 | 835 | 261 |
| C=1 | 0.994940 | 0.961446 | 294 | 198 |
| C=10 | 0.999897 | 0.968976 | 31 | 180 |

These results use 8,000 training rows and 2,000 validation rows, stratified 4,000/4,000 and 1,000/1,000 by class. They use seed `42`, matching the teammate notebook's `train_test_split` call on the same CSV row order. There were no exact text passages shared between the partitions. We chose `C=10` by validation AUROC. Since the same validation set selected `C`, its score may be somewhat optimistic.

### Additional Q1 comparison, pending data run

`q1_comparison.py` adds two distinct approaches to the existing character model: (A) 30 hand-designed writing-style features with a logistic classifier and Adam optimizer implemented in NumPy, and (B) word unigram/bigram TF-IDF with logistic regression. It also refits the existing character model (C) as a split-alignment check. All three use the same seed-42 stratified 8,000/2,000 partition as `train.py`. A searches L2 strengths `{0.001, 0.01, 0.1}`; B searches logistic `C` values `{0.1, 1, 10}`. Those two searches use three-fold cross-validation on the training rows only. The script flags text overlap across partitions and writes metrics and validation probabilities to `data/q1-comparison/`.

```powershell
py q1_comparison.py
```

On macOS, run `.venv/bin/python q1_comparison.py`. The new approaches have **no measured competition results yet** because their experiment requires the course `train.csv`. Replace the pending Q1 report cells only after reviewing the generated metrics. The prior character baseline's reported scores come from its separate completed run.

## Exploratory improvement: explicit wording cues

The [Wikipedia field guide on signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing) suggests negative parallelism and the rule of three as possible cues. We tested whether counts of `not ... but`, `rather than`, and approximate three-item lists improve our character model. These regex counts are divided by passage length (per 100 words), standardized using training rows only, and appended to the TF-IDF features with a fixed weight of `0.1`. Both runs use the same 8,000/2,000 split and logistic `C=10`.

```powershell
py style_experiment.py
```

| Configuration | Train AUROC | Validation AUROC | Validation errors at 0.5 | Fit time (s) |
| --- | ---: | ---: | ---: | ---: |
| Character TF-IDF alone | 0.999897 | 0.968976 | 180 | 8.07 |
| Character TF-IDF + wording cues | 1.000000 | 0.968779 | 181 | 10.42 |

The added cues reduced validation AUROC by 0.000197, the competition metric, and increased 0.5-threshold validation errors from 180 to 181. We therefore retain the simpler character TF-IDF model as our candidate. In the training partition, `not ... but` appeared in 6.1% of human and 7.8% of AI passages, and `rather than` in 1.5% and 2.9%. Our approximate three-item-list pattern appeared in 53.0% of human and 45.9% of AI passages. This pattern does not capture every rhetorical rule of three; none of these cues establishes authorship by itself. The field guide also notes that some observations are specific to Wikipedia, so transfer to this dataset is uncertain.

## Refit and generate a local Kaggle submission

After reviewing the validation results, refit the selected approach on all 10,000 labeled rows and write 3,000 test probabilities:

```powershell
py train.py --refit-all
py predict.py
```

The CSV columns are `id,label`, where `label` is the predicted probability that the passage is AI-generated. The file remains local until a team member reviews and uploads it. Q2–Q4 analyses are separate from this Q1 script; the Q5 wording-cue ablation runs through `style_experiment.py`.

Running `train.py` again with unchanged code, data, seed, and package versions reproduces the same AUROC and error counts; measured fit time can change with system load. It overwrites the local Q1 result files, and updates `model.joblib` only when `--refit-all` is supplied. Running `predict.py` again overwrites the local `submission.csv` with the same probabilities when the saved model and test data are unchanged. All of these outputs are Git-ignored.

## AI assistance

Codex helped set up the repository and draft the Q1 training and wording-cue experiment scripts. Team members should run the experiments, inspect the data schema and results, review the code, and describe any changes and conclusions in the final report.
