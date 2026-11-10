# Group Number 30 — Human or AI?

Cornell Tech CS 5785, Fall 2026. All three days are integrated and submitted.

- **Course submission:** [PR #4](https://github.com/sekhari/human-or-ai-submissions/pull/4), from the course fork's `main` to the original course repository's `main`.
- **Final report:** [PDF](report/output/pdf/Group30_Final_Report.pdf) · [LaTeX source](report/main.tex). This is the only current report.
- **Gradescope:** final 12-page PDF submitted November 9, 2026, with all three members and all 21 question assignments verified.
- **Kaggle:** team **Group Number 30**, public AUROC **0.91312**, submission selected for final scoring. Private AUROC is not yet revealed.
- **Verification:** [final run record](results/final_verification.json) · [submission handoff](FINAL_HANDOFF.md). Twenty unit tests passed during final assembly.

## Team and ownership

| Member | NetID | Work |
| --- | --- | --- |
| Divyesh Thirukonda Gowri Shankar | dt579 | Day 1: model comparison, learning curves and structural hypotheses |
| Anusha Ramachandrareddy | ar2664 | Day 2: pseudo-labeling and compact blend experiments |
| Shenye Wang | sw2486 | Day 3: subgroup/error analysis and report integration |

## Final model and measured results

The final model averages character TF-IDF logistic regression (C=3) and word TF-IDF plus 28 style/POS features logistic regression (C=1), with weights 0.5/0.5. Its runnable configuration is [results/recommended_config.json](results/recommended_config.json). Character-only C=3 in `results/selected_config.json` is the fallback and learning-curve configuration.

Final reproduced validation AUROC is **0.978162** on the fixed 2,000-row holdout. The separate Day 3 refit/analysis recorded **0.978174**; its predictions are not identical. Final test probabilities match Day 1 exactly. Local validation and Kaggle public scores use different datasets and are not interchangeable. The validation split was reused for model selection, so its score is not an independent test estimate.

Day 2's 11-feature compact blend is distinct from the selected 28-feature blend. Its pseudo-label experiment did not improve validation AUROC. Detailed comparisons, length-bin analysis and ten error cases are in the final report and recorded evidence.

## Data and environment

Download the official files from the [Kaggle Data tab](https://www.kaggle.com/competitions/cornell-tech-cs-5785-combined-2026-ai-vs-human-text/data) with your authorized account. Put them in the repository-root `data/` directory:

```text
data/train.csv                 10,000 rows: id,text,label
data/unlabeled.csv              3,000 rows: id,text
data/test.csv                   3,000 rows: id,text
data/sample_submission.csv      3,000 rows: id,label
data/train_starter_2000.csv      2,000-row subset of train.csv
```

Label 0 means human; label 1 means AI. Predictions are probabilities of AI authorship. The starter subset is not extra training data. Dataset files, credentials and fitted weights are excluded from Git. See [DATA_NOTES.md](DATA_NOTES.md) for the data audit and limitations.

Use **Python 3.12.14** and **requirements-day3.txt** for the final reproduction environment. `requirements.txt` and `results/environment.json` preserve the earlier Day 1 environment record; they are not the final setup instructions. No GPU or external LLM prediction service is needed.

From `submissions/group-number-30/`:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-day3.txt
.venv/bin/python -m nltk.downloader -d ../../data/nltk_data averaged_perceptron_tagger_eng
export NLTK_DATA="$PWD/../../data/nltk_data"
.venv/bin/python -m unittest discover -s tests -v
```

The pretrained English POS tagger annotates syntax; it does not supply authorship labels. Its recorded hash and package versions are in `results/final_verification.json`.

## Reproduce evaluation and generate submission.csv

Run from the same team directory with the environment above. Use fresh output paths: scripts deliberately preserve existing measurements and artifacts.

```bash
mkdir -p artifacts/final_validation
cp results/split.json artifacts/final_validation/split.json
.venv/bin/python train.py --train ../../data/train.csv --config results/recommended_config.json --results artifacts/final_validation
.venv/bin/python analyze_day3.py --train ../../data/train.csv --predictions artifacts/final_validation/verify/selected.predictions.tsv --out artifacts/final_validation/analysis
.venv/bin/python train.py --train ../../data/train.csv --config results/recommended_config.json --full-data --model-out artifacts/final_submission.joblib
.venv/bin/python predict.py --test ../../data/test.csv --sample-submission ../../data/sample_submission.csv --model artifacts/final_submission.joblib --out artifacts/submission.csv
```

Evaluation reuses the saved ordered 8,000/2,000 split, seed **20261023**. Full-data fitting uses all 10,000 labeled rows. Prediction checks unique test/template IDs, template order and probability bounds, and writes 3,000 `id,label` rows. These commands generate the CSV locally; they do not upload it.

## Reproduce the earlier experiments

The daily handoffs are historical experiment records. Their pending-task and submission statements describe their original dates, not current status:

- [Day 1 handoff](HANDOFF.md): core models, learning curves and structural ablations.
- [Day 2 handoff](DAY2_HANDOFF.md): pseudo-labeling and compact-style blending; [notebook](DAY2_Q3_Q5_Q6.ipynb).
- [Day 3 handoff](DAY3_HANDOFF.md): refit, length-bin analysis, errors and replay.

```bash
mkdir -p artifacts/day1_reproduction
cp results/split.json artifacts/day1_reproduction/split.json
cp results/extension_plan.json artifacts/day1_reproduction/extension_plan.json
.venv/bin/python inspect_data.py --data ../../data --results artifacts/day1_reproduction
.venv/bin/python run_day1.py --data ../../data --results artifacts/day1_reproduction --bonus --blend
.venv/bin/python run_day2.py --data ../../data --out artifacts/day2_reproduction
```

Every controlled run uses seed **20261023**, the same balanced 8,000/2,000 split, and two numerical-library threads. Ordered split IDs and input hashes are persisted. Learned vocabulary, IDF and scaling use training data only. Learning curves use nested stratified subsets of 500, 1,000, 2,000, 5,000 and 8,000 training rows, refitting preprocessing at each size while keeping the selected hyperparameters fixed.

The predeclared core comparison is:

| Approach | Representation | Candidate settings |
| --- | --- | --- |
| Word logistic regression | Word 1–2 grams; minimum document frequency 2; 60,000-feature cap | C = 0.3, 1, 3 |
| Character logistic regression | Character 3–5 grams; minimum document frequency 3; 80,000-feature cap | C = 0.3, 1, 3 |
| Complement Naive Bayes | Same word representation | alpha = 0.1, 1, 3 |
| Calibrated linear SVM | Same word representation | C = 1, three seeded stratified calibration folds, sigmoid |

TF-IDF uses lowercase text, sublinear term frequency, smoothed IDF and L2 normalization. Word tokenization retains tokens of at least two word characters, including stop words. Calibration uses only the outer training partition; internal calibration folds share its vocabulary and IDF. Maximum full-holdout AUROC selects the winner, with first-listed candidates breaking exact ties. Log loss, runtime and source-overlap sensitivity are diagnostics, not alternative selection metrics.

Optional fixed-C=1 ablations compare 28 measured style/POS features, POS 1–3 grams, class-conditional POS trigram surprise, and word TF-IDF combined with style features at fixed weights 1 and 0.1. The class-conditional feature model uses cross-fitted training features. It tests the handwritten sequence idea statistically; it is not a neural next-word model or a grammatical-error detector. Paired bootstrap comparisons use word logistic regression C=1 as the baseline. Their intervals are exploratory, conditional on fitted models, and do not account for training variability, model search, multiple comparisons or correlated sources.

`--resume` is available for interrupted reproduction runs with matching provenance and model files. The checked-in `results/` is a compact evidence set, not a resumable working directory. Original core, curve and bonus records predate resume hardening and ensemble support; their fingerprint intentionally differs from current scoring code. Use a fresh results folder to reproduce them. Resume is an operational convenience, not an independent reproducibility experiment.

## Build the final report

From the team directory, refresh the learning-curve figure from saved measurements if needed:

```bash
.venv/bin/python run_day1.py --results results --plot-only --figure-out report/images/learning_curve.pdf
cd report
mkdir -p output/pdf
pdflatex -interaction=nonstopmode -halt-on-error -jobname=Group30_Final_Report -output-directory=output/pdf main.tex
pdflatex -interaction=nonstopmode -halt-on-error -jobname=Group30_Final_Report -output-directory=output/pdf main.tex
```

## Evidence and development history

`results/` contains the fixed split, configurations, aggregate comparisons and validation predictions. `results/day2/` and `results/day3/` contain the later experiments and analysis. `results/day1-evidence.zip` preserves original compact Day 1 records with a SHA256 manifest. `results/software_verification.json` records the historical 18-test check; `results/final_verification.json` records the final 20-test check and submission status.

Substantive commits retain authors and dates in the course fork. The final report's Q6.2 links to those public commits. Workspace Day 2 PR #1 and Day 3 PR #2 are merged; course PR #4 remains open for staff review. The obsolete root Day 1 report was removed from the working repository; earlier versions remain available in Git history.

## AI assistance

Codex/ChatGPT assisted with implementation, experiment execution, programmatic checks, error interpretation and report drafting/integration. The team confirmed day ownership as listed above. Programmatic checks do not establish independent human verification of every result; Q6.3 states this distinction. No external LLM API was used to label or predict competition passages, and hidden test labels were not retrieved.
