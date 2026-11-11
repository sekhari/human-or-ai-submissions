# Group Number 7 — Human or AI?

Cornell Tech | CS5785 Applied Machine Learning | Fall 2026

**Kaggle team name:** `Group Number 7`  
**Team members:** Yuchen (Wayne) Yang (`yy2458`); Yuqi (Yokie) Zhang (`yz3657`); Yaoan (Ben) Xie (`yx749`).

## Methods, contributions, and validation results

| Module | Responsibility | Method / purpose | Validation AUROC |
| --- | --- | --- | ---: |
| [`model1/`](model1/) | Yaoan Xie | Word TF-IDF (1–2 grams) + logistic regression; baseline | 0.936066 (original reported) |
| [`model2/`](model2/) | Yuqi Zhang | Character TF-IDF (2–4 grams) + `LinearSVC(C=1)` with 3-fold sigmoid calibration; **selected final model** | **0.970440** (original recorded) |
| [`model3/`](model3/) | Yuchen Yang | Fifteen ConvoKit/spaCy stylometric features + logistic regression | 0.826659 (recorded experiment) |
| [`model2_q2_q3/`](model2_q2_q3/) | Yaoan Xie, using the credited Model 2 implementation | Q2 nested-sample learning curve; Q3 one-round pseudo-labeling | 0.970820 for Q3 exploratory student |
| [`ensemble/`](ensemble/) | Yuchen Yang | Q5 probability averaging using the same 2,000 validation IDs | 0.969829 for three equal-weight models; exploratory 0.972428 for 20/60/20 |
| [`q1_tuning/`](q1_tuning/) | Cross-model report experiment | Q1.3 retrospective train-only CV comparisons | See report and generated CSV |
| [`q4_diagnostics/`](q4_diagnostics/) | Model 2 validation analysis | Q4 text-length bins, AUROC gap, selected errors | See report and generated CSV |

All reported validation scores use the shared 8,000/2,000 training/validation membership with seed 42. The final Kaggle entry is the **standalone Model 2**, not the Q3 student or validation-tuned ensemble. For the selected Model 2, the recorded Kaggle public AUROC was **0.85048**; that score is not a replacement for the unseen private evaluation. Small score differences may arise with changed Python/scikit-learn environments. The reconstructed Model 3 notebook is a backed-up reconstruction, not the original executed Colab file.

## 1. Obtain data locally (never commit it)

Download `train.csv` (10,000 labeled rows), `test.csv` (3,000 rows), `sample_submission.csv`, and (for Q3) `unlabeled.csv` (3,000 rows) from the [official Kaggle competition](https://www.kaggle.com/competitions/cornell-tech-cs-5785-combined-2026-ai-vs-human-text/data). Keep the original row order and CSV bytes. Use the absolute path to your **private** Kaggle data folder in the commands below; for example, replace `/absolute/path/to/kaggle-data` with the actual directory containing these CSVs.

This public repository contains no downloaded articles, hidden labels, API credentials, fitted checkpoints, or validation-level predictions. Only the `model2_q2_q3/config/shared_split.csv` file contains **ID-only** train/validation assignments; these are not article text or labels.

## 2. Python setup

Run commands **from `submissions/group-number-7/`**, not from the repository root:

```bash
cd submissions/group-number-7
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

This core environment supports `model1`, `model2`, `ensemble`, `q1_tuning`, and `q4_diagnostics`. For exact historical reproduction, the selected Model 2 original run used Python **3.11.5**, NumPy **2.4.6**, pandas **3.1.1**, SciPy **1.17.1**, scikit-learn **1.9.1**. The Model 1 original run used Python **3.12.6**, and the Q2/Q3 module includes its own pinned [`requirements.txt`](model2_q2_q3/requirements.txt) tested under Python **3.12.6**. Model 3 is a Colab notebook with its own [`requirements.txt`](model3/requirements.txt), spaCy model, and NLTK data. Environments were not fully identical between collaborators.

## 3. Evaluate the final Model 2 on the agreed holdout

The entrypoint is `model2/run.py`, which fits TF-IDF **inside** each training/calibration fold. It does not use validation or Kaggle test labels during fitting.

```bash
python model2/run.py evaluate \
  --data-dir /absolute/path/to/kaggle-data \
  --outdir outputs/model2_eval
```

Outputs include `outputs/model2_eval/train_predictions.csv`, `validation_predictions.csv`, and `run_info.json`. The prediction files contain `id`, **true validation/training** `label`, SVM `margin`, calibrated `probability` (P(AI)), and `word_count`. The canonical split is a stratified original-row-order 80/20 split with seed 42; no new validation IDs are selected on a rerun.

## 4. Generate the **final Kaggle `submission.csv`**

```bash
python model2/run.py submit \
  --data-dir /absolute/path/to/kaggle-data \
  --outdir outputs/model2_final
```

**Output:** `outputs/model2_final/submission.csv` with exactly `id,label`, where `label` is **P(AI)** for every official test ID. This command refits on **all 10,000 labeled training rows**, using the same `char` 2–4 TF-IDF, `LinearSVC(C=1)`, three stratified sigmoid-calibration folds, and seed 42. This script **writes predictions locally; it does not upload to Kaggle**. Use a **new** `--outdir` for each rerun (the scripts intentionally refuse to overwrite existing runs).

## 5. Reproduce additional models and required experiments

Run these from the same group directory and install the core requirements above.

**Model 1 (baseline; not final Kaggle submission):**

```bash
python model1/run.py evaluate --data-dir /absolute/path/to/kaggle-data --outdir outputs/model1_eval
python model1/run.py submit --data-dir /absolute/path/to/kaggle-data --outdir outputs/model1_final
```

Model 1 uses word 1–2 grams, `min_df=2`, a 50,000-feature limit, and logistic regression `C=1`, seed 42. The original Model 1 validation AUROC was 0.936066; a later cross-environment reproduction yielded approximately 0.936077.

**Q1.3 training-only hyperparameter comparisons:**

```bash
python q1_tuning/run.py --data-dir /absolute/path/to/kaggle-data --outdir outputs/q1_cv
```

Compares Model 1's `C` values 0.1, 1, 10 and uncalibrated Model 2 SVM's `(2–4 grams,C=0.5)`, `(2–4,C=1)`, `(3–5,C=1)` with 3-fold training-only AUROC. **This is a retrospective comparison**, not the historical tuning/selection of Model 2; it does **not** use the fixed 2,000 validation labels to select hyperparameters, and it does not replace the selected final calibrated Model 2.

**Q2/Q3 learning curve and pseudo-labeling:** use the separate pinned environment from [`model2_q2_q3/README.md`](model2_q2_q3/README.md). To run from the group directory after installing the module's pinned requirements in a dedicated environment:

```bash
python model2_q2_q3/run_q2.py --data-dir /absolute/path/to/kaggle-data --output-dir outputs/q2
python model2_q2_q3/run_q3.py --data-dir /absolute/path/to/kaggle-data --q2-dir outputs/q2 --output-dir outputs/q3
python model2_q2_q3/verify.py --data-dir /absolute/path/to/kaggle-data --q2-dir outputs/q2 --q3-dir outputs/q3
```

The Q2 run fits balanced nested 500/1,000/2,000/5,000/8,000 subsamples on the same validation set. Q3 uses confidence threshold `0.95` and one pseudo-labeling round; observed validation AUROC was 0.970440 (supervised) vs. 0.970820 (pseudo-labeling). The Q2/Q3 scripts require original-file SHA-256 checks and the **tracked** ID-only manifest [`model2_q2_q3/config/shared_split.csv`](model2_q2_q3/config/shared_split.csv); keep the original Kaggle CSV bytes and do not delete the manifest. They may take multiple minutes to run.

**Q4 generalization and error analysis:**

```bash
python q4_diagnostics/run.py \
  --train outputs/model2_eval/train_predictions.csv \
  --validation outputs/model2_eval/validation_predictions.csv \
  --outdir outputs/q4_analysis
```

This produces the training-validation AUROC gap, four text-length bins (`<100`, `100–199`, `200–299`, `>=300` whitespace-delimited words), and 5 highest-confidence FP + 5 highest-confidence FN IDs with probabilities. Output includes local **row-level** `error_cases_LOCAL_ONLY.csv`; the qualitative explanations in the report are hypotheses, not model-attribution proofs.

**Model 3 (stylometric baseline):** open [`model3/model3_convokit_stylometry.ipynb`](model3/model3_convokit_stylometry.ipynb) in Google Colab, supply the official CSVs privately, and run the notebook in order. It uses ConvoKit + spaCy/NLTK parsing, a 15-feature standardized classifier, and seed 42. Feature extraction is much slower than its logistic-regression fit (~554 seconds for 8,000 training + 2,000 validation passages in the recorded run). This recovered notebook was **not** newly rerun end-to-end during packaging; consult the [`model3/README.md`](model3/README.md) for limitations.

**Q5 equal-weight and exploratory probability ensembles:** after creating the three **matching-validation-ID** probability files (Model 3's notebook yields `results/model3_val_predictions.csv` in Colab), run:

```bash
python ensemble/run.py \
  --model1 outputs/model1_eval/validation_predictions.csv \
  --model2 outputs/model2_eval/validation_predictions.csv \
  --model3 /absolute/path/to/model3_val_predictions.csv \
  --outdir outputs/q5_ensemble
```

The script verifies ID and true-label agreement and writes summary AUROC results. The 20/60/20 mixture's 0.972428 AUROC is **exploratory** because its weights were assessed on the same validation set; it was not chosen for the final competition submission. Outputs with per-row validation labels/predictions stay local. Do not rerun `ensemble/run.py` with Kaggle test labels (they are unavailable).

## 6. Reproducibility, source attribution, and AI disclosure

- **Seeds and splits:** primary split seed **42**, Model 1/2 classifiers seed **42**, Model 2 calibration folds seed **42**, Q1 3-fold CV seed **42**, Q2/Q3 feature/selection seeds **42**. The fixed validation split has 8,000 fitting and 2,000 held-out labeled rows; final Kaggle prediction fits all 10,000 labeled rows.
- **Source provenance:** Model 1 is adapted from Yaoan Xie's [verified repository](https://github.com/yx749-lab/cornell-tech-cs5785-model1); Model 2 from Yuqi Zhang's [reproduction repository](https://github.com/yokiezz/cornell-tech-ai-human-model2), which credits an earlier `JD-TB1` source revision. The Q2/Q3 extension is attributable to Yaoan Xie with Model 2's underlying source attribution preserved in [`MODEL2_PROVENANCE.md`](model2_q2_q3/MODEL2_PROVENANCE.md). The Model 3 notebook is reconstructed from Yuchen Yang's previously recorded experiment. These attributions are intentional and must not be removed.
- **AI assistance:** ChatGPT and OpenAI Codex assisted with code construction/refactoring, debugging, experiment review, reproducibility checks, explanatory documentation, and report drafting. Human involvement included checking Kaggle file formats, model and split settings, matching validation IDs and labels, AUROCs, error analyses, and the decision not to use an unconfirmed ensemble gain. The individual participants must review and confirm their own tool-use descriptions before final submission. No external LLM API was used to label or predict the hidden competition test examples.
- **Evidence limits:** Original and later reproduced scores differ very slightly across environments. The Q1 CV search and unequal ensemble weighting were assessed retrospectively; do not treat them as independent Kaggle/generalization evidence. The local Model 3 notebook is a reconstruction and does not include the original Colab execution outputs. The Q2/Q3 modules have recorded original runs but were not exhaustively rerun during this group packaging.
- **Development history:** Preserve commits for initialization, Model 2 integration, Model 1/3/ensemble integration, moving code into the required team folder, Q2/Q3 inclusion, and the fixed split manifest. Do not squash away this history before the course review. A final pull request should target `sekhari/human-or-ai-submissions:main` from `WayneYYC/human-or-ai-submissions:submission/group-number-7`.
- **Model 2 Development and Reproducibility**:
Yuqi (Yokie) Zhang maintained a separate Model 2 GitHub repository documenting the attributed SVM implementation, local reproduction, experiment records, and generalization analysis. https://github.com/yokiezz/cornell-tech-ai-human-model2
 The repository includes validation results (AUROC: 0.970440), reproducibility instructions, and commit history from November 8, 2026. The original implementation and inherited experiments are explicitly credited.

## Submission checklist

- [ ] Confirm registered Kaggle team name is exactly `Group Number 7` and the final Model 2 CSV was uploaded to Kaggle.
- [ ] Verify the commands above on the official local competition files.
- [ ] Confirm each team member's contribution and AI-assistance statements.
- [ ] Open **one** pull request to the course repository, and insert the actual PR URL on the report title page and in the course submission portal.

**Do not push Kaggle competition CSVs, secrets, hidden labels, raw per-example prediction files, or trained model checkpoints to this public repository.**
