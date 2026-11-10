# Team 35: Human or AI? Text Classification
**Cornell Tech | Applied Machine Learning (CS 5785)**

- **Team Members**:
  - Zhitong Liu (`zl2358`)
  - Sorathorn Thongpitukthavorn (`st2329`)
  - Keyan Miao (`km2299`)
- **Kaggle Team Name**: `team 35` (registered on Kaggle as `Group Number 35`)
- **Final Public Leaderboard AUROC**: **`0.98640`** ([`submission.csv`](submission.csv))
- **Primary Executable Entrypoint**: [`error_directed_model_refinement.ipynb`](error_directed_model_refinement.ipynb)

---

## 1. Environment & Dependencies

- **Python Version**: **Python 3.12** (laptop-friendly, fully runnable on standard x86_64 CPU)
- **Hardware**: Standard multi-core CPU (GPU not required; optional CUDA/DirectML used for exploratory testing only)

### Installation
From the repository root:
```bash
python3.12 -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

python -m pip install -r submissions/team-35/requirements.txt nbconvert ipykernel
```

### Data Placement
Ensure competition datasets are placed in `./data/` relative to the repository root:
```
data/
├── train.csv                 # 10,000 labeled examples (5,000 human, 5,000 AI)
├── unlabeled.csv             # 3,000 unlabeled examples
├── test.csv                  # 3,000 evaluation examples
└── sample_submission.csv     # Template CSV (id, label)
```

---

## 2. Exact Command that Writes `submission.csv` for Kaggle

To regenerate the exact winning Kaggle submission (`submission.csv`, scoring **`0.98640`** on the public leaderboard):

```bash
# 1. From repository root, navigate to team directory:
cd submissions/team-35

# 2. Fit base 9-specialist ensemble across all 10,000 labeled rows & cache artifacts:
jupyter nbconvert --to notebook --execute --inplace final_model_improvements.ipynb

# 3. Execute error-directed quote and verse calibrations across test predictions:
jupyter nbconvert --to notebook --execute --inplace error_directed_model_refinement.ipynb

# 4. Copy the refined output to submission.csv:
cp submission_error_directed_refinement.csv submission.csv
```

**What this does**:
1. `final_model_improvements.ipynb` fits the 9-specialist ensemble (linear models, word/character generative n-gram LMs, LightGBM, CatBoost, and XGBoost) on all 10,000 labeled training passages with semi-supervised training turned off, generating cached Stage-1 test predictions.
2. `error_directed_model_refinement.ipynb` loads those base predictions and applies automated calibration checks:
   - Strips leading Markdown quote markers (`> `) to expose AI technical abstracts.
   - Distinguishes capitalized proper-noun lists from degenerative AI token repetition loops.
   - Evaluates syllable consistency, meter regularity, and end-rhyme patterns in poetry/verse.
3. Writes `submission_error_directed_refinement.csv` (and copies to `submission.csv`), containing all 3,000 test IDs with well-calibrated probabilities in $[0, 1]$.

---

## 3. Exact Commands to Train & Evaluate on Labeled Validation Protocol

Our primary validation protocol uses a fixed, stratified **2,000-row held-out partition** (1,000 human, 1,000 AI; seed 42) from `train.csv`, reserving 8,000 rows for training.

### A. Final 9-Specialist Base Ensemble (§6 & §7)
Trains the 9 specialists on 8,000 training rows and evaluates on the 2,000-row holdout:
```bash
cd submissions/team-35
jupyter nbconvert --to notebook --execute --inplace final_model_improvements.ipynb
```
- **Holdout Validation AUROC**: **`0.999449`** (Training AUROC: `1.000000`, gap: `0.000551`).
- Identifies exactly 20 misclassifications at threshold 0.50 (6 False Positives, 14 False Negatives).

### B. Error-Directed Model Refinement Evaluation (§7 & §8)
Evaluates the targeted structural calibrations on the held-out validation set:
```bash
cd submissions/team-35
jupyter nbconvert --to notebook --execute --inplace error_directed_model_refinement.ipynb
```
- **Diagnostic Holdout AUROC**: **`0.999506`** (resolves 8 mistakes: 20 errors $\to$ 12 errors; 6 FP $\to$ 1, 14 FN $\to$ 11).

### C. Word TF-IDF Baseline Models (§3)
Keyan Miao's baseline script comparing word unigram/bigram TF-IDF with Logistic Regression, Linear SVM, and Naive Bayes on the 8,000 / 2,000 validation split:
```bash
python submissions/team-35/km2299/baseline_models.py --train data/train.csv --out submissions/team-35/km2299/results --tune
```
- Model A (Logistic Regression): Val AUROC `0.9367`
- Model A2 (Multinomial NB): Val AUROC `0.8966`
- Model C (Linear SVM): Val AUROC `0.9766`

### D. Character TF-IDF, Learning Curves & Pseudo-Labeling (§4 & §5)
Zhitong Liu's baseline and empirical data-size / semi-supervised scripts:
```bash
# Model F Character TF-IDF baseline (C=10, Val AUROC: 0.968976):
python submissions/team-35/zl2358/train.py --train data/train.csv

# Learning curve (N=500 -> 8,000) & pseudo-labeling (1,112 rows, Val AUROC: 0.968447):
python submissions/team-35/zl2358/baseline_analysis.py --train data/train.csv --unlabeled data/unlabeled.csv
```

---

## 4. Random Seeds & Key Configurations

- **Primary Random Seed**: Pinned to **`42`** across all train/test splits, cross-validation folds, feature tokenizers, and tree model initializations (`random_state=42`).
- **Secondary Random Seed**: **`43`** used for secondary stability checks and multi-split variance validation.
- **Validation Partition**: Fixed stratified 8,000 train / 2,000 validation split (`split_ids.csv`). No validation or test labels were used in feature extraction or vocabulary generation.
- **YAML Configuration Files**: Modular hyperparameters are defined in `submissions/team-35/models/`:
  - `final_model_improvements.yaml` & `final_model_improvements_selected.yaml`: Master ensemble parameters.
  - `final_ngram_lm.yaml`: Generative word and character n-gram language models.
  - `final_nb_logreg.yaml`: Regularization strengths ($C$) and loss configurations for linear specialists.
  - `model_selection_v3_high_score.yaml`: v3 multi-view weights and calibration settings.
- **Semi-Supervised Learning Protocol**: Semi-supervised pseudo-labeling on `unlabeled.csv` was empirically evaluated in Section 5 (AUROC decreased by $-0.000529$ due to label noise) and was deliberately disabled for the final submission.

---

## 5. AI-Tool Disclosure Summary (Matching Report Q6.3)

In accordance with competition rules and matching **Question 6.3** of our team report:

> *We used AI assistants, including ChatGPT/Codex, to brainstorm approaches, draft code, suggest improvements, and help format and organize files and report text. The team chose the direction of the experiments, reviewed and changed suggested code and wording, checked the reasoning, and validated the reported results. We compared numerical claims with saved notebook outputs and Kaggle score records rather than treating AI suggestions as measurements.*

- **Educational & Evaluation Integrity**:
  - No commercial/paid LLM APIs (e.g., OpenAI API, Anthropic API) were used to directly infer, classify, or predict test set labels.
  - No test labels were retrieved from external sources, nor was manual hand-labeling performed.
  - All submitted models and pipelines are fully reproducible locally on CPU using open-source packages.

---

## 6. Directory Structure & File Map

```
submissions/team-35/
├── README.md                                  # Master team reproduction guide (this file)
├── requirements.txt                           # Unified Python 3.12 dependencies
├── .gitignore                                 # Git ignore rules
├── submission.csv                             # Final winning Kaggle submission (AUROC: 0.98640)
│
├── [Core Deliverable Notebooks]
│   ├── model_selection_v3_high_score.ipynb    # §3: v3 multi-view ensemble (Kaggle: 0.98552)
│   ├── validation_gap_diagnosis.ipynb         # §6: Error diagnostics, length bins & domain shift
│   ├── final_model_improvements.ipynb         # §7: 9-specialist hybrid ensemble (Kaggle: 0.98629)
│   ├── error_directed_model_refinement.ipynb  # §7, §8: Final winning model refinement (Kaggle: 0.98640)
│   └── transformer_hybrid_ensemble.ipynb      # Exploration: DistilBERT hybrid testing (Kaggle: 0.98575)
│
├── models/                                    # Modular YAML hyperparameter configurations
│   ├── final_model_improvements.yaml
│   ├── final_ngram_lm.yaml
│   ├── final_nb_logreg.yaml
│   └── ...
│
├── report/                                    # Course report deliverables
│   ├── team_35_report.tex                     # LaTeX source for completed team report
│   ├── team_35_report.pdf                     # Compiled PDF report (exactly 11 pages)
│   └── figures/
│       └── baseline_learning_curve.pdf        # §4: Learning curve plot
│
├── km2299/                                    # Keyan's word TF-IDF baseline scripts (§3)
│   ├── baseline_models.py
│   ├── requirements.txt
│   └── README.md
│
├── zl2358/                                    # Zhitong's individual baseline scripts (§4, §5)
│   ├── train.py
│   ├── baseline_analysis.py
│   └── writeup/
│       ├── seed42_results.json
│       └── baseline_analysis_results.json
│
└── plum/                                      # Plum's multi-view exploration & logs
```

---

## 7. Report Section Mapping & Progress Milestones

| Report Section | Focus | Runnable File / Entrypoint | Methodology & Role in Progression | Key Metric |
| :--- | :--- | :--- | :--- | :--- |
| **§2** | Final Model Summary | [`submission.csv`](submission.csv) | Error-directed refinement of 9-specialist ensemble | Kaggle **0.98640** |
| **§3** | Model Comparison & Selection | [`model_selection_v3_high_score.ipynb`](model_selection_v3_high_score.ipynb) | v3 multi-view ensemble (word/char TF-IDF, style, topic-masking) | Kaggle 0.98552 |
| **§4** | Training Data Size | [`zl2358/baseline_analysis.py`](zl2358/baseline_analysis.py) | Character TF-IDF learning curve ($N = 500 \to 8,000$) | Val 0.904 $\to$ 0.969 |
| **§5** | Unlabeled Data | [`zl2358/train.py`](zl2358/train.py) | Pseudo-labeling (1,112 rows at $p \le 0.05 / p \ge 0.95$) | Val 0.96845 (-0.0005) |
| **§6** | Generalization & Error Analysis | [`validation_gap_diagnosis.ipynb`](validation_gap_diagnosis.ipynb) | Holdout error autopsy, length bins, formatting sensitivity | 40% error drop |
| **§7 & §8** | Open-Ended Improvements | [`final_model_improvements.ipynb`](final_model_improvements.ipynb) | 9-specialist hybrid with dual generative n-gram LMs & GBDTs | Kaggle 0.98629 |
| **§7 & §8** | Final Winning Entrypoint | [`error_directed_model_refinement.ipynb`](error_directed_model_refinement.ipynb) | Blockquote unmasking (`> `) & rhyme/verse cadence features | Kaggle **0.98640** |

---

## 8. Chronological Kaggle Submissions & Progression

```
0.86800  [Course baseline benchmark: Word TF-IDF + Logistic Regression]
0.93670  [Local Reference Baseline: Word TF-IDF + Logistic Regression (val AUROC)]
0.98379  [Synthetic formatting stress candidate (further_model_tuning.ipynb)]
0.98552  [model_selection_v3_high_score.ipynb (v3 multi-view ensemble)]
0.98629  [final_model_improvements.ipynb (9-specialist hybrid ensemble)]
0.98604  [best_recipe_seed_average.ipynb (3-seed replicate test)]
0.98640  [error_directed_model_refinement.ipynb (submission.csv - WINNING BEST SUBMISSION)]
0.98575  [transformer_hybrid_ensemble.ipynb (DistilBERT hybrid test - lower score)]
```
