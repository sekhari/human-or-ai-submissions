# Model Improvement Progress & Empirical Diagnostic Report

**Project**: Human or AI? Text Classification (Cornell Tech CS 5785 / Kaggle)  
**Team**: Team 35 (`plum`)  
**Directory**: `submissions/team-35/plum/`  
**Date**: November 2026  
**Primary Metric**: Area Under the ROC Curve (AUROC)  
**Benchmark Reference Baseline**: Stratified 8,000/2,000 Word $(1,2)$ TF-IDF + Logistic Regression (Val AUROC: `0.9367`, Kaggle Benchmark: `~0.868`)

---

## 1. Executive Summary & Current Standings

Across iterative architectural iterations, feature engineering, and model exploration, our submissions on the Kaggle public leaderboard are:

1. **`submission_error_directed_refinement.csv`** — **`0.98640`** (**WINNING BEST KAGGLE SUBMISSION**)
   - *Originating Notebook*: `error_directed_model_refinement.ipynb`
   - *Architecture*: Error-directed refinement of the winning 9-specialist recipe addressing the 20 holdout misclassifications: markdown blockquote unmasking (`> `), proper-noun repetition differentiation, and verse cadence / rhyme regularity modeling for the 432 test poems.
2. **`submission_transformer_hybrid.csv`** — **`0.98575`** (Experiment F: Capstone Contextual Hybridization — *Lower Score*)
   - *Originating Notebook*: `transformer_hybrid_ensemble.ipynb`
   - *Architecture*: Capstone Improvement for Report Section 7 (Open-Ended Improvements). Fuses the winning `0.98640` GBDT ensemble with fine-tuned `distilbert-base-uncased` (66M parameters). While improving in-domain holdout AUROC to `0.99954` and resolving 70% of holdout errors, on the Kaggle public leaderboard it scored `0.98575` (-0.00065). The extreme logit bimodality of the transformer displaced rank calibrations on 21.2% of borderline test passages where GBDT n-grams were more accurate.
3. **`submission_final_model_improvements.csv`** — **`0.98629`** (Previous Benchmark)
   - *Originating Notebook*: `final_model_improvements.ipynb`
   - *Architecture*: 9-Specialist Two-Tier Hybrid Ensemble (Pure Lexical/LM View + Structural View) combining Regularized Logistic Regression, Calibrated LinearSVC, Dual Word/Char N-gram LMs, and GBDT Triad (LightGBM, CatBoost, XGBoost) trained on all 10,000 labeled samples.
4. **`submission_best_recipe_seedavg_3.csv`** — **`0.98604`** (Variance-Reduction Experiment — *Lower Score*)
   - *Originating Notebook*: `best_recipe_seed_average.ipynb`
   - *Architecture*: Exact 3-seed arithmetic average ($\text{seeds} \in \{42, 43, 44\}$) of the 0.98629 recipe. It scored slightly lower than the single-seed fit ($-0.00025$), demonstrating that random seed variance was not the performance bottleneck.
5. **Robustness-Constrained Candidates (`candidate_01`, `candidate_04`)** — **`0.98379`**, **`0.98363`**
   - *Originating Notebook*: `further_model_tuning.ipynb`
   - *Architecture*: Models heavily regularized against synthetic formatting perturbations (`canonical`, `wrap60`, `wrap90`). Over-regularization degraded discriminative signal on Kaggle.

```
Chronological Kaggle Submissions & Leaderboard Progression:
  0.86800  [Kaggle Baseline Benchmark (Course assignment reference)]
  0.93670  [Local Reference Baseline: Word TF-IDF + LogReg]
  0.97430  [Multi-View Linear Baseline]
  0.98379  [candidate_01 (further_model_tuning.ipynb - formatting stress candidate)]
  0.98629  [submission_final_model_improvements.csv (final_model_improvements.ipynb)]
  0.98604  [submission_best_recipe_seedavg_3.csv (best_recipe_seed_average.ipynb) - Lower score]
  0.98640  [submission_error_directed_refinement.csv (error_directed_model_refinement.ipynb) - Current Best]
  0.98575  [submission_transformer_hybrid.csv (transformer_hybrid_ensemble.ipynb) - Capstone Hybrid, -0.00065]
```

---

## 2. Winning Recipe Architecture (`0.98629`)

The winning model (`submission_final_model_improvements.csv`) was developed in `final_model_improvements.ipynb` and parameterized via `models/final_model_improvements_selected.yaml`. It addresses the problem by recognizing that human vs. AI text detection requires two complementary views:

### 2.1 Multi-View Representation
- **Pure Lexical & Subword Views (Formatting-Free)**:
  - Sublinear Word $(1,3)$-gram TF-IDF (85,000 features, capturing topical phrasing and transition discourse markers).
  - Word-boundary Character `char_wb` $(3,5)$-grams (95,000 features, capturing morphology, prefixes, suffixes, punctuation cadence).
  - Masked Token View (stripping stopwords/formatting to isolate syntactical flow).
  - 72 Dense Pure Stylometric Features (lexical diversity `log_ttr`, `root_ttr`, `yules_k`, repetition frequencies, sentence length coefficient of variation `cv_sent_len`, Coleman-Liau readability, POS ratios).
- **Dual N-gram Generative Language Models**:
  - Fold-locally trained Word Trigram and Character 3-gram generative language models (`final_ngram_lm.yaml`) calculating log-likelihood ratio features $\log P(x \mid \text{AI}) - \log P(x \mid \text{Human})$ and per-token perplexity differences.
- **Structural View**:
  - Paragraph counts, whitespace distributions, line length variances, structural entropy, and 24 genre-conditioned stylometric Z-scores.

### 2.2 Nine-Specialist Hybrid Ensemble
The architecture integrates 9 diverse specialist learners split into two decoupled pipelines:
1. **No-Format Specialists (`nofmt_weights`: 0.62 total weight)**:
   - 4-View Logistic Regression (`lr_4v`, weight $0.22$, $C=2.0$)
   - 4-View Calibrated LinearSVC (`svc_4v`, weight $0.22$, $C=0.5$, Platt sigmoid calibration)
   - Masked-Text Logistic Regression (`mask_lr`, weight $0.14$)
   - Pure GBDT Triad blend (`gbdt_pure`, weight $0.42$: LightGBM $450$ trees, CatBoost $850$ trees, XGBoost $450$ trees)
2. **Structural Specialists (`struct_weights`: 0.38 total weight)**:
   - Structural LightGBM (`lgb_struct`, weight $0.30$)
   - Structural CatBoost (`cat_struct`, weight $0.40$)
   - Structural XGBoost (`xgb_struct`, weight $0.30$)

Final probabilities are fused via calibrated logit-space combination and retrained over the complete 10,000 labeled dataset.

---

## 3. Systematic Record of Subsequent Experiments

To push beyond `0.98629`, four focused experimental campaigns were conducted and recorded across dedicated notebooks in this directory:

### Experiment A: Formatting-Stress & Robustness Candidates (`further_model_tuning.ipynb`)
- **Hypothesis**: The competition assignment warns about synthetic distribution shifts (e.g. line-wrap, stripping `<br>`, canonicalizing quotes/dashes). Training models on augmented data and increasing structural invariance should prevent test-time performance collapse.
- **Implementation**:
  - Evaluated 20 distinct branch configurations over 5 grouped cross-validation folds.
  - Implemented real-time formatting stress evaluators: `canonical`, `typography`, `lowercase`, `wrap60`, and `wrap90`.
  - Selected 5 submission candidates (`candidate_01` to `candidate_05`) based on a balanced metric combining original AUROC and stress AUROC.
- **Kaggle Results**:
  - `candidate_01`: **`0.98379`** (vs `0.98629` baseline)
  - `candidate_04`: **`0.98363`**
- **Post-Mortem & Lesson**:
  - Aggressive formatting-invariance regularization significantly damaged test-set discriminative power ($-0.0025$ to $-0.0027$ AUROC).
  - The actual test distribution retains subtle structural and punctuation signals that artificial augmentation smoothed over or penalized excessively.

### Experiment B: Extended Boosting Iteration Scaling (`longer_training_iterations.ipynb`)
- **Hypothesis**: Tree-based gradient boosters (LightGBM, CatBoost, XGBoost) might underfit subtle tail interactions when constrained to 450–850 trees. Scaling iteration budgets ($2\times, 4\times, 8\times$) could improve classification margins.
- **Implementation**:
  - Scaled LightGBM to 3,600 trees, CatBoost to 6,800 iterations, and XGBoost to 3,600 rounds.
  - Monitored full 5-fold iteration curves and verified exact reproduction at the $1\times$ checkpoint.
- **Validation Outcome**:
  - Pooled local development AUROC increased monotonically from `0.99827` ($1\times$) to `0.99837` ($4\times$) and `0.99839` ($8\times$).
  - Mean holdout AUROC slightly improved from `0.999449` to `0.999486`.
- **Post-Mortem & Lesson**:
  - Despite local CV gains, formatting robustness slightly decreased (`0.98049` $\to$ `0.98003`), and submissions produced lower test rankings.
  - Boosting on lexical TF-IDF features simply Memorizes training-domain co-occurrences without improving out-of-domain generalization. Local CV at $\ge 0.998$ suffered severe saturation.

### Experiment D: Multi-Seed Averaging (`best_recipe_seed_average.ipynb`)
- **Hypothesis**: If tree models exhibit stochastic variance across different random seeds (subsampling, feature bagging, CatBoost random permutations), arithmetic probability averaging across independent runs of the winning recipe (`0.98629`) will reduce variance and raise test AUROC.
- **Implementation**:
  - Kept outer folds and all hyperparameter weights identical to the winning recipe.
  - Trained 5 full independent replicates with seeds $42, 43, 44, 45, 46$.
  - Averaged 3 seeds ($\{42, 43, 44\}$) and 5 seeds ($\{42, 43, 44, 45, 46\}$).
  - Verified numerical reconstruction of the seed 42 baseline within $2.2 \times 10^{-16}$.
- **Kaggle Result**:
  - `submission_best_recipe_seedavg_3.csv`: **`0.98604`** (vs `0.98629`, $-0.00025$).
- **Post-Mortem & Lesson**:
  - Seed averaging of tree models produces negligible difference ($\Delta < 0.0003$). Tree variance is *not* the limiting factor. The limiting factor is a fundamental representation and domain gap.

### Experiment E: Error-Directed Model Refinement (`error_directed_model_refinement.ipynb`)
- **Hypothesis**: By directly conducting an autopsy on the 20 held-out validation misclassifications (6 FPs, 14 FNs) and addressing the 28.3% test domain shift, targeted feature engineering and de-biasing will systematically resolve false predictions and improve test AUROC.
- **Implementation**:
  1. **Markdown Blockquote Normalization (`> `)**: Unmasks academic AI text prefixed with `>` (resolving False Negatives `ID: 4951, 1515, 4290`).
  2. **Proper Noun vs. Degenerate Repetition Disambiguation**: Differentiates human Wikipedia lists/tracklists (repeated proper nouns) from AI repetition loops (resolving False Positives `ID: 3660, 4947, 4062, 5079, 7182`).
  3. **Poetic Cadence & Rhyme Regularity**: Models couplet rhyming (`rhyme_aabb`), line syllable variance (`line_cv`), and line capitalization consistency for the 432 test poems.
  4. **Safe Domain Z-Score Calibration**: Centers domain Z-scores for unsupported genres ($Z=0$), preventing spurious tree splits on poetry.
- **Measured Validation Results**:
  - **Holdout AUROC**: Improved from **`0.999449`** $\to$ **`0.999506`** ($+0.000057$).
  - **Holdout Brier Error**: Dropped from `0.009228` $\to$ **`0.008355`** ($-9.5\%$).
  - **False Positives**: Cut from 6 down to **1** (**5 resolved**!).
  - **False Negatives**: Cut from 14 down to **11** (**3 resolved**!).
  - **Total Misclassifications**: Reduced by **40%** (from 20 down to 12).
- **Exported Submission**:
  - `submission_error_directed_refinement.csv` generated, verified, and scored **0.98640** on Kaggle.

### Experiment F: Contextual Transformer Fine-Tuning & Grand Hybrid Ensemble (`transformer_hybrid_ensemble.ipynb`)
- **Motivation & Empirical Grounding (Section 7, Q5.1)**:
  - While classical GBDT ensembles achieved near-perfection on in-domain validation (`0.99951`), Kaggle public leaderboard performance plateaued around `0.98640`.
  - Detailed diagnostic auditing in `validation_gap_diagnosis.ipynb` revealed the mathematical bottleneck: **28.33% of the Kaggle test set consists of Literature & Poetry** (14.40% poetry, 13.93% book summaries), whereas training contained almost zero poetry (only 12 passages total, with just 1 human poem). Classical n-grams and surface stylometrics exhibited complete semantic blindness on unseen verse, erroneously assigning $\hat{P}(\text{AI}) > 0.99$ to classic human poetry (e.g. Kenneth Patchen eulogies, Victorian rhyming verse).
- **Implementation (Section 7, Q5.2)**:
  - **Architecture**: `distilbert-base-uncased` (66M parameters, 6 transformer layers, 768 hidden dimension, 12 attention heads), pre-trained on BookCorpus (over 11,000 unpublished narrative books) and English Wikipedia.
  - **Hardware Acceleration**: Fine-tuned on an **NVIDIA GeForce RTX 5060 Laptop GPU** using `torch-directml` (DirectX 12 backend), achieving 40 ms/sample for full backpropagation (~1.5 ms/sample for inference).
  - **Training Protocol**: 2 epochs on the 8,000 development partition (`lr = 3e-5`, Adam optimizer, batch size 32, max sequence length 256). Total training duration: **764.4 seconds (12.7 minutes)**.
  - **Leakage Prevention**: The 2,000 holdout set (`split_ids.csv`) remained strictly untouched during training.
- **Measured Results (Section 7, Q5.3)**:
  - **Standalone DistilBERT Holdout AUROC**: **`0.98556`** (Brier score: `0.06022`).
  - **Error Recovery Rate**: DistilBERT successfully resolved **14 of the 20 GBDT holdout misclassifications (70.0% recovery rate)**! Specifically, it corrected false human classifications on book summaries (ID 1362: GBDT 0.049 $\to$ BERT 0.699 AI), recipe imitations (IDs 5498, 9298), academic abstracts (IDs 4951, 5010, 1515), and first-person posts (IDs 4290, 3062, 5369).
  - **Holdout Blending Optimization**: Probability blending with $w_{\text{BERT}} = 0.08$ improved holdout AUROC beyond the GBDT baseline to **`0.999543`** ($+0.000094$).
  - **Test Domain Correction**: On the 3,000 Kaggle test set, DistilBERT drastically corrected GBDT false positives on classic human poetry:
    - ID 13058 (Kenneth Patchen poem): GBDT gave `0.9950` AI $\to$ DistilBERT recognized Human with `0.0084` (rank dropped from 0.93 to 0.85).
    - ID 13491 (Frost-kissed dawn poem): GBDT gave `0.9950` AI $\to$ DistilBERT recognized Human with `0.0047` (rank dropped from 0.93 to 0.85).
    - ID 14768 (Cathedral of the wild nature poem): GBDT gave `0.9950` AI $\to$ DistilBERT recognized Human with `0.0095`.
    - ID 13819 (Alexander's splendid feast historical poem): GBDT gave `0.9950` AI $\to$ DistilBERT recognized Human with `0.0112`.
- **Exported Deliverable**:
  - `transformer_hybrid_ensemble.ipynb` and `submission_transformer_hybrid.csv` generated, strictly validated (3,000 rows, IDs 13001..16000, probabilities strictly in $[0, 1]$).

---

## 4. Diagnostic Root Cause: The Massive Test Domain Shift

In `validation_gap_diagnosis.ipynb`, a deep structural audit of the dataset revealed the exact reason why local validation AUROC ($\approx 0.9985$) diverges from Kaggle leaderboard AUROC ($\approx 0.9863$):

### 4.1 Observable Genre Support Matrix

| Genre / Heuristic Domain | Train Human % | Train AI % | Train Total % | Test Set % | **Domain Shift (pp)** |
|---|---:|---:|---:|---:|---:|
| **3. Poetry / Verse / Lyrics** | **0.02%** | 0.22% | **0.12%** | **14.40%** | **$+14.28$ pp** |
| **4. Book / Novel Summary** | 3.12% | 4.88% | **4.00%** | **13.93%** | **$+9.93$ pp** |
| 1. Recipe / Cooking | 20.72% | 20.92% | 20.82% | 12.77% | $-8.05$ pp |
| 2. Academic / ArXiv Abstract | 18.36% | 21.08% | 19.72% | 13.43% | $-6.29$ pp |
| 6. Encyclopedia / Wikipedia | 12.16% | 9.54% | 10.85% | 6.57% | $-4.28$ pp |
| 5. Film / TV Review | 8.32% | 5.52% | 6.92% | 3.93% | $-2.99$ pp |
| 7. First-Person / Social | 17.34% | 17.16% | 17.25% | 13.13% | $-4.12$ pp |
| 8. General / News / Other | 19.96% | 20.68% | 20.32% | 21.83% | $+1.51$ pp |

### 4.2 Key Findings
1. **The Poetry Blind Spot**:
   - In `train.csv` (10,000 rows), Poetry represents only **0.12%** (12 passages total, with only **1 human poem** in the entire training split!).
   - In `test.csv` (3,000 rows), Poetry constitutes **14.40%** (~432 passages)!
   - Sparse N-grams and surface stylometrics trained on recipes and academic abstracts completely fail to generalize to poetic cadence, meter, stanza formatting, and rhyming patterns.
2. **The Book Summary Shift**:
   - Book summaries triple from 4.0% in train to **13.93%** in test.
   - Combined, **Poetry and Book Summaries make up 28.33% of the Kaggle test set**, but only **4.12%** of the training data.
3. **Implication**:
   - Classical tabular models and bag-of-words classifiers have reached their mathematical limit on the supported 71.7% of the test set, but are operating near chance or with biased priors on the remaining 28.3%.
   - Any further hyperparameter tuning on the existing training distribution will only cause further overfitting to the dominant training genres (recipes, arxiv, wikipedia).

---

## 5. Artifact & Experiment Provenance

| Artifact / Notebook | Purpose / Contents | Primary Metric / Score | Status |
|---|---|---|---|
| `final_model_improvements.ipynb` | Canonical 9-specialist engine, multi-view TF-IDF, fold-local training | Kaggle: **0.98629** | Committed (Baseline Best) |
| `submission_final_model_improvements.csv` | Full 10k retrain submission of the winning 9-specialist model | Kaggle: **0.98629** | Committed |
| `further_model_tuning.ipynb` | Robustness-focused tuning across 5 stress modes, 20 configs | Kaggle: 0.98379 | Untracked $\to$ Staged |
| `longer_training_iterations.ipynb` | Tree iteration scaling ($2\times \dots 8\times$ depth/tree budgets) | Dev: 0.99839 / Kaggle: $\downarrow$ | Untracked $\to$ Staged |
| `validation_gap_diagnosis.ipynb` | Root cause discovery of 28.3% test domain shift (Poetry + Books) | Diagnostic Analysis | Untracked $\to$ Staged |
| `best_recipe_seed_average.ipynb` | Multi-seed replicate averaging ($\text{seeds } 42\dots 46$) | Kaggle: **0.98604** | Committed |
| `submission_best_recipe_seedavg_3.csv` | 3-seed probability average CSV export | Kaggle: **0.98604** | Stored in `artifacts/` |
| `error_directed_model_refinement.ipynb` | Autopsy of 20 holdout errors, quote unmasking, rhyme features | Val AUROC: **0.99951**, 40% error drop | Committed (Kaggle: **0.98640**) |
| `submission_error_directed_refinement.csv` | Error-directed refinement submission candidate | Kaggle Public LB: **0.98640** | Current Kaggle Best |
| `transformer_hybrid_ensemble.ipynb` | Capstone Improvement (Section 7): DistilBERT fine-tuning + GBDT fusion | Holdout AUROC: **0.99954**, fixes 70% GBDT errors | Committed (Grand Hybrid) |
| `submission_transformer_hybrid.csv` | Grand Hybrid test submission ($0.92 \cdot \text{GBDT} + 0.08 \cdot \text{BERT}$) | Kaggle: **0.98575** / Holdout: 0.99954 | Completed Experiment |

---

## 6. Next Strategic Directions

See the subsequent brainstorming analysis for our proposed structured experiment loop to break past 0.98629, addressing the domain shift via domain-specific representations, unsupervised feature transfer from test/unlabeled, and pretrained transformer hybridization.
