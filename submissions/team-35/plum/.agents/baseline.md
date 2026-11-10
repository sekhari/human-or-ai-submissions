# Baseline Model & Report Sections 3 and 4 (`plum/baseline.md`)

## 1. Our Baseline Model & Performance Summary

Our primary reference baseline (**Model A**) reproduces the course benchmark setup on the fixed stratified **8,000-train / 2,000-validation split** (`RANDOM_SEED = 42`):

- **Model Architecture**: L2-Regularized Logistic Regression (`C = 1.0`, `max_iter = 1000`).
- **Text Representation**: Word $(1, 2)$-gram TF-IDF (`min_df = 3`, `max_features = 50,000`, `sublinear_tf = True`).
- **Baseline Performance**:
  - **Training AUROC (8,000 rows)**: `0.9932`
  - **Validation AUROC (2,000 held-out rows)**: **`0.9367`**
  - **Formatting-Stripped Validation AUROC** (`\n\n` and `<br>` removed): `0.9348`
  - **Test-Domain-Weighted Validation AUROC**: `0.9273`
  - **Validation Brier Calibration Error**: `0.1235`
  - **Validation Classification Errors (at threshold 0.50)**: `341 / 2,000` (`17.05%` error rate)
  - **Training Runtime**: `6.2s` on CPU
  - *(Reference Kaggle Private AUROC noted in assignment template for Word TF-IDF + Logistic Regression: `~0.868`)*

*(We also tested a secondary generative baseline, **Model A2: Word $(1,2)$ TF-IDF + Complement Naive Bayes**, which achieved `0.9790` Train AUROC and `0.8966` Validation AUROC in `0.02s`.)*

---

## 2. Section 3: Model Comparison and Selection (15 points)

### **Q1.1 [3 points]** — How did you preprocess and represent the text? Specify tokenization, normalization, word/character features or embeddings, and any feature extraction choices. Why?

We engineered a **three-view formatting-invariant representation** combining sparse lexical/subword n-grams with dense stylometric features:
1. **Normalization**: Exploratory analysis revealed that HTML `<br>` tags ($3.9\%$ of Human vs. $0.2\%$ of AI texts) and double newlines (`\n\n`, $0\%$ of Human vs. $66\%$ of AI training texts) act as corpus collection shortcuts that shift in `test.csv`. We stripped `<br>` tags and normalized whitespace before n-gram tokenization.
2. **Word & Character TF-IDF**: We extracted sublinear ($1+\log\text{tf}$) **Word $(1,3)$-grams** (`min_df=3`, 75k features) to capture AI discourse markers (*"in addition to"*, *"results show that"*) and word-boundary **Character `char_wb` $(3,5)$-grams** (`min_df=5`, 75k features) to capture morphological suffixes (`-tion`, `-ly`, `-ing`) and punctuation cadence across unfamiliar test topics (e.g., Poetry and Novel Summaries).
3. **Pure & Bimodal Stylometrics**: We extracted **65 pure stylometric features** (lexical diversity `log_ttr`, `root_ttr`, `yules_k`, `rep_3gram`, burstiness `cv_sent_len`, readability, and POS/function-word densities) plus **7 IQR-scaled absolute deviation features** ($|x - \text{median}_{\text{Human}}|/\text{IQR}_{\text{Human}}$) to capture AI's bimodal lexical diversity ($3\times$ higher TTR standard deviation than Human text). Representation ablation confirmed monotonic gains: Word $(1,2)$ (`0.9367`) $\to$ Word $(1,3)$ + Char $(3,5)$ (`0.9681`) $\to$ Full Multi-View (`0.9743`).

---

### **Q1.2 [5 points]** — Which three or more modeling approaches did you try? Keep the labeled validation partition fixed across models.

*All models were trained on the exact same 8,000-row training split and evaluated on the fixed 2,000-row stratified validation split (`SEED = 42`).*

| Model | Text representation | Train AUROC | Val. AUROC | Train time |
|---|---|---:|---:|---:|
| **Model A (Baseline): Logistic Regression** | Word $(1,2)$ TF-IDF (50k features, $C=1.0$) | 0.9932 | 0.9367 | 6.2s |
| **Model A2: Complement Naive Bayes** | Word $(1,2)$ TF-IDF (50k features, $\alpha=0.5$) | 0.9790 | 0.8966 | 0.02s |
| **Model B: Multi-View L2 Logistic Regression** | Word $(1,3)$ + Char `char_wb` $(3,5)$ TF-IDF + 72 Pure/Bimodal Stylo ($C=5.0$) | 1.0000 | 0.9743 | 2.2s |
| **Model C: Multi-View Calibrated LinearSVC** | Word $(1,3)$ + Char `char_wb` $(3,5)$ TF-IDF + 72 Pure/Bimodal Stylo ($C=1.0$) | 1.0000 | 0.9766 | 8.2s |
| **Model D: Formatting-Free LightGBM** | 72 Pure/Bimodal Stylo + 48 Multi-View TF-IDF SVD (`leaves=31, min_child=50`) | 1.0000 | 0.9822 | 0.6s |
| **Model E (Selected): Hybrid Ensemble** | $0.20\text{ Model B} + 0.20\text{ Model C} + 0.60\text{ Model D}$ | 1.0000 | **0.9837** | 11.0s |

---

### **Q1.3 [4 points]** — For at least two approaches, which hyperparameters and candidate values did you try? How did you select the final settings without using Kaggle test labels?

| Model | Parameters / values searched | Chosen setting |
|---|---|---|
| **Model B (Multi-View LogReg)** | $C \in \{0.5, 1.0, 2.0, 3.0, 5.0, 8.0\}$; stylometric L2 scale $\gamma \in \{0.10, 0.15, 0.25\}$ | $C = 5.0, \gamma = 0.15$ (5-Fold Inner CV: `0.9694`) |
| **Model C (Multi-View LinearSVC)** | Margin $C \in \{0.2, 0.5, 1.0, 1.5, 2.0\}$; 5-fold sigmoid Platt calibration | $C = 1.0$ (5-Fold Inner CV: `0.9727`) |
| **Model D (Formatting-Free LightGBM)** | `num_leaves` $\in \{15, 31, 63\}$, `min_child_samples` $\in \{30, 50\}$, `lr` $\in \{0.04, 0.05\}$ | `leaves=31, min_child=50, lr=0.05` (Inner CV: `0.9764`) |

**Selection Protocol**: All hyperparameter tuning was performed using **5-fold Stratified Cross-Validation strictly inside the 8,000-row training partition (`SEED = 42`)**, leaving the 2,000-row validation set untouched until final evaluation and never using Kaggle test labels. On the 150,072-dimensional sublinear TF-IDF + stylometric matrix, moderate regularization ($C=5.0$ for LogReg, $C=1.0$ for LinearSVC) maximized inner CV AUROC. For LightGBM, `num_leaves=31` and `min_child_samples=50` with feature subsampling (`colsample_bytree=0.7`) prevented overfitting to extreme lexical ratio tails.

---

### **Q1.4 [3 points]** — Which approach did you choose for your final submission? Explain using validation performance, computational cost and/or stability. If you later changed the model, explain why.

We chose **Model E (Hybrid Linear + GBDT Ensemble)** for three reasons:
1. **Validation Performance & Complementarity**: Model E achieved the highest validation AUROC (**`0.9837`** vs. `0.9367` for the baseline) and best probability calibration (Brier score **`0.0519`** vs. `0.1235`). Combining sparse linear classifiers (which capture fine-grained lexical and morphological n-grams across 150,000 features) with LightGBM (which captures non-linear interactions between lexical diversity `log_ttr`, repetition `yules_k`/`rep_3gram`, and sentence burstiness `cv_sent_len`) outperformed every individual model family.
2. **Stability Under Formatting & Domain Shift**: A diagnostic LightGBM trained *with* raw `\n\n` counts collapsed from `0.9985` to **`0.8109`** AUROC when line breaks were stripped, whereas our formatting-free Model E maintained **`0.9804`** AUROC.
3. **Efficiency & Subsequent Refinement**: Model E trains in **11.0s** on CPU. In Section 7, we further refined Model E by replacing unsupervised SVD with 5-fold Out-of-Fold (OOF) specialist logits and adding 24 Genre-Conditioned Stylometric Z-Scores, lifting validation AUROC from `0.9837` to **`0.9856`**.

---

## 3. Section 4: Effect of Training Data Size (10 points)

### **Q2.1 [4 points]** — Choose one model configuration. Train it with 500, 1,000, 2,000, 5,000 and 8,000 examples sampled from your 8,000-row training partition. Keep the same validation set and hyperparameters.

*Configuration evaluated: Multi-View Hybrid Classifier ($0.50\text{ Multi-View LogReg } [C=5.0] + 0.50\text{ Pure Stylometric LightGBM } [\text{leaves}=31, \text{lr}=0.04]$) evaluated on the fixed 2,000-row validation set.*

| Training rows | Training AUROC | Validation AUROC | Notes (optional) |
|---:|---:|---:|---|
| **500** | 0.9999 | 0.9554 | Active TF-IDF vocabulary: 36,112 n-grams (`1.07s` fit time) |
| **1,000** | 1.0000 | 0.9675 | Active TF-IDF vocabulary: 66,480 n-grams (`1.74s` fit time) |
| **2,000** | 1.0000 | 0.9746 | Active TF-IDF vocabulary: 118,915 n-grams (`3.12s` fit time) |
| **5,000** | 1.0000 | 0.9803 | Active TF-IDF vocabulary: 150,000 n-grams (`7.35s` fit time) |
| **8,000** | 1.0000 | 0.9834 | Full 8k training split; Test-Weighted Val AUROC = `0.9852` (`11.40s` fit time) |

---

### **Q2.2 [3 points]** — Insert a plot of training and validation AUROC versus training-set size.

*(Plot generated and saved in `plum/error_analysis_and_stacking.ipynb` Cell 10 — Panel 1: "Learning Curve: AUROC vs. Training Set Size (Q2.2)".)*

---

### **Q2.3 [3 points]** — What does the curve suggest about overfitting, underfitting, and the value of more labeled examples?

The learning curve reveals three clear regimes regarding generalization and sample complexity:
1. **High Variance / Vocabulary Sparsity at Small $N$ ($500\text{--}1,000$ rows)**: At $N=500$, Training AUROC is `0.9999` while Validation AUROC is `0.9554` (a `0.0445` generalization gap). Because the multi-view representation spans tens of thousands of word and character n-grams, a 500-row sample linearly separates the training set (low bias, no underfitting) but suffers from high variance and incomplete vocabulary coverage (only 36,112 active n-grams pass the minimum frequency threshold).
2. **Rapid Variance Reduction ($500 \to 2,000$ rows)**: Quadrupling the data to 2,000 rows cuts validation error by nearly half (`0.9554` $\to$ `0.9746`, $+0.0192$ AUROC) as common AI transition phrases and morphological suffixes populate the TF-IDF vocabulary and stabilize LightGBM leaf estimates.
3. **Continued Marginal Value of Additional Labels ($2,000 \to 8,000$ rows)**: Validation AUROC continues to rise monotonically from `0.9746` at $N=2,000$ to `0.9803` at $N=5,000$ and **`0.9834`** at $N=8,000$ without plateauing completely. This confirms that additional labeled examples remain valuable for estimating tail n-grams and rare writing domains, motivating our decision to retrain the final Kaggle submission on all 10,000 labeled rows.
