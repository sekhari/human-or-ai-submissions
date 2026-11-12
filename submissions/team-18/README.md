# Kaggle Competition: Human or AI? - Team 18 

**Team Members:** Pallavi Khanna (pk633), Minghan Gao (mg2328), Peter Chang (pc783) 

## Project Overview 

This project was about developing a binary test classifier by experimenting with different machine learning models to identify whether a passage was AI-generated or written by a human, since it's starting to be difficult to differentiate between an LLM generated passage versus a human generated passage.

## Models 

**For validation experiments, we used a random seed of 42 and split the data into 8,000 training and 2,000 validation examples. The selected Kaggle models were retrained on all 10,000 labeled examples.**

### Pallavi's Classification Model Experiments

The models I experimented with were logistic regression, multinomial Naive Bayes, and linear SVM which used TF-IDF features for the text. To improve the performance of logistic regression and linear SVM, we tuned the regularization parameter C. Of these models, Logistic Regression score the highest validation AUROC score of 0.9246 after hyperparameter tuning, however when we made the first submission we received a public AUROC score of 0.81356 from Kaggle. 

| Model | Main settings | Train AUROC | Validation AUROC | Training time |
|---|---|---:|---:|---:|
| Logistic Regression (baseline) | TF-IDF, C=1, max_iter=1000 | 0.9744 | 0.9222 | 0.12 s |
| Multinomial Naive Bayes | TF-IDF, default settings | 0.9202 | 0.8654 | 0.01 s |
| Linear SVM (baseline) | TF-IDF, C=1, max_iter=5000 | 0.9997 | 0.9221 | 0.15 s |
| **Logistic Regression (tuned)** | TF-IDF, C=10, max_iter=1000 | 0.9995 | **0.9246** | 0.19 s |
| Linear SVM (tuned) | TF-IDF, C=0.1, max_iter=5000 | 0.9757 | 0.9236 | 0.08 s |


### Minghan’s CPU experiments

Minghan compared a decision tree, random forest, histogram gradient boosting, and a multilayer perceptron (MLP) against a separate logistic regression baseline using word and character TF-IDF. The four models used 128 SVD components and 10 writing-style features. The MLP also scaled its inputs. All feature transformations were fitted on the training partition.

| Model | Main settings | Train AUROC | Validation AUROC | Validation 95% CI | Fit and save |
| --- | --- | ---: | ---: | --- | ---: |
| Logistic regression (baseline) | Sparse word and character TF-IDF, max 500 iterations | 0.999239 | 0.964987 | 0.957674–0.972119 | 15.4 s |
| Decision tree | Maximum depth 12, minimum 10 samples per leaf | 0.989933 | 0.963850 | 0.954426–0.972596 | 21.5 s |
| Random forest | 350 trees, minimum 3 samples per leaf | 0.999997 | 0.992921 | 0.990047–0.995223 | 23.5 s |
| Histogram gradient boosting | 250 iterations, learning rate 0.07 | 1.000000 | **0.998082** | 0.996775–0.999039 | 21.0 s |
| MLP | Hidden layers of 256 and 64 units, alpha 0.01, max 45 iterations | 1.000000 | 0.983124 | 0.978277–0.987254 | 20.4 s |

Histogram gradient boosting had the highest validation AUROC in this comparison (**0.998082**). The boosting submission scored approximately **0.87 public AUROC** on Kaggle.

### Peter’s text representation and formatting experiments

Peter compared word-level and hashed character-level TF-IDF with logistic regression, then tested formatting adjustments and word–character prediction ensembles. The selected configurations combine 75% character 3–5-gram predictions with 25% word TF-IDF predictions, add 2 to the log-odds for passages containing a double newline, and subtract 0.5 for a double space only when no double newline is present.

| Kaggle submission | Word features | Public AUROC |
| --- | --- | ---: |
| `word25_char75_blank2_doublespace05_v3.csv` | Word 1–2 grams | 0.95687 |
| `experiment_word_trigrams.csv` | Word 1–3 grams | **0.95792** |

## How to Run

Run the notebooks in Jupyter and Minghan’s CPU experiments from the terminal. Both workflows use the same stratified split of 8,000 training and 2,000 validation examples with seed `42`.

### Notebook experiments

1. Open `Experiment_git.ipynb` or `logisticreg_experiment.ipynb` in Jupyter from `submissions/team-18/`.
2. Install `pandas`, `numpy`, `scipy`, `scikit-learn`, and `matplotlib` in the notebook’s Python environment.
3. Put the competition files `train.csv`, `test.csv`, and `unlabeled.csv` next to the notebooks.
4. Run all cells in order to train and evaluate the models.

### Minghan’s CPU experiments

Run these commands from `submissions/team-18/`. The experiments were run with Python 3.13.5 on a laptop CPU.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create a `data/` folder inside `submissions/team-18/` and place a copy of `train.csv` there. Keep its original row order to reproduce the validation split.

Run the logistic regression baseline and the four model comparisons:

```bash
python train.py --train data/train.csv --model linear --seed 42 --run-dir models/val-linear
python train.py --train data/train.csv --model tree --seed 42 --run-dir models/val-tree
python train.py --train data/train.csv --model forest --seed 42 --run-dir models/val-forest
python train.py --train data/train.csv --model boost --seed 42 --run-dir models/val-boost
python train.py --train data/train.csv --model mlp --seed 42 --run-dir models/val-mlp
```

Each run saves its configuration, AUROC, and runtime in `run.json` and its per-example validation scores in `validation_predictions.csv`. Use a fresh `--run-dir` for each rerun.

### Final Kaggle submission

Note: The original Kaggle submissions were generated using Python 3.9.6. Running final_submission.ipynb in that environment reproduced both submitted CSV files exactly, including all 3,000 IDs and prediction values. We also tested the notebook in a newer Python environment. Both submissions were reproduced with a maximum absolute prediction difference below 0.0001. Minor numerical differences may arise from differences in Python or library versions.

Open `final_submission.ipynb` in Jupyter from `submissions/team-18/`, with `train.csv` and `test.csv` in the same folder. Install the notebook dependencies listed above and run all cells in order. This notebook trains three logistic regression classifiers on all 10,000 labeled examples: a shared hashed character TF-IDF (3–5 grams) model, a word TF-IDF (1–2 grams) model, and a word TF-IDF (1–3 grams) model. It blends the character model with each word model, applies the same formatting adjustments, and writes the two Kaggle submissions:

- `word25_char75_blank2_doublespace05_v3.csv` — word-bigram ensemble, **0.95687 public AUROC**.
- `experiment_word_trigrams.csv` — word-trigram ensemble, **0.95792 public AUROC** (selected model).

`Experiment_git.ipynb` records Peter’s exploratory experiments, and the same submissions could be generated by running all cells in order, along with some additional experimental predictions that were not used as final submission; `final_submission.ipynb` is the cleaner reproduction entry point for the selected Kaggle submissions.

## AI Disclosure

For this assignment, we used AI tools like ChatGPT to help us understand concepts, as a coding aid, and to help
with LaTeX formatting. It helped explain equations and what some questions were asking, how to
approach problems, and developing and troubleshooting Python code.

