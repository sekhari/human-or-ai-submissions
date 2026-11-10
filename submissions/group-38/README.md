# Group 38 — Human or AI?

Cornell Tech CS 5785 Applied Machine Learning Kaggle Competition (Fall 2026).

## Team Members
- Constance Chen [sc3569]
- Mia Qianyu He [mqh4]
- Michelle Wang [sw2482]

## Project Overview

This project develops and evaluates machine learning models to distinguish human-written text from AI-generated text.

The competition provides 10,000 labeled training passages, 3,000 unlabeled passages, and 3,000 test passages. The task is to predict the probability that each test passage was AI-generated.

Models are evaluated using Area Under the Receiver Operating Characteristic Curve (AUROC).

Our experiments explore traditional text classification, neural networks, handcrafted writing-style features, semi-supervised learning, pretrained transformer embeddings, and ensemble methods.

## Setup and Dependencies

The project uses Python, Jupyter Notebook, scikit-learn, PyTorch, and Sentence Transformers.

The hybrid MLP and transformer experiments were developed using Python 3.13.9. The improved ensemble was originally tested using Python 3.14.7 and has separate pinned dependencies.

Install the project dependencies:

```bash
cd submissions/group-38
python -m pip install -r requirements.txt
```

The transformer notebook additionally requires `sentence-transformers`.

The improved ensemble has its own environment requirements:

```bash
cd improved_ensemble
python -m pip install -r requirements.txt
```

We recommend using a separate virtual environment for the improved ensemble to avoid dependency conflicts.

### Dataset

Download the competition dataset from Kaggle and place the original ZIP file at:

```text
submissions/group-38/data/
  cornell-tech-cs-5785-combined-2026-ai-vs-human-text.zip
```

Most notebooks load the original ZIP using the shared `data_utils.py` helper.

The improved ensemble pipeline expects CSV files at the root of the ZIP archive. If the downloaded ZIP contains a nested directory, create a local flattened copy before running that pipeline.

Competition datasets, generated predictions, and model artifacts are kept local and are not committed to GitHub.

## Models and Experiments

Our experiments include:

| Experiment | Description |
|---|---|
| Data Exploration | Dataset inspection, text lengths, and class distributions |
| Logistic Regression | Word TF-IDF features and regularization experiments |
| Training Set Size | Learning curve using different labeled training sizes |
| Unlabeled Data | Confidence-based pseudo-labeling |
| Multinomial Naive Bayes | Traditional probabilistic text classification |
| Neural Network MLP | Neural classification using TF-IDF features |
| Hybrid MLP | Word TF-IDF, character TF-IDF, and 99 writing-style features |
| Improved Ensemble | Style gradient boosting, style MLP, and lexical Logistic Regression |
| Model Blending | Weighted probability averaging and generalization analysis |
| Transformer Embeddings | Frozen MiniLM embeddings with Logistic Regression |

### Hybrid MLP

The hybrid MLP combines:

- 20,000 word TF-IDF features
- 30,000 character TF-IDF features
- 99 handcrafted writing-style features

The network uses hidden layers of 256 and 64 neurons, ReLU activation, dropout, and the AdamW optimizer.

The best validation checkpoint was selected based on validation loss. The final Kaggle model was retrained on all 10,000 labeled examples.

### Improved Ensemble

The improved ensemble combines three components:

- 75% style-based gradient boosting
- 15% style-based neural MLP
- 10% word/character Logistic Regression

The standalone pipeline is located in `improved_ensemble/`, with execution instructions in its own README.

### Model Blending and Transformer Embeddings

We combined the Hybrid MLP and improved ensemble using weighted probability averaging.

We also evaluated `sentence-transformers/all-MiniLM-L6-v2`, a publicly available pretrained encoder licensed under Apache 2.0. Its frozen embeddings were used to train a Logistic Regression classifier on the competition's labeled data.

Although MiniLM performed relatively poorly by itself, it provided complementary predictions that improved ensemble validation and public leaderboard performance.

No external labeled datasets were used.

## Training and Evaluation

We use a stratified 80/20 split of the 10,000 labeled examples:

- Training: 8,000 examples
- Validation: 2,000 examples
- Random seed: 42

Data-dependent feature transformations are fitted on the training partition for validation experiments.

Final submission models are trained using all 10,000 labeled examples after validation and model selection.

### Selected Results

| Model | Validation AUROC | Public Kaggle AUROC |
|---|---:|---:|
| Hybrid MLP | 0.994118 | 0.96013 |
| Improved Ensemble | 0.997794 | 0.95782 |
| 25% Hybrid / 75% Ensemble | 0.998052 | 0.96436 |
| Frozen MiniLM + Logistic Regression | 0.782262 | — |
| Three-Model Blend | 0.998244 | 0.96530 |
| **Selected Final BERT/CNN/Tree Ensemble** | **0.999393** | **~0.97 (user reported)** |

The three-model blend uses 22.5% Hybrid MLP, 67.5% improved ensemble, and 10% MiniLM.

Validation results are exploratory because the same fixed validation set was used for multiple model comparisons and ensemble selections. The public leaderboard is also not a reliable substitute for the private leaderboard.

## Generalization and Error Analysis

We investigated the gap between our near-perfect validation AUROC and lower public Kaggle performance.

Adversarial validation achieved an AUROC of 0.7448 when distinguishing training passages from test passages, suggesting detectable distribution differences.

Character TF-IDF similarity analysis also showed that test passages were generally less similar to training passages than validation passages were.

Our ensemble performed worse on validation examples that were less similar to the training data, suggesting that a random validation split may overestimate performance on unfamiliar writing styles.

We also analyzed errors shared by the Hybrid MLP and improved ensemble, and evaluated whether individual ensemble components contributed complementary information.

These analyses are documented in `07_model_blending.ipynb`.

## Kaggle Submission

Prediction files contain exactly two columns:

- `id`: Test passage identifier
- `label`: Predicted probability of AI-generated authorship, between 0 and 1

The team selected the BERT/CNN/tree/previous-model ensemble in [`final_ensemble/`](final_ensemble/) on November 9. The user confirmed the final recommended CSV including BERT and reported public AUROC of approximately **0.97**. Exact leaderboard precision and private AUROC are unknown. The previous MiniLM three-model blend scored 0.96530.

The selected ensemble uses 28% improved ensemble, 28% character CNN, 24% enriched-feature trees and 20% fine-tuned compact BERT. See the final model README for pinned dependencies, reproduction instructions, component weights, result records, pretrained-resource provenance and the original submission SHA-256.

## Reproducibility

Shared preprocessing and feature-extraction utilities are provided in `data_utils.py` and `style_feature_utils.py`.

To reproduce the experiments:

1. Install the relevant dependencies and download the original competition dataset.
2. Run the baseline and neural-network notebooks to reproduce individual model evaluations.
3. Run `06_hybrid_mlp.ipynb` to generate Hybrid MLP validation and test predictions.
4. Run the `improved_ensemble/` pipeline to generate ensemble validation predictions and test predictions.
5. Run `07_model_blending.ipynb` to evaluate probability blending and generalization.
6. Run `08_transformer_embeddings.ipynb` to reproduce the historical MiniLM three-model blend.
7. For the selected final ensemble, follow [`final_ensemble/README.md`](final_ensemble/README.md). Its runner reuses the sibling `improved_ensemble/` source and trains the CNN, enriched-feature trees and compact BERT locally.

Notebooks 07 and 08 require prediction CSVs generated by earlier experiments. The blending workflow also uses the original Kaggle-scored improved-ensemble submission, stored locally in `data/submission_improved.csv`. Regenerating Mia's ensemble in a different software environment may yield slightly different predictions.

Generated CSVs, competition data, and cached artifacts are excluded from version control.

## AI Assistance

Generative AI tools, including ChatGPT and Codex, were used during development for code assistance, debugging, experiment planning, implementation support, explanations, and documentation.

AI assistance included work on the improved ensemble pipeline, hybrid neural network experiments, model blending, transformer embedding experiments, and implementation, execution and packaging of the final BERT/CNN/tree ensemble.

The team remains responsible for understanding, executing, reviewing, and reporting the experiments and results. Only completed experiments and observed results are reported.

## Notes

The project includes both successful and unsuccessful experiments to document model selection and provide evidence of iterative development.

Final competition performance is determined by the private Kaggle leaderboard, which may differ from the public scores reported here.