# Notes

## Approach
- Embedding model: EmbeddingGemma 2 (Ollama)
- Classifier: Gaussian Naive Bayes
- Each text is converted into a 768-dimensional embedding.
- Labels: Human (0), AI (1)

## Dataset
- `train_starter_2000.csv` (2,000 samples)
- Train: 1,600
- Validation: 400
- Split: 80/20, stratified, random_state=20261023

## Results
- Train AUROC: 0.9021
- Validation AUROC: 0.8557
- Validation Accuracy: 0.755
- Training time: ~0.01s (excluding embeddings)

## Hyperparameter Tuning
Tested `var_smoothing`:
- 1e-9
- 1e-8
- 1e-7
- 1e-6

All gave the same validation AUROC (0.855675), so I kept the default `1e-9`.