# Improved model results — November 6, 2026

Recommended upload: `submission_improved.csv`.

| Model | Fixed validation AUROC |
|---|---:|
| Previous character SVM (calibrated) | 0.969563 |
| Word/character logistic regression | 0.969059 |
| Neural MLP over style features (two seeds) | 0.994286 |
| Style gradient boosting | 0.996350 |
| 75% boosting + 15% MLP + 10% logistic regression | **0.997763** |

The final ensemble's AUROC by word count is 0.993675 (<150), 0.998958 (150–249), and 0.998001 (250+). All comparisons use the same 2,000-row stratified validation partition, seed 42. The validation set was used for model and ensemble selection; it does not provide an independent estimate of the new Kaggle score. No claim is made that the new hidden-test score is 0.997763.

Verified submission: exactly 3,000 unique test IDs, official sample ID order, columns id/label, all finite probabilities in [0,1].

DistilBERT pilot was stopped due to runtime; the delivered neural component is an MLP, not a transformer.

## Kaggle feedback

The team reported a public Kaggle AUROC of approximately **0.95** for the delivered submission, compared with approximately 0.85 for the original character SVM. These rounded scores were supplied by the user and have not been independently checked in the Kaggle UI. The private score is unknown. The submitted CSV SHA-256 is recorded in experiment_results.json.
