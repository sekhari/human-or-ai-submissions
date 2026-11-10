# Model B — Character TF-IDF + Linear SVM (Mia)

Status: implementation prepared; competition experiments have NOT been run. No scores or winning parameters are claimed.

## Integration
Place `character_svm.py`, this README and the notebook in `submissions/group-38/` (the notebook belongs in `notebooks/`). Keep the existing `data_utils.py` beside the script. Use the team's `submission/group-38` branch and preserve the other members' work.

Install the team's requirements plus joblib explicitly:

```sh
python -m pip install numpy pandas scikit-learn joblib matplotlib jupyter
```

Run from `submissions/group-38/`:

```sh
python character_svm.py --data data/cornell-tech-cs-5785-combined-2026-ai-vs-human-text.zip
# Optional, more expensive: evaluate calibrated probabilities and generate test predictions.
python character_svm.py --data data/cornell-tech-cs-5785-combined-2026-ai-vs-human-text.zip --probabilities
```

Use Python 3.10+ and scikit-learn 1.2+ (`CalibratedClassifierCV(estimator=...)`). The actual environment versions are saved with results. Never commit the competition data, fitted model files, or credentials.

## Experimental protocol
- Import the existing `load_data` and `split_data` functions: same row order and stratified 8,000/2,000 partition, seed 42 as Model A. Save IDs to verify identical partitions across teammates.
- TF-IDF: `analyzer='char'`, lowercase, character ranges (3,5), (3,6), (4,6); `min_df=2`, `max_features=100000`, `sublinear_tf=True`, L2 normalization (default), float32 storage. Character features include punctuation and cross-word sequences; no stemming or stop-word removal.
- LinearSVC: C in [0.1, 1, 10, 100], seed 42, max_iter=10000. Twelve predeclared configurations. Vocabulary and IDF are learned only from training text. Reusing training feature matrices across C values avoids redundant computation.
- Choose the converged configuration with highest fixed-validation AUROC; deterministic ties prefer smaller C, then shorter character ranges. This validation score is used for model selection and may be optimistic after tuning; it is not an independent test estimate.
- AUROC uses `decision_function` margins, which are valid ranking scores but NOT probabilities. The default decision threshold is margin 0.
- Optional probability mode uses three-fold sigmoid calibration of the full pipeline inside the 8,000-row training partition. Each fold learns its own vocabulary/IDF, without validation-label leakage. Record calibrated validation AUROC separately because calibration/ensembling may change ranking. Then refit that calibrated estimator on all 10,000 rows to generate test probabilities. Do not use a naive sigmoid of raw margins as a claimed calibrated probability.
- Runtime includes vocabulary fitting plus SVM fitting, excluding validation transformation and scoring; compare hardware/timing definitions with teammates.

## Outputs (local)
`results/character_svm/` contains:
- `tuning_results.csv`: training/validation AUROC, feature count, timing and convergence for all settings.
- `summary.json`: selected configuration, scores, seed and package versions.
- `train_ids.csv`, `validation_ids.csv`: exact split membership.
- `validation_predictions.csv`: IDs, true labels, margins, predictions; calibrated probabilities if requested.
- Local fitted models for reuse (do not commit).
- `submission_character_svm.csv` if probability mode is enabled; 3,000 IDs and AI probabilities. This script does not upload to Kaggle.

## Report contributions
After running, use the actual output for Q1.1 (representation), Q1.2 (model comparison row), Q1.3 (search space and selected setting), and the team's Q1.4 discussion. The group has not yet selected its final model. No learning-curve, unlabeled-data or error-analysis section is assigned to Mia by this implementation.

## AI assistance disclosure draft
Codex helped implement the character TF-IDF/linear SVM experiment, document the validation and calibration protocol, and check the code. The team must run the real experiments, inspect the outputs, verify the split, review modeling choices and understand the final solution before submission. Update this statement to describe what actually occurred.
