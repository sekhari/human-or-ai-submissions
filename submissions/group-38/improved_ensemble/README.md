# Improved Human-or-AI classifier

This completed experiment uses only supplied competition labels. Final model: 75% style gradient boosting + 15% two-seed neural MLP + 10% word/character logistic regression. No pretrained model or external inference API is used by the delivered prediction model.

## Run
Use Python 3.14 (tested 3.14.7). From this folder:

```sh
python -m pip install -r requirements.txt
export COMPETITION_ZIP="/absolute/path/to/cornell-tech-cs-5785-combined-2026-ai-vs-human-text.zip"
python run.py --data "$COMPETITION_ZIP" --output results
```

CSV files in the ZIP should be at its root, matching the supplied Nov 6 download. Outputs default to `results/`; set MODEL_OUTPUT_DIR to change this. Upload only `results/deliverables/submission_improved.csv` to Kaggle. Do not commit competition data, model weight files or prediction CSVs to the course repo.

## Validation and selection
All outer comparisons use the same stratified 8000/2000 split (seed 42) as Constance's `split_data`. Feature scaling and model fitting exclude outer validation labels. Word/character TF-IDF is fit only on training texts during validation. Feature extraction uses 99 observable style statistics: text/word/sentence length, vocabulary repetition, punctuation, capitalization, function-word frequencies and contractions. IDs are used only for joining outputs.

The neural MLP has hidden layers (64,32), ReLU activations, L2 alpha=1, batch size 128, Adam learning rate .001, 75 epochs selected with an inner 10% stopping split. Two final seeds (42,43) are averaged. Hyperparameters and ensemble weights are selected on the outer validation set, so reported AUROC is a model-selection estimate, not an independent test estimate. The boosted trees use 250 maximum iterations, learning rate .05, 15 leaves, L2=1 and internal early stopping. Word/character logistic regression uses C=4.

Five predeclared ensemble candidates were compared; selected weights are 0.75/0.15/0.10. Final components are refit using the full supplied labeled dataset; boosting reserves its internal early-stopping fraction. No hidden test labels are used. `submission_neural_mlp.csv` is the standalone neural comparison, not the recommended first submission.

## Measured results
Original character SVM calibrated validation AUROC: 0.969563; user-reported public Kaggle AUROC: approximately 0.85. Improved fixed validation: neural MLP 0.994286; style boosting 0.996350; final ensemble 0.997763. The user reported an approximately 0.95 public Kaggle AUROC after upload (rounded, not independently checked); the private score is unknown. Large training-to-test distribution differences may still reduce performance.

A DistilBERT pilot loaded pretrained weights and began fine-tuning its upper two layers. It was interrupted because measured throughput was about 2 seconds/batch; it produced no completed validation comparison and contributes no predictions to the final model. This negative/unfinished pilot must not be reported as a completed transformer experiment.

## AI disclosure
Codex implemented and ran these experiments, checked data splits, compared results, and verified the output format. Team members should review the code and results, understand the feature and model choices, and state their actual participation honestly.

## Historical result record
See [RESULTS.md](RESULTS.md) and [experiment_results.json](experiment_results.json) for the measured validation results, subgroup analysis, submitted-file checksum and user-reported public Kaggle result. These records contain no raw competition passages or hidden labels. The selected model is a gradient-boosting/MLP/logistic-regression ensemble, not DistilBERT.
