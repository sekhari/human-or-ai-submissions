# Group 36 integration history

## Existing Alma work — 2026-11-07

Alma's original incremental commits and all RF comparisons are preserved in Git history. Member-level details and original local references are now in `alma_rf/EXPERIMENT_HISTORY.md` and `alma_rf/LOCAL_COMMIT_HISTORY.txt`. Historical no-upload statements describe the state at the time of those experiments, not the current team state.

## Jianing integration and layout update — 2026-11-08

- Added Jianing Li's existing, reviewed notebook and recorded experiments; these are migrated results, not a claim that Git tracked the earlier local development.
- Retained normalized TF, full-text MiniLM, equal-weight fusion, all 30 optimization trials, negative results, ROC figures, length subgroups and training-only CV diagnostics.
- Initial best: TF + chunked MiniLM, C=1, validation AUROC 0.922313; earlier public score 0.84249.
- Optimized candidate: word 1–2 gram TF-IDF + case-sensitive character 3–5 gram TF-IDF + MiniLM with weight 0.25, overall L2 normalization, LinearSVC C=1. Validation AUROC 0.974103. Training-only CV mean 0.967231 versus 0.916529 for the previous model.
- The model was selected on validation, then checked using training-only CV. This is not a nested-CV estimate and does not prove cross-domain robustness.
- Added both probability-generation scripts. Calibration uses five internal folds, with fold-specific vocabulary/IDF, then full-data refitting. Calibrated validation AUROC matched 0.974103 before the 10,000-row optimized refit.
- Jianing reported public AUROC **0.90456** for the optimized CSV. Published file bytes match the generation receipt. The score is user-reported; no Kaggle API retrieval or private-score claim was made.
- Confirmed Alma and Jianing have identical train/validation ID membership despite different manifest formats.
- Organized member work into `alma_rf/` and `jianing_svm/`. Updated executable paths to use Jianing's requested name. Historic receipt filenames and prior score rows are preserved with explicit aliases.
- Fixed Alma's export notebook to use the tracked selected metadata under `results/`, removing its dependency on an untracked historical output directory. RF settings were not changed.

## November 10 — Team model selection

The team selected Abishek’s submission.csv (prob_average_final.csv) (public AUROC 0.922). His separate previous rank-average experiment combined four NB-weighted logistic models and one stylometric gradient boosting model. Jianing evaluated five training sizes from 500 through 8,000. See the team README for results and the switch to the weighted probability average.
