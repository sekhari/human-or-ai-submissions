# Model Hyperparameter Metadata & Configuration Files

This directory contains external YAML configuration files defining model hyperparameters, vectorization settings, cross-validation parameters, and ensemble weights for the Jupyter notebooks under `submissions/plum/` (`submissions/team-35/plum/`).

Editing any YAML file here immediately updates the parameters used by the corresponding notebook upon running its cells, eliminating the need to modify code within `.ipynb` files directly.

## Configuration Files Mapping

| YAML File | Associated Notebook | Description & Controlled Components |
|---|---|---|
| `model_selection.yaml` | `model_selection.ipynb` | Vectorizers (`vec_w12`, `vec_w13`, `vec_c35`), Stylometric scale, SVD components, ablation LogReg, 5-fold inner CV grids, Models A, A2, B, C, D0, D hyperparameters, and Model E hybrid blend weights. |
| `error_analysis_and_stacking.yaml` | `error_analysis_and_stacking.ipynb` | Multi-view N-gram vectorizers (85k/95k), Level-1 specialists (LogReg, Calibrated LinearSVC, ComplementNB, LightGBM), Level-2 meta-learner LogReg (`C`), semi-supervised learning thresholds (0.95/0.05), and learning curve sample sizes. |
| `leaderboard_boost.yaml` | `leaderboard_boost.ipynb` | 4-view vectorizers, OOD holdout domains, 5 orthogonal specialists (`lr_4v`, `svc_4v`, `mask_lr`, `lgb_pure`, `lgb_struct`), blend weights, and Stage 2 SSL refinement hyperparameters. |
| `model_selection_v3_high_score.yaml` | `model_selection_v3_high_score.ipynb` | Dual trigram LM smoothing `alpha`, 4-view vectorizers (95k), 9-specialist Stage 1 models (GBDT triad: LightGBM, CatBoost, XGBoost), Stage 1 fusion weights, Stage 2 transductive self-training thresholds/models, and final logit fusion weights. |
| `feature_analysis.yaml` | `feature_analysis.ipynb` | Adversarial validation LightGBM classifier settings, adversarial word TF-IDF vectorizer, and baseline Logistic Regression parameters. |
