# Experiment results

These JSON files retain measured evidence for the report. Their paths remain stable so the training, analysis and report scripts can reuse them. Generated row-level CSVs remain local and are ignored by Git.

## Final submission

| File | Contents |
| --- | --- |
| [delivery_manifest.json](delivery_manifest.json) | CSV format/identity checks, validation AUROC and team-reported public result |
| [optimization/current_best.json](optimization/current_best.json) | Locked final candidate `medium_v4`, public score and scored CSV SHA256 |
| [optimization/medium_v4_delivery.json](optimization/medium_v4_delivery.json) | Outer blend comparison and selected weights |
| [optimization/final_analysis.json](optimization/final_analysis.json) | Final diagnostics, length subgroups, error IDs and training rank certificate |
| [optimization/medium_v4_analysis.json](optimization/medium_v4_analysis.json) | Candidate-specific copy of the final analysis |

## Required baseline experiments

| File | Contents |
| --- | --- |
| [audit.json](audit.json) | Fixed split, class counts, duplicate checks and input checksums |
| [experiments.json](experiments.json), [selection.json](selection.json) | Model/hyperparameter comparisons and baseline blend selection |
| [learning_curve.json](learning_curve.json) | Fixed model at 500, 1,000, 2,000, 5,000 and 8,000 examples |
| [unlabeled.json](unlabeled.json) | Controlled vocabulary/IDF experiment using supplied unlabeled passages |
| [analysis.json](analysis.json) | Reference baseline error and subgroup analysis |
| [robustness.json](robustness.json), [improvement.json](improvement.json) | Supplementary splits, format stress tests and augmentation follow-up |
| [baseline_delivery_manifest.json](baseline_delivery_manifest.json) | Historical baseline delivery checks |

## Optimization history

`optimization/` also contains finite search logs, grouped diagnostics and each intermediate candidate's delivery/analysis records. The [optimization notes](../docs/OPTIMIZATION.md) explain the progression from the rejected `robust_v1` to `language_v2`, `neural_v3` and `medium_v4`. `search_with_extra_partial.json` records an incomplete exploratory search; it is not a completed model result. Historical records remain available for the course's development-history requirement.

The current final result is **0.98921 public / 0.999822 validation AUROC**. Earlier public scores in historical records belong to earlier candidates. Public scores are team-reported; private results are unknown.
