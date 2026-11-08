# Kenny's Kaggle-tested neural experiments

Only candidates with public Kaggle outcomes reported by the user are listed
here. Shared source code, training split definitions, validation results, and
provenance needed by these experiments accompany the submissions.

| Candidate | Validation AUROC | Public Kaggle AUROC |
| --- | ---: | ---: |
| [DeBERTa text/style hybrid + boosting](../../submissions/deberta_hybrid_blend_maximum_8000.csv) | 0.999485 | 0.98267 |
| [RoBERTa task-adaptive-pretraining blend](../../submissions/roberta_tapt_blend_8000.csv) | 0.999837 | 0.98256 |
| [Supervised RoBERTa + GPT-2 curvature/bagging](../../submissions/curvature_roberta_single_10000.csv) | **0.999911** | **0.98660** |

Public scores were supplied in chat, not independently retrieved from Kaggle;
private scores are unknown. The last candidate improved the earlier team best
of 0.98544. The user subsequently reported a teammate's `improve_all_blend`
score of approximately 0.991; its implementation was not available at that
time. The earlier MLP submission and results are already in the shared history.

## Methods and evidence

- [DeBERTa hybrid report](../deberta_hybrid_blend/README.md) and [public outcome](../deberta_hybrid_blend/kaggle_result.json).
- [RoBERTa report](../roberta_experiments/README.md) and [public outcome](../roberta_experiments/kaggle_result.json).
- [Supervised RoBERTa/curvature validation](../curvature_roberta/single_validation.json), [public outcome](../curvature_roberta/kaggle_result.json), and [submission verification](../curvature_roberta/first_submission_verification.json).

The final curvature blend uses 35% supervised RoBERTa and 65% curvature/bagging
in log-odds space. Epoch and weight were selected on training-only holdouts.
Its validation score comes from a separate 8,000-row fit; its submission refits
the neural component on all 10,000 labels and uses the teammate's existing
10,000-row curvature predictions. The two older submissions use 8,000-row fits.
Validation was repeatedly reused across the project and is not a fresh test.

## Code and storage

Entry points are `experiments/deberta_hybrid_blend.py`,
`experiments/roberta_experiments.py`, and `experiments/curvature_roberta.py`.
The original pipelines share data preparation and model helpers; those helper
files are included even when they also support other experiment variants.
Controls and training artifacts within these runs explain the selected models.

Pretrained weights, learned checkpoints, and feature caches stay in ignored
`.cache/` directories. They are not included in Git. The original pipelines
require these locally generated dependencies; sharing the code does not copy
trained weights to another computer. Package additions are recorded in the
experiment-specific requirements files. Exact data/model/source provenance is
retained in the adjacent JSON records.

The three-seed ensemble candidate is omitted from this collection because no
public Kaggle outcome has been confirmed for it. Independent unsuccessful
follow-up experiment results are also omitted. Original shared repository
files are left unchanged.
