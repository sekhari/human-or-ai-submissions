# Expanded text models

Issue #10 runs the fixed ten-candidate grid in notebook 10. Five stratified folds
fit vocabulary, NB ratios, style scaling, and classifiers only on fit rows.
Seed is 42. Eligible unlabeled text supplies the issue's label-free margin
reference; test rows never fit features or score normalization.

| Candidate | Pooled training OOF AUROC | Validation AUROC | Final fit seconds |
|---|---:|---:|---:|
| word12_lr4 | 0.939897 | 0.941252 | 2.797 |
| word12_lr16 | 0.941195 | 0.943306 | 2.577 |
| word13_lr16 | 0.946027 | 0.946630 | 3.547 |
| word12_svm1 | 0.940804 | 0.943079 | 2.648 |
| charwb_lr16 | 0.931337 | 0.937988 | 9.051 |
| casechar_lr16 | 0.972219 | 0.978012 | 19.111 |
| casechar_svm1 | 0.972277 | 0.978329 | 15.867 |
| word13_casechar_lr16 | 0.973497 | 0.977144 | 23.209 |
| nbword13_lr4 | 0.947950 | 0.946824 | 4.707 |
| word12_style_lr4 | 0.944285 | 0.944379 | 6.781 |

Training OOF selects word1-3 plus case-preserving char2-5 logistic C16. The best
attempted greedy addition gains only .000346, below the fixed .0005 threshold.
The selected ensemble therefore has one member. Do not force extra members.
Case-character SVM leads validation, but validation did not choose the model.
The selected model's train AUROC is 1.0 and validation AUROC is .977144.
This gap and repeated validation reuse limit interpretation. No Kaggle score is
claimed. Sigmoid scores are bounded but are not claimed calibrated.

Search took 528.051 seconds; the complete fresh-kernel notebook took 755.204
seconds. All ten candidate fits, weaker arms, per-fold scores/times, reference
mean/stds, attempted greedy additions, and saved selection are retained.
The selection JSON predates validation. Safe record:
`results/10_expanded_text/20261110T201406444014Z.json`.

```bash
uv run --locked jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=3600 notebooks/10_expanded_text.ipynb
uv run --locked python tools/check_notebooks.py
uv run --locked python tools/check_text_handoff.py 10_expanded_text
```

The independent reload check matched all 2,000 held-out probabilities exactly,
without notebook globals or fitting. All private artifact hashes passed. The
notebook also checks NB ratio direction, empty style inputs, and greedy stopping
on synthetic rows. All ten pipelines were separately checked on synthetic text.

For #11, read `artifacts/10_expanded_text/latest.json`. Verify the safe record hash,
split checksum, and artifact hashes. It names all candidate pipelines/configs,
reference stats, and the selected standalone cloudpickle ensemble. Use
`class1_probabilities` after `cloudpickle.load`; inference keeps the stored
reference statistics. No test-batch normalization.

To reuse the factory without fitting or writing, set
`EXPANDED_TEXT_HELPERS_ONLY=1` before `%run ./10_expanded_text.ipynb`.
NB ratios, nine-style features, model construction, greedy selection, and margin
inference remain readable in the notebook. Raw text, models, and row predictions
stay ignored. Teammate review is requested; no automatic upload occurs.
