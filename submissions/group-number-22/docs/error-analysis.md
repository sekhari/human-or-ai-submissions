# Subgroup and error analysis

Issue #8 checks where the #5 handoff (word logistic C4) is weak on the fixed
2,000 validation rows. Nothing is fitted. Bin edges come from training quartiles
and apply unchanged to validation. Validation bins get class-stratified bootstrap
95% intervals, 1,000 draws, seed 42. Errors use threshold .5. FP means human
text predicted AI.

| Partition | AUROC | Accuracy at .5 | 95% interval |
|---|---:|---:|---|
| Train (in-sample) | 0.999925 | 0.9925 | not computed |
| Validation | 0.941252 | 0.8780 | 0.930520 to 0.950925 |

Train minus validation AUROC is 0.058673. Training AUROC is at least 0.99983 in
every bin, so the gap is generalization, not underfitting.

| Measure | Weakest validation bin | AUROC (95% CI) | Strongest bin | AUROC (95% CI) |
|---|---|---|---|---|
| Uppercase per letter | above 0.0446 | 0.8653 (0.8349 to 0.8946) | 0.0181 or less | 0.9801 (0.9695 to 0.9893) |
| Words | 160 or fewer | 0.9120 (0.8874 to 0.9356) | 206 to 256 | 0.9660 (0.9492 to 0.9797) |
| Line breaks | 1 to 4 | 0.9089 (0.8724 to 0.9405) | 0 | 0.9544 (0.9383 to 0.9691) |
| Punctuation per char | 0.021 to 0.0274 | 0.9220 (0.8948 to 0.9451) | 0.021 or less | 0.9663 (0.9507 to 0.9799) |

All 16 quartile bins have both classes, and no edges tied on this data. For words,
uppercase, and punctuation, the weakest and strongest bins have non-overlapping
intervals; for line breaks they overlap. These bins were picked after seeing the
scores, so separation is weaker evidence than it looks. At .5 there are 61 FP and 183
FN. The notebook lists the five most confident of each by ID, probability, and
measurements, with short hypotheses. Three of five confident FN contain a blank
line and none of five FP do. That is ten rows, a lead for #11, not a result.
Genre and source guesses in the notebook are speculation. Hidden test domains
and generators are unknown.

Analysis took 8.0 seconds; the notebook took about 11 seconds. Record:
`results/08_error_analysis/20261110T191604750902Z.json`.

## Reproduce

The handoff model and predictions are ignored, so regenerate them first. Run
notebooks 03, 04, and 05 fresh, then 08:

```bash
uv run --locked jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=900 notebooks/08_error_analysis.ipynb
uv run --locked python tools/check_notebooks.py
```

This run regenerated 03 to 05 on Linux aarch64. Their recorded metrics matched the
committed macOS records except for float noise: word C4 validation AUROC was
identical (0.941252), one word training fold moved by 5.6e-7, and the losing
character arm moved by 4e-6. No selection changed.

## Reuse for #17

Set `ERROR_ANALYSIS_HELPERS_ONLY=1` in the environment, then
`%run ./08_error_analysis.ipynb`. That defines `measure`, `quartile_edges`,
`subgroup_table`, `confident_errors`, and `stratified_auc_interval` without
running this analysis. Derive edges from the new model's training rows.
No upload occurs. Teammate review is requested in the PR.
