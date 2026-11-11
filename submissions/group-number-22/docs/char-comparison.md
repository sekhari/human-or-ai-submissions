# Character comparison and error-analysis handoff

Issue #5 fits char_wb 3-5 TF-IDF with lowercase text, min_df 2, 100,000 features,
sublinear TF, L2 normalization, and float32. Logistic C values .25, 1, and 4 use
exactly the same three seed-42 training folds as #4. Every fold refits features.
Original passage text remains unchanged for later representations.

Training CV selects C4. Character CV mean is .928301, versus .936100 for word C4.
The precommitted handoff policy keeps word unless character improves CV by more
than .002. Selection is saved before character validation.

| Model | Train AUROC | Validation AUROC | Final fit seconds |
|---|---:|---:|---:|
| Word logistic C4 | .999925 | .941252 | 2.785 |
| Character logistic C4 | .998412 | .938334 | 8.594 |

Character minus word validation AUROC is -.002918, with paired bootstrap 95%
interval [-.009703, .003980]. This run gives no clear evidence of a character gain.
Keep the losing character arm. Bootstrap uses 2,000 class-stratified paired draws
at seed 42, conditional on fitted models and reused validation rows. It does not
include training variation or selection bias.

The character search took 65.164 seconds; the full notebook took 90.969 seconds.
The fresh-kernel run passed. Reload differences were zero, and standalone handoff
predictions matched the saved word probabilities. Safe record:
`results/05_char_comparison/20261110T184002363091Z.json`.

```bash
uv run --locked jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=900 notebooks/05_char_comparison.ipynb
uv run --locked python tools/check_notebooks.py
```

For notebook 08, read `artifacts/05_char_comparison/latest.json`. Check its split
checksum, safe record hash, and artifact hashes. Its `model`, `train_predictions`,
and `validation_predictions` fields name a standalone selected pipeline and
ID-aligned probabilities. Use notebook 02's `class1_probabilities` for inference
and `evaluate_predictions` for joins. These artifacts remain ignored and must be
regenerated locally after dependencies. No raw text, models, or row predictions
are committed. No upload occurs. Teammate review is requested in the PR.
