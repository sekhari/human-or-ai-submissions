# Rich style features with nonlinear models

Issue #13 builds 147 deterministic style features that ignore word identity, then
trains HistGB and ExtraTrees with the issue settings. The schema (names, order,
regexes, denominators, empty-text guards) and the selection rule were committed
before any run. Schema hash: `3126aec1687a`. Features learn nothing from labels,
so they are computed once per partition; only models fit, and only on fit rows.

| Candidate | Pooled OOF AUROC | Fold AUROC | Train AUROC | Validation AUROC | CV s | Fit s |
|---|---:|---|---:|---:|---:|---:|
| HistGB (selected) | 0.998065 | 0.997970, 0.998606, 0.997613 | 1.000000 | 0.998105 | 3.35 | 1.51 |
| ExtraTrees | 0.996777 | 0.996797, 0.997514, 0.995973 | 0.999996 | 0.996831 | 11.96 | 6.33 |

HistGB won on pooled OOF before validation was scored. For reference, the word C4
handoff scores 0.941252 on the same validation rows.

**Shortcut check (post-hoc, added after the first run).** The top ExtraTrees
importances were paragraph statistics and blank-line rate. Paragraphs split on blank
lines, so these mostly encode whether a blank line exists. Blank lines appear in
16 of 4,000 human and 2,629 of 4,000 AI training rows (1 of 1,000 and 676 of 1,000
in validation). On the 1,323 validation texts with no blank line:

| Model | AUROC, no blank line |
|---|---:|
| HistGB | 0.994692 |
| ExtraTrees | 0.991246 |
| Word C4 handoff | 0.950889 |

So style still beats words without the shortcut. Guess, not tested: the remaining
signal (line breaks, vocabulary spread) may also reflect how the dataset was
built. Hidden test domains and generators are unknown, and the course notes say
the test includes unfamiliar domains and modified AI text. #12 and #14 test
formatting robustness.

Validation was reused in #4, #8, and #9, so these scores are optimistic.
ExtraTrees with n_jobs=2 sums tree votes in thread order, so its probabilities
change by about 1e-16 between calls and its AUROC can move by about 2e-6. HistGB
is deterministic. Features took 3.8 s for 10,000 texts; the whole run took 28.3 s.
Record: `results/13_rich_style/20261110T195146556937Z.json`.

```bash
uv run --locked jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=1800 notebooks/13_rich_style.ipynb
uv run --locked python tools/check_notebooks.py
```

Needs the #5 handoff regenerated locally for the post-hoc comparison. Both models,
the schema, and validation probabilities stay under ignored artifacts. The private
pointer `artifacts/13_rich_style/latest.json` names the HistGB model and schema
for #14, #15, and #17. To reuse the features, set `RICH_STYLE_HELPERS_ONLY=1` and
`%run ./13_rich_style.ipynb`. No upload occurs.
