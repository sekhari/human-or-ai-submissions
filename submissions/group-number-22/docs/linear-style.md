# Small style feature test

Issue #9 adds nine scaled style features to the #5 handoff (word TF-IDF, logistic
C4) and keeps everything else fixed: split, word settings, classifier, seed 42,
and five training folds. The hypothesis and adoption rule were committed before
any run. Features: log word count, line breaks per word, punctuation per char,
uppercase per letter, digits per char, log words per sentence, type-token ratio,
non-ASCII per char, list-line rate. The scaler is fit inside the pipeline.

| Arm | Train AUROC | Validation AUROC | CV mean (sd) | Fit s | CV s |
|---|---:|---:|---|---:|---:|
| Reference, word C4 | 0.999925 | 0.941252 | 0.939819 (0.002942) | 1.56 | 7.08 |
| Word C4 + style | 0.999443 | 0.944372 | 0.944227 (0.001467) | 3.58 | 15.04 |

The reference reproduces the #5 handoff validation AUROC exactly.

| Adoption condition | Measured | Passed |
|---|---|---|
| Validation gain at least .002 | +0.003120 | yes |
| Paired 95% CI entirely above 0 | -0.002509 to +0.009001 | no |
| Every CV fold improves | fold gains +0.001767, +0.007648, +0.005487, +0.007491, -0.000356 | no |

**Not adopted.** The bootstrap uses 2,000 paired class-stratified draws, seed 42;
86.7% of draws were above zero. It conditions on the fitted models and reused
validation rows.

Subgroups (bins from training quartiles, descriptive only): the top uppercase bin
went 0.8653 to 0.8955 (+0.0301), as predicted. The shortest bin went 0.9120 to
0.8867 (-0.0253), against the hypothesis. The lowest uppercase bin also dropped
(-0.0171). Guess, not tested: per-text rates are noisy on short passages.

A near-identical test in our earlier team repo was also inconclusive. The rule
here came from the issue, not from that result.

Whole run took 33.7 seconds. Record: `results/09_linear_style/20261110T193404498410Z.json`. The plan JSON was saved before any fit.

```bash
uv run --locked jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=900 notebooks/09_linear_style.ipynb
uv run --locked python tools/check_notebooks.py
```

Needs the #5 handoff regenerated locally first (notebooks 03 to 05). No models
are saved; #10 refits its own word plus style candidate. No upload occurs.
