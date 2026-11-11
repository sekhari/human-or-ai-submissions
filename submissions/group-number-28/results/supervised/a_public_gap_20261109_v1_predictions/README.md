# Submission and distribution audit

This exploratory audit performs no model fitting or test-label inference.

Reproduce in a new output directory:

```sh
/opt/anaconda3/bin/python -m experiments.supervised.audit_public_predictions \
  --output results/supervised/a_public_gap_reproduction_predictions
```

## Verified findings

- All five candidate submissions have exactly 3,000 unique canonical test IDs,
  the required schema, and finite probabilities in [0, 1]. The associated
  validation files pass the canonical 2,000-row ID and label checks.
- There are no empty passages or duplicate passages after lowercase/whitespace
  normalization within the three input datasets. There are no such exact
  overlaps between labeled training data and test or unlabeled data. This
  check does not rule out near duplicates or shared sources.
- Original and expanded word-SVM submissions are numerically identical.
- Test predictions have Spearman correlation 0.99205 for quote-normalized
  WC120 versus WC120, 0.94068 for C60+richness versus WC120, and 0.88415 for
  word SVM versus WC120. Quote normalization changes 72 decisions at 0.5;
  richness changes 237. Threshold counts are descriptive; AUROC uses ranks.

## Observable shift

| Feature | Human training mean | AI training mean | Test mean | Train/test KS |
|---|---:|---:|---:|---:|
| Digit fraction | 0.012087 | 0.013237 | 0.007413 | 0.22007 |
| Newline fraction | 0.004435 | 0.009046 | 0.009911 | 0.14227 |
| MATTR50 | 0.791133 | 0.794553 | 0.780615 | 0.09083 |
| Normalized token entropy | 0.841800 | 0.843726 | 0.830922 | 0.09490 |

Test digit fraction, entropy and MATTR means are below both corresponding
training-class means. If these were exact population quantities, changing only
the AI/human mixture could not produce them. The differences from the nearer
class means are -14.15, -6.77 and -4.92 approximate independent-mean standard
errors, respectively. These are diagnostic measures, not adjusted hypothesis
tests; observations may share sources and the public subset is unknown.

Training-only feature quartile boundaries show that the lowest digit-rate bin
contains 468/2,000 validation examples (23.4%) and 1,403/3,000 test examples
(46.8%). On that validation subgroup, however, C60+richness still improves
AUROC from 0.9784 to 0.9826. Observable marginal shift therefore does not by
itself reproduce or explain the public richness regression.

## Interpretation and next checks

The submission contract is intact. Distribution differences are observable,
but their causal role in hidden-label errors is unverified. Compare future
models on validation stress conditions and credible source-held-out groups
when source metadata becomes available. A new random split alone is unlikely
to represent the hidden domains. Do not infer true AI prevalence from the
roughly 41% predicted-positive rate, or tune prediction thresholds to address
an AUROC deficit.

`metadata.json` records file checksums and dependencies. `feature_shift.json`
contains full feature summaries; `prediction_comparison.json` contains aligned
prediction comparisons; `validation_subgroups.json` records training-derived
boundaries, class counts and exploratory subgroup AUROCs.
