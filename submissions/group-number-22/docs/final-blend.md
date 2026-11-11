# Augmented style and text blend

Issue #17. Owner: Jalil-g (reassigned from Gtpatel02 on November 10, 2026).
Depends on #11, #14, #15, all merged.

## What we tried

#15 picked a style + text + NB blend, but its style arm drops to about .81
AUROC on collapsed whitespace. #14 trained the same HistGB style model with
collapsed copies (`plus_collapse`). Here we swap that arm into the #15 grid and
rerun the selection.

- Components, each standardized on eligible unlabeled rows:
  - style: #13 HistGB, 147 features, `plus_collapse` copies at weight .25.
    Score is the logit of clipped P(AI).
  - old soft text (#11/#15): word1-3 + case char2-5 LR C16 on all fit rows,
    case char2-5 SVM C1 on no-blank fit rows, +2 literal blank flag when no
    human fit row has one, then outer standardization.
  - NB transfer (#15 treatment, unchanged). The teacher is still the original
    #13 HistGB, so the style arm is the only change.
- Grid: the same six #15 weight vectors over (style, old soft, NB). Final score
  is sigmoid of the weighted standardized scores. This is score averaging, no
  test-batch normalization.
- Rule, committed before any fit (commit f7b0d15): on internal 6400/1600 (same
  rows as #14 and #15), keep candidates with remove-blank and collapse each
  >= pure augmented hist - .002, pick max original AUROC, ties go to pure hist.
  wrap80 is measured only. Selection is saved before validation.
- Check: the same grid with the original style arm must reproduce #15.

## Measured results

Internal 1600 AUROC, augmented style (selection grid):

| Candidate | original | remove_blank | collapse | wrap80 | eligible |
|---|---:|---:|---:|---:|---|
| hist | 0.997469 | 0.982400 | 0.944064 | 0.971556 | True |
| hist75_old25 | 0.998589 | 0.987512 | 0.954522 | 0.987059 | True |
| hist50_old50 (selected) | 0.998848 | 0.988433 | 0.953923 | 0.991481 | True |
| hist75_nb25 | 0.998392 | 0.990895 | 0.971175 | 0.989280 | True |
| hist50_old25_nb25 | 0.998755 | 0.991642 | 0.972616 | 0.993444 | True |
| old_soft | 0.994148 | 0.957148 | 0.908414 | 0.984825 | False |

Floors were remove-blank >= .980400 and collapse >= .942064. Only `old_soft`
failed. The original-style grid matches #15 exactly (max difference 0, same
pseudo acceptance and NB vocabulary), and pure augmented hist matches #14
`plus_collapse` exactly.

Frozen 2000, selected `hist50_old50`, all components refit on 8000:

| View | AUROC | #15 blend | accuracy at .5 |
|---|---:|---:|---:|
| original | 0.998426 | 0.998492 | 0.9805 |
| remove_blank | 0.991248 | 0.992831 | 0.8065 |
| collapse | 0.957370 | 0.931158 | 0.8355 |
| wrap80 | 0.992791 | 0.994044 | 0.9060 |

The #15 column comes from our own original-style refit, which matches the #15
record exactly. All six candidates on frozen 2000 for both style arms are in the
record as a descriptive table only. They did not choose anything.

Notebook 08 diagnostics on the held-out model: validation AUROC .998426,
bootstrap 95% interval .997004 to .999464. Lowest subgroups are
`upper_per_letter` > .0446 (.994259) and no line breaks (.996223). Train AUROC
is 1.0, which is not held-out performance.

Calibration is poor. Frozen ECE10 .270, Brier .0867, log loss .3415. Scores sit
between .2 and .8, and the .5 cut gives 37 false negatives vs 2 false positives.
AUROC ignores this, but do not read the outputs as probabilities.

## Takeaway

- The swap fixes much of the collapse loss (.931 to .957 frozen) for a .00007
  original cost. Remove-blank and wrap80 get slightly worse.
- The rule picked `hist50_old50`. `hist50_old25_nb25` was more robust on
  internal collapse (.9726 vs .9539) but lost by .000093 on original. We keep
  the fixed pick. Switching now would be a post-hoc choice.
- Validation was reused many times before (#4, #8, #9, #13, #14, #15), so
  frozen numbers are optimistic. None of this is a Kaggle score.

## Runs

Run 1 (`20261110T231600343244Z`, commit f7b0d15, clean tree) had a placeholder
takeaway. Runtime 309.6 s: internal fits 99.6 s, 8000 refit 91.4 s,
confirmation 76.1 s, diagnostics 8.1 s.

Run 2 (`20261110T232156602149Z`, commit ee144c6, clean tree) has the measured
takeaway and is the main record. Runtime 297.4 s. Every metric matches run 1
exactly. The clean-process handoff check reproduces all 2000 validation
probabilities with maximum difference 0 and verifies all private hashes.

## Reproduce

```bash
uv run --locked jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=3600 notebooks/17_final_blend.ipynb
uv run --locked python tools/check_notebooks.py
uv run --locked python tools/check_text_handoff.py 17_final_blend
```

Needs ignored `data/raw` and `artifacts/01_data_setup`. No #15 artifacts are
needed; every component refits from raw files. The private pointer
`artifacts/17_final_blend/latest.json` names the 8000-fit blend, validation
probabilities, standardized component scores on frozen 2000 (all four views),
eligible unlabeled, and test for the #16 neural blend step, and
`refit_spec.json` for #18. The spec has the schema, settings, weights, rule
state, and reference stats. A full 10000 refit must recompute every reference
statistic. Models, row predictions, and error cases stay ignored. Nothing is
uploaded.
