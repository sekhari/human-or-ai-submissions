# Seed, split, and formatting stress

Issue #12. Owner: Jalil-g. Depends on #11 (merged in PR #31).

## What we tried

We took the #11 primary soft ensemble (`rule_all_wordchar16` +
`rule_no_blank_case_svm1`, offset 2) and refit it fresh under five conditions.
All work stays inside original8000. Frozen2000 is not scored here.

- Split sensitivity: stratified 6400/1600 inner splits at seeds 42, 7, 2026,
  solver seed 42. Notebook 02 `split_ids` with exact 800/800 eval quotas.
  No duplicate groups exist inside original8000, so plain stratified splits.
- Solver randomness: solver seeds 7 and 2026 on fixed inner split 42. The seed
  goes into classifier `random_state`. Every other setting is exact #10/#11.
- Formatting sensitivity: on each fixed fit, score the 1600 eval rows as
  original, blank lines removed, whitespace collapsed, and each line wrapped at
  80. Stress touches eval text only. Fit rows and the eligible unlabeled margin
  reference stay original, and reference stats stay frozen.
- Each stress is scored with rule off (offset 0), soft (2), and hard (1000).

Ordered fit/eval ID digests per split seed are in the run record
(`config.inner_split_digests`). Eval sets overlap by 306 to 324 rows across seeds.

## Measured results

Original-text AUROC on 1600 inner eval rows:

| Condition | Rule off | Soft | Hard |
|---|---:|---:|---:|
| split42_solver42 | 0.961927 | 0.994148 | 0.994148 |
| split7_solver42 | 0.960228 | 0.994044 | 0.994048 |
| split2026_solver42 | 0.963822 | 0.993523 | 0.993523 |
| split42_solver7 | 0.961927 | 0.994148 | 0.994148 |
| split42_solver2026 | 0.961927 | 0.994148 | 0.994148 |

| Source | Rule off range | Soft range | Hard range |
|---|---:|---:|---:|
| split (3 seeds) | 0.003594 | 0.000625 | 0.000625 |
| solver (3 seeds) | 0 | 0 | 0 |

Solver max raw margin change vs seed 42: 0 for the LR arm, 3e-6 for the SVM arm.

Stress loss = original AUROC minus stressed AUROC, same condition and mode,
over five conditions:

| Stress | Mode | Min loss | Mean loss | Max loss |
|---|---|---:|---:|---:|
| collapse_whitespace | off | 0.051891 | 0.056249 | 0.068816 |
| collapse_whitespace | soft | 0.081592 | 0.088285 | 0.102631 |
| remove_blank_lines | off | 0.003459 | 0.004518 | 0.004797 |
| remove_blank_lines | soft | 0.033161 | 0.036555 | 0.038613 |
| wrap80 | off | 0.025450 | 0.026916 | 0.032273 |
| wrap80 | soft | 0.005800 | 0.006102 | 0.006775 |

Hard losses match soft within 2e-5. Max raw margin change under stress:
`rule_all_wordchar16` 3.90 (collapse), 4.51 (remove blank), 4.15 (wrap80);
`rule_no_blank_case_svm1` 1.16, 0.56, 1.16. Removing blank lines or collapsing
whitespace turns every rule flag off (0 active flags), so the rule gain vanishes.
Wrap80 keeps blank lines, so the rule still fires on 497 to 521 rows and covers
part of the loss the base model takes from new line breaks.

Rule state: active in all five fits (zero human literal-blank rows in each
6400 fit set). SVM specialist fit rows: 4271 to 4295.

Runtime: about 23 s to fit both arms and 60 s to score four stresses per
condition. Five conditions took about 420 s. Full timings are in the record.

## Takeaway

- Solver randomness is not a concern for this model. Split noise is small
  (soft range .0006). Formatting is the big risk.
- Most of the soft gain over rule off comes from the blank-line flag. If test
  text loses its blank lines, soft drops to roughly rule-off level or below.
- Collapsing whitespace also hurts the base text model (.05 to .07), so the
  learned features lean on line structure too.

These are inner local splits, not Kaggle results. Stress tests do not reveal
Kaggle subgroup labels and do not prove why hidden errors happen. They only
show what changes when we change formatting on our own labeled rows.

## Runs

Two fresh-kernel runs, both clean trees. Run 1 (`20261110T220448554156Z`, commit
1088bb5) had a placeholder takeaway. Run 2 (`20261110T221222713672Z`, commit
68b1ddf) has the measured takeaway in the notebook and is the main record. All
60 AUROCs match exactly across the two runs. Run 2 total runtime: 424.2 s.
Data, split, lock, and version hashes are in each record.

## Reproduce

```bash
uv run --locked jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=3600 notebooks/12_seed_stress.ipynb
uv run --locked python tools/check_notebooks.py
```

Needs ignored `data/raw` and `artifacts/01_data_setup` (duplicate components).
Raw eval margins per condition stay in ignored `artifacts/12_seed_stress/`.
No model is saved and nothing is uploaded.
