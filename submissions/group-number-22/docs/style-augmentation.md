# Label-preserving formatting augmentation

Issue #14. Owner: Jalil-g (reassigned from Gtpatel02 on November 10, 2026).
Depends on #12 (PR #32) and #13.

## What we tried

#12 showed formatting changes hurt our models, and #13's HistGB leans on blank
lines. So we added whitespace-only copies of training rows to the #13 HistGB
style model (same 147 features, same settings) and checked if it holds up on
reformatted text.

- Presets: `original_only`; `plus_collapse`; `plus_newline_wrap` (single
  newline + wrap64 + wrap80). Originals weight 1, each copy weight 0.25. Every
  fit row gets every copy of its preset, changed or not.
- Views change whitespace only. We checked on all 10000 labeled texts that
  `text.split()` is identical under every view, so labels are kept.
- Collapse: `" ".join(text.split())`. Single newline: each run of blank lines
  becomes one `\n`. Wrap: split on blank lines, `textwrap.fill` each chunk
  (`break_long_words=False, break_on_hyphens=False`), join with a blank line.
  This wrap80 differs from #12's (#12 wrapped each line and kept line breaks).
- Rule, fixed and committed before any fit: on inner 6400/1600 (seed 42, same
  IDs as #12 split42), keep presets with original AUROC >= baseline - 0.005,
  pick the highest mean AUROC over original/collapse/wrap80. Tie to fewer
  copies. Selection is saved before frozen validation is touched.
- Confirm: refit all three presets on 8000, score frozen 2000 on every view.
  Held-out rows are never augmented for fitting; their views are scored only.

## Measured results

Inner 1600 AUROC (fit 6400):

| Preset | original | collapse | single_newline | wrap64 | wrap80 | selection mean |
|---|---:|---:|---:|---:|---:|---:|
| original_only | 0.997975 | 0.809845 | 0.982770 | 0.979655 | 0.951977 | 0.919932 |
| plus_collapse (selected) | 0.997469 | 0.944064 | 0.982400 | 0.986102 | 0.971556 | 0.971030 |
| plus_newline_wrap | 0.996288 | 0.862248 | 0.991402 | 0.990480 | 0.989258 | 0.949265 |

All three passed the original floor (0.992975).

Frozen 2000 validation AUROC (fit 8000):

| Preset | original | collapse | single_newline | wrap64 | wrap80 |
|---|---:|---:|---:|---:|---:|
| original_only | 0.998105 | 0.804365 | 0.983410 | 0.982761 | 0.961117 |
| plus_collapse (selected) | 0.997762 | 0.954828 | 0.985360 | 0.988185 | 0.970871 |
| plus_newline_wrap | 0.996430 | 0.865940 | 0.992322 | 0.993687 | 0.993298 |

`original_only` validation matches #13 HistGB exactly (0.998105), so the refit
path is the same model.

Timings (run 2): features 17.0 s inner, 21.9 s development (5 views each).
Fits on 6400: 4.1 / 7.0 / 12.9 s. Fits on 8000: 5.2 / 8.7 / 17.2 s
(original / collapse / newline+wrap). Whole run 97.8 s.

## Takeaway

- `plus_collapse` is the pick. It costs about 0.0005 original AUROC and cuts
  the collapse loss from 0.19 to about 0.05.
- No preset is robust to everything. Each mostly fixes the views it trained
  on. `plus_newline_wrap` keeps line-break losses under 0.008 but still loses
  0.13 on collapse. The selection views overlap the training views, so these
  gains are partly by design.
- A preset with all four copies was not tested. Trying it now would be a
  post-hoc pick on the same rows. Validation was reused before (#4, #8, #9,
  #13), so validation numbers are optimistic. None of this is a Kaggle score.

## Runs

Two fresh-kernel runs from clean trees. Run 1 (`20261110T221929949660Z`,
commit cf6cd31) had a placeholder takeaway. Run 2 (`20261110T222137757157Z`,
commit d9470b1) has the measured takeaway and is the main record. All 30 AUROCs
match exactly across runs. Data, split, lock, version, and artifact hashes are
in each record.

## Reproduce

```bash
uv run --locked jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=1800 notebooks/14_style_augmentation.ipynb
uv run --locked python tools/check_notebooks.py
```

Needs ignored `data/raw` and `artifacts/01_data_setup`. All six models stay in
ignored `artifacts/14_style_augmentation/`. The private pointer
`artifacts/14_style_augmentation/latest.json` names the 8000-row
`plus_collapse` model for #17. No row predictions are saved. Nothing is uploaded.
