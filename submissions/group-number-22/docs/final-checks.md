# Final refit and handoff

Issue #18. Owner: Jalil-g. Depends on #6, #7, #8, #9, #12, #16, #17, all merged.

## What we tried

1. **Neural blend pick (#16 carry-over).** The plan was fixed in #16 before #17
   existed. Inputs are the saved 8000-fit scores: #17 classical blend score
   (`hist50_old50`) and #16 ELECTRA margin. Each is standardized on eligible
   unlabeled rows. Score = (1 - w) * classical + w * ELECTRA for w in 0, .1,
   .25, .5, .75, 1. Objective is mean AUROC over original, remove blank lines,
   collapse whitespace, wrap80. Guardrail: original >= w=0 original - .002.
   **Update (#37):** this pick used frozen 2000. #37 redid it on inner 1600
   training rows with 6400-fit models and got the same w = .25. See
   `docs/internal-neural-pick.md`.
2. **Replay.** Refit the classical arms on the same 8000 rows with the copied
   code. It must match the #17 scores before we trust the full refit.
3. **Full refit.** Style HistGB on all 10000 in original `train.csv` order.
   Old soft text LR/SVM and ELECTRA on all 10000 in manifest order (training
   then validation). Every reference statistic is recomputed on eligible
   unlabeled rows. One sklearn thread. ELECTRA on MPS, 3 epochs, #16 settings.
4. **Export.** `submissions/18_final_checks/<run>/submission.csv`, columns
   `id,label`, label = P(AI). The file is read back with text IDs. It must
   match sample IDs exactly, with no missing, extra, or duplicate IDs and only
   finite [0, 1] values. It must also match predictions recomputed by ID from
   shuffled test order. The checker was tested on one valid file and eight
   broken ones.
5. **Repeatability.** Reran notebooks 01-17 from fresh kernels in dependency
   order in a separate checkout of `d09956d`, then compared every metric with
   `tools/compare_reruns.py`.

**Planned vs actual.** The issue planned a classical candidate with hist
threads 2. #17 changed the pick to `hist50_old50` with one thread everywhere,
and the #16 plan then added ELECTRA. We used those choices.

**Caveat.** Only frozen 2000 has 8000-fit scores for both models, so w was
picked on frozen validation, as the #16 plan said. The frozen numbers for the
picked blend are selection numbers, not a clean confirmation. Frozen 2000 was
also reused in #4, #8, #9, #13, #14, #15, #17.

## Measured results

Neural blend grid, frozen 2000, 8000-fit scores:

| w | original | remove_blank | collapse | wrap80 | mean | eligible |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.998426 | 0.991248 | 0.957370 | 0.992791 | 0.984959 | yes |
| .1 | 0.998948 | 0.994612 | 0.977205 | 0.996595 | 0.991840 | yes |
| **.25 (picked)** | 0.999083 | 0.995365 | 0.987313 | 0.997861 | 0.994905 | yes |
| .5 | 0.998143 | 0.993931 | 0.989754 | 0.997216 | 0.994761 | yes |
| .75 | 0.994891 | 0.991588 | 0.989427 | 0.994188 | 0.992524 | no |
| 1 | 0.987481 | 0.987481 | 0.987481 | 0.987480 | 0.987481 | no |

The floor was .996426. w = .5 had better collapse but a lower mean.

Picked blend on frozen 2000: AUROC .999083, bootstrap 95% interval .997946 to
.999772. Accuracy at .5 is .9895 (17 false negatives, 4 false positives).
Lowest subgroup is texts with 160 words or fewer (.997124). Calibration is
still poor: ECE10 .2745, Brier .0865, log loss .3421 on original. AUROC does
not care, but do not read the labels as calibrated probabilities.

Replay of #17 on 8000 rows: max abs difference 4.4e-16 on the component
scores.

Full 10000 refit (main record `20261111T003634149593Z`, commit d3f7698, clean
tree): no held-out score. In-sample AUROC .99999912 is only a check that class
1 means AI. The old blank-line rule stays active (no human row has `\n\n`).
The no-blank specialist used 6699 rows. ELECTRA train loss went .3348, .0971,
.0478 over 3 epochs in run 1.

Submission (run 2): 3000 rows, label range .1771 to .8376, mean .4849, 45.6%
at or above .5. sha256 `2404467eff7eb02f369b173fa3ec427310087a0c863aac18c8d65d8900cdaf8a`. All eight broken fixtures were
rejected (wrong columns, missing row, extra row, duplicate ID, NaN, out of
range, shifted values, wrong class column).

## Repeatability

Notebook 18 ran twice from fresh kernels:

| Run | Commit | Runtime | Notes |
|---|---|---:|---|
| `20261110T234226916505Z` | f0c937c | 671.8 s | before the rerun summary existed |
| `20261111T003634149593Z` | d3f7698 | 582.6 s | main record, shows the rerun summary |

The neural grid, replay, classical reference stats, and in-sample AUROC are
identical across the two runs. The ELECTRA reference mean moved 2.16949 to
2.16935 on MPS. Test probabilities differ by up to 7.9e-4 between runs
(Spearman .9999998), so the submission bytes differ. Use the run 2 file.

Clean rerun of 01-17 (`results/18_final_checks/20261111T002621723580Z-rerun.json`):
all 17 completed, and every selected candidate and non-numeric field matched.
Metrics were identical for 01, 02, 03, 06, 07, 12, 14, 17. Largest gaps:

| Notebook | Max abs metric diff | Where |
|---|---:|---|
| 04, 05, 08 | <= 4e-6 | AUROC rounding |
| 09 | .0005 | bootstrap share; validation AUROC .944372 vs .944379 |
| 10 | .00056 | SVM margin reference std in folds |
| 11 | .00125 | SVM margin reference std in folds |
| 13 | .00044 | permutation importances |
| 15 | .0005 | one accuracy flip at .5; AUROC within 2e-6 |
| 16 | 0 | AUROC equal to 6 places; raw margins moved up to 4e-4 (MPS) |

## Kaggle upload log

Nothing was uploaded by any notebook or agent. Jalil uploads by hand and fills
this in. Five per day is shared across the whole team, so check the team
count first.

| UTC | File sha256 | Who | Receipt | Status | Public | Private |
|---|---|---|---|---|---|---|
| pending | `2404467eff7eb02f369b173fa3ec427310087a0c863aac18c8d65d8900cdaf8a` | Jalil | | | | pending |

Target public AUROC >= .98 is a goal, not a measured result.

## Member contributions

From merged PRs and Git history (non-merge commits on main before #18:
Jalil-g 20, Rahul Devasia 14, Gtpatel02 10).

- Jalil-g: #1 data audit, #2 split and helpers, #6 learning curve, #12 seed
  and stress checks, #14 augmentation, #16 local ELECTRA, #17 final classical
  blend, #18 final refit and handoff. #14 and #17 came from Gtpatel02 on
  November 10, 2026.
- rdevasia25: #3 word baseline, #4 word learners, #5 char features, #7
  self-training, #10 expanded text, #11 formatting rule (from Gtpatel02), #15
  normalized transfer.
- Gtpatel02: #8 error analysis, #9 linear style, #13 rich style.

## Disclosures and open items

- AI coding assistants (OpenAI Codex and Anthropic Claude Code) were used to
  write and run notebook code under member direction. Course disclosure text
  must say this in the final deliverables.
- ELECTRA uses the pretrained `google/electra-small-discriminator` checkpoint
  (Apache 2.0), fine-tuned locally. No remote inference service was used.
- Not confirmed yet: exact cutoff time and timezone, the course PR
  destination, the portal steps, and which Kaggle submission is marked final.
  Supplied deadline is November 11, 2026.
- The report template is still deferred and was not opened. Course delivery is
  not complete.

## Reproduce

```bash
uv sync --project envs/electra --locked
uv run --project envs/electra --locked jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=7200 notebooks/18_final_checks.ipynb
uv run --locked python tools/check_notebooks.py
uv run --locked python tools/compare_reruns.py /path/to/clean/rerun/checkout
```

Needs ignored `data/raw`, `artifacts/01_data_setup`, the pinned
`artifacts/16_local_electra/checkpoint`, and the #16/#17 private score files
named in the record. Their hashes are checked against the #16 and #17 records.
The refit itself trains from raw files and uses no saved models. Models, error
cases, and submissions stay in ignored `artifacts/` and `submissions/`.
