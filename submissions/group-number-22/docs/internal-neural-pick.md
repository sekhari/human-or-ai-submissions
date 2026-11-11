# Re-pick the ELECTRA blend weight on training rows

Issue #37. Owner: Jalil-g. Depends on #18.

## Why

#18 picked the ELECTRA weight (.25) on frozen 2000 validation rows. AGENTS.md
says tuning stays inside training. #16 never saved inner 1600 ELECTRA scores,
so #18 had no training rows with both scores.

## What we tried

Refit #17 classical (`hist50_old50`) and #16 ELECTRA (3 epochs, seed 42) on the
inner 6400 rows, same split as #12, #14, #16, #17. Scored inner 1600 under
original, remove blank lines, collapse whitespace, wrap80. Standardized on
eligible unlabeled. Same #18 grid and rule: w in 0, .1, .25, .5, .75, 1, max mean
AUROC over four views, original >= w=0 original - .002, ties to smaller w. Pick
saved before frozen 2000 was read.

## Results

Inner 1600 (picks w):

| w | original | remove blank | collapse | wrap80 | mean | eligible |
|---:|---:|---:|---:|---:|---:|---|
| 0 | .998848 | .988433 | .953923 | .991481 | .983171 | yes |
| .1 | .999008 | .992292 | .971695 | .994675 | .989418 | yes |
| **.25** | .998556 | .992792 | .981716 | .995623 | **.992172** | yes |
| .5 | .996242 | .990838 | .984481 | .994325 | .991471 | no |
| .75 | .992231 | .988259 | .985297 | .991013 | .989200 | no |
| 1 | .983045 | .983045 | .983055 | .983050 | .983049 | no |

- Pick: **w = .25**, same as #18. Submitted model stands.
- Frozen 2000 for w = .25 (confirmation only, 8000-fit): original .999083,
  collapse .987313. Same numbers as #18, but now they do not pick anything.
- Checks: classical inner reproduces #17 internal exactly (diff 0). ELECTRA inner
  original .983045 vs #16 .983039 (MPS is not bitwise repeatable). Saved #18
  10000-fit models reproduce the submitted file to 8e-17.
- Runtime 343 s on Apple M5 MPS. Record:
  `results/19_internal_neural_pick/20261111T024705598557Z.json`, source 12da826.

## Caveats

Frozen 2000 was reused in #4, #8, #9, #13, #14, #15, #17, so confirmation is
optimistic. Kaggle public 0.98553 (ref 58222131) is the real held-out number.
No new submission. Nothing uploaded by this notebook.
