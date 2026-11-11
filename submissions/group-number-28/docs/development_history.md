# Q1 development evidence and archive history

## What this history represents

The baseline and expanded Q1 experiments were completed in the local workspace
before their implementation/results were committed. The commits in this PR
retrospectively archive those completed stages. Their commit dates are the real
archive dates, not the original execution dates; they must not be described as
contemporaneous experiment checkpoints. The original scaffold commit is
`17aba0e1f0c7b87fa597b4afb53e0cf8db61b2f9`.

Run provenance truthfully records the scaffold commit, a dirty working tree,
the UTC execution time, dependency versions, and hashes of the code actually
executed. Study directories also preserve source snapshots. These original
records are not rewritten to point at later archive commits.

## Experimental progression

1. **Baseline:** `a_q1_20261107_v1_study` compares word TF-IDF + LR, NB and
   calibrated SVM. Three candidate values per family, three training-only folds;
   SVM validation AUROC 0.943876, LR 0.943436, NB 0.906051. The small SVM/LR
   difference is unresolved. Vocabulary/IDF and calibration avoid outer holdout
   fitting. A separate 10,000-label SVM creates the local prediction candidate.
2. **Expanded comparison:** `a_q1_expanded_20261108_v2_study` adds decision
   trees, random forests, histogram gradient boosting and an MLP, plus an SVD LR
   representation control. A predeclared coarse search and one local refinement
   evaluate 64 configurations / 192 candidate-fold fits. LR's old grid boundary
   motivates a broader C range; training-only CV selects C=33.33333333333333.
   Dense models fit fold-local SVD/scaling. The same validation set had already
   been inspected, so the expansion is a development comparison, not a fresh test.
3. **Checks and debugging:** tests exercise split/prediction contracts,
   fold-local feature fitting, calibration scope, probability class direction,
   immutable run paths and refinement behavior. A floating-point exact-equality
   assertion in the new refinement test was changed to an approximate assertion;
   this corrected the test without changing model training. The broken test
   state was not committed. The frozen training logger emits a scalar-conversion
   deprecation warning with pinned NumPy; original code/hashes are retained.
   All eight selected expanded models were refit in fresh instances and matched
   their saved validation probabilities exactly in the existing environment.
4. **Selection and handoff:** SVM remains the nominal candidate; retuned LR
   is only 0.000177 behind. Full search tables, conditional bootstrap intervals,
   negative results, MLP seed variability and experiment figures are retained.
   The expanded full-data candidate is byte-identical to the baseline candidate.
   No Kaggle upload or leaderboard-based model selection occurred.

## Evidence index

- `experiments/supervised/README.md`: current results, exact dependencies,
  Python version, training/evaluation/export commands, and limitations.
- `results/supervised/*_study/`: plans, code snapshots, source/split hashes,
  fitting IDs, fold assignments, all candidate scores and selection rationale.
- `results/supervised/*/{config.json,metrics.json,validation_predictions.csv}`:
  actual settings, observed metrics/timing, and probabilities aligned by ID.
- `results/supervised/*_verification.json` and delivery/candidate checks:
  model reloads, fresh refits, metric reconstruction and output-contract checks.
- `report/figures/`: experiment comparison and search figures. Report prose,
  TeX sources, build entrypoints and PDFs are maintained separately in Overleaf
  and are excluded from the current code PR.
- `submissions/*.metadata.json`: candidate provenance and SHA-256; the matching
  small probability CSVs are included, without raw passages or model weights.
- `docs/lin_q1_handoff.md`: team transfer details and AI-use disclosure draft.

## Scope and remaining work

The original archive is an internal Q1 PR in the private team repository. It
is not the final course submission. Q5 has now been measured separately as
described below. Q2-Q4/Q6, team-wide final selection, and a clean-environment
cross-machine reproduction remain to be integrated. Team members' personal
verification is not claimed on their behalf. The final course PR must retain
the substantive archive/development commits when moving into the course fork's
per-team directory; record any old/new commit mapping if that migration changes
commit IDs. Do not squash away this evidence before course review.

Future experiments and fixes should be committed when they happen, with the
hypothesis, measured result and resulting decision recorded alongside the code.

## Subsequent Q5 execution

After the four-condition protocol was discussed and the user authorized its
execution, `a_q5_20261108_v1_study` ran from commit
`2c6f5960fa4f1d5266b8e5321a0c0cf0b18c508e` with the new Q5 files uncommitted.
Its provenance preserves that actual dirty state, exact executing-source
hashes and frozen snapshots. The subsequent real archive commits are
listed below; their dates are archive dates, not earlier execution checkpoints.

- Q5 implementation: [cccc067](https://github.com/lins369/kaggle_ai_vs_human/commit/cccc067bb48e8d35a13447247d13a6b513b12d7f)
- Q5 measured evidence: [784adce](https://github.com/lins369/kaggle_ai_vs_human/commit/784adce0691d0bd8f97dbffb190dd7c4f290f66e)

The fixed SVM/representation comparison records W60 0.943876, C60 0.969550,
WC120 0.970006, and W120 0.944597. The primary WC120-W60 difference is
+0.026130, conditional paired interval [+0.020083, +0.031971]. Training-only
folds agree in direction; the original W60 probabilities reproduce exactly.
Four Q5 tests and ten shared tests passed, and the saved-model/metric/bootstrap
verification passed. The detached local job finished without interruption.
No adaptive grid, full-data submission, or Kaggle upload was added in that study.
Independent reproduction, hidden-test transfer, source leakage, and the small
WC120-C60 difference remain unresolved. Technical results/commands are in the
supervised README; report prose remains in Overleaf.

### Subsequent Q5 full-label candidate export

The user later requested a suitable Kaggle submission file. The existing WC120
selection was refit, without tuning, on the canonical train-plus-validation
order of all 10,000 supplied labels. This separately recorded operation ran
from commit `38a5c0b0d87c90e805c5ce9626e940ddec32ddb6` with the new exporter
uncommitted; provenance and exact executing-source snapshots retain that state.
Fit time was 41.49 seconds and total export/check time was 55.41 seconds.

The resulting `submissions/a_q5_20261108_v1_wc120_full10000.csv` contains 3,000
validated probabilities, with exact agreement after reloading its saved model.
CSV/model/test/source hashes are recorded. The exporter did not upload to
Kaggle or modify original held-out artifacts. This full-label model has no
held-out evaluation; the original 0.970006 remains a development score for the
8,000-row model. No publication commit for this later export is claimed here.

### Bounded regularization follow-up after public feedback

The user then authorized C60/WC120 x C={0.1,1,10}, keeping the original split
and representations. `a_q5_tuned_20261108_v2_study` preserves the plan frozen
before its scores, actual dirty-state provenance from commit
`38a5c0b0d87c90e805c5ce9626e940ddec32ddb6`, exact source snapshots and all 18
candidate-fold results. Both representations selected C=1; independent fresh
8,000-row fits reproduce the historical validation predictions exactly.

The predeclared eligibility checks retained the existing WC120 candidate and
prevented exporting a duplicate or unsupported replacement. Quote/dash
evaluation-only sensitivity, changed counts, paired intervals and all negative
results are retained. Five new tests and full saved-artifact verification
passed. The local background job completed without interruption in 174.30
seconds including verification and plotting; no new full-label model or Kaggle
upload was produced. No archive commit for this follow-up is claimed here.

### Quote intervention resumed after preparation interruption

The next interruption occurred while preparing the quote-study verifier;
`a_q5_quotes_20261108_v3` had not started training. On resumption, historical
source/artifact hashes were checked, verification and serialization checks were
completed, and the frozen single-intervention plan was executed. The run used
commit `38a5c0b0d87c90e805c5ce9626e940ddec32ddb6` with uncommitted additions;
actual source snapshots and dirty-state provenance are preserved. Its local
execution date was November 8 (America/New_York); machine timestamps are UTC.

The job ran from 2026-11-09 01:54:48.759192 UTC to 02:01:08.393732 UTC,
completing 12 training-fold fits, two normalized 8,000-row models, the full
12-condition comparison and independent verification. Quote-normalized WC120
improved matched quote-condition validation AUROC by 0.001522, conditional
interval [0.0005494, 0.0025033], while original-text AUROC fell by 0.001331.
This meets the frozen exploratory tradeoff rules, not a claim of general
superiority. C60 normalized was ineligible against the incumbent.

One full-10,000-label WC120 normalized refit produced
`submissions/a_q5_quotes_20261108_v3_wc120_norm_full10000.csv` with 3,000
validated probabilities. That refit has no held-out evaluation. Five tests,
artifact reconstruction, exact model-reload predictions and figure inspection
passed. A completed-run resume check performed no fits or overwrites. Runtime,
test and postprocessing-source receipts accompany the evidence. The previous
public-0.87399 CSV remains the incumbent; no Kaggle upload, new public score or
publication commit is claimed for this run.

### Explicitly authorized four-group length/richness ablation

The user next explicitly authorized Route B2's four groups. The separate
`a_q5_richness_20261108_v4_study` froze document-local formulas, numeric scaling
and weights, the primary R3-R1 contrast, training-only candidate selection and
export guards before its scores. Original character features, C=1, calibration,
seeds and canonical split were retained; quote normalization was not added.
Actual execution provenance again records commit
`38a5c0b0d87c90e805c5ce9626e940ddec32ddb6` with uncommitted additions and exact
source snapshots, rather than claiming a publication commit.

The local November 8 job ran from 2026-11-09 03:28:12.380990 UTC to
03:36:43.884712 UTC. Twelve training-fold fits and three new calibrated
8,000-row models completed; the checksum-guarded historical C60 model was
reused. Held margins and fold scaler statistics allow independent reconstruction.
R0/R1/R2/R3 validation AUROCs are 0.969550/0.969825/0.974508/0.974838.
The primary richness increment after explicit length adjustment is +0.005013,
conditional interval [+0.00200635, +0.008241825], with three positive training
fold differences. Additional length and interaction benefits are not established.

Training-only CV selected R2 before validation, with R2/R3 means nearly tied
(difference 0.0000004095). R3's higher validation point score did not trigger a
post-validation switch. Selected R2 improves against WC120 by 0.004502,
conditional interval [+0.00103495, +0.008509275], passes all frozen export
checks and was refit on all 10,000 labels. The resulting
`submissions/a_q5_richness_20261108_v4_c60_richness_full10000.csv` contains
3,000 validated probabilities; its full-label model has no held-out score.

Nine tests, independent feature/scaler/margin/prediction/bootstrap/selection
verification, exact model-reload predictions and visual figure inspection
passed. All recorded fit warning lists are empty. Training/evaluation/export
took 469.01 seconds and the complete job took 511.50 seconds without
interruption. A completed-run resume check performed no fits or overwrites.
The old public-0.87399 incumbent and all prior study evidence are unchanged.
There was no Kaggle/GitHub upload or new public score. Hidden-domain transfer,
source leakage and the individual contributions of the correlated richness
features remain unresolved.

## Follow-up publication and historical archive commits

The subsequent follow-up archive on `lin/q5-robustness-richness-results`
includes the full-label exporter, bounded regularization study, quote
intervention and four-group richness study, with their verified local evidence,
figures and candidate CSVs. Execution-time commit/dirty-state provenance and
source snapshots remain unchanged. Earlier statements about absent publication
describe the state when each experiment was executed. This archive does not
add a Kaggle upload, score, model refit or claim of independent-machine testing.

Earlier Q1 archive commits:

- baseline: [4696415](https://github.com/lins369/kaggle_ai_vs_human/commit/469641522e434d896d1f6afe3a240ccc664d491a)
- expanded: [0936f15](https://github.com/lins369/kaggle_ai_vs_human/commit/0936f152c7f66d0a897f004a8614ae815fd314e0)
- checks: [7640f1b](https://github.com/lins369/kaggle_ai_vs_human/commit/7640f1b5b0e90176b5b04a123f650a5ff683cbf8)

Consolidated documentation was added in the following documentation commit.
These are real archive commits created after the recorded experiments.
That earlier commit also included local report artifacts; a later scope
correction removes them from the current tree because report writing belongs
in the team's separate Overleaf project. Original execution records and
substantive experiment history remain unchanged. Historical delivery receipts
may retain checksums of those earlier local PDFs; these are historical checks,
not a requirement to include or regenerate reports in the code repository.

## Public-score diagnosis and crop transfer archive (November 9)

The user's later screenshot records quote WC120 public AUROC 0.87225 and
C60+richness 0.85812; WC120 at 0.87399 remains the incumbent. New overlap,
marginal-shift and paired stress diagnostics found no strong lexical duplicate
boundary issue and measured crop/whitespace sensitivity. The crop-weight-0.5
pilot and two separately recorded weak-crop studies preserve their own
fitting boundaries and development limitations.

The original weak-crop run stopped after six diagnostic random-CV fits when
its fixed topic proxy failed two engineering checks. The user then authorized
an adaptive revision making only content-topic count and maximum held-fold
size diagnostic. The exact same topic/fold IDs and frozen reviews were reused,
with a new exactly count-matched random control. Twelve new fits measured
topic clean difference -0.003219 and prefix difference +0.037687. The clean
cost narrowly exceeded the unchanged operating budget, with uncertainty
spanning its boundary; this is a tradeoff, not a model-invalidity claim.

No new candidate, Kaggle upload or canonical-2,000 evaluation was produced by
the two weak-crop studies. Independent zero-fit verification passed. Their
execution provenance, parent hashes and frozen snapshots remain byte-exact.
Publication excludes passages, model binaries, caches and local report files.
Checkpoint-dependent commands require the ignored local reference artifacts;
a fresh-clone model-level rerun was not performed.

Actual archive commits created after these experiments:

- Implementation: [95fc79a](https://github.com/lins369/kaggle_ai_vs_human/commit/95fc79a8b88c56e19fab3b0615622fcbf4b9dcba)
- Evidence: [62e2030](https://github.com/lins369/kaggle_ai_vs_human/commit/62e2030fc38f905a7e3629fccc5a20ba81e7b240)
