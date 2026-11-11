# Supervised comparisons and controlled improvements — Q1, Q5

Owner: Lin (A). **Q1 and the frozen Q5 feature ablation are measured locally.**
Alex owns Q2/Q4; Gabriel owns Q3/Q6 integration.

## Public-score review (November 9)

**Latest feedback: raw RoBERTa Public Score 0.96970**, versus previous WC120
0.87399 (+0.09571), in the user's Kaggle screenshot. The submitted candidate
`a_roberta_20261109_v1_raw_full10000.csv` uses a separate full 10,000-label refit;
all 3,000 predictions and full checkpoint reload agreement pass verification.
See the [public result and screenshot](../../docs/roberta_public_result_20261109.md).
The prior experiment/export receipts remain unchanged historical snapshots.

The subsequent contextual comparison is complete in
`results/supervised/a_roberta_context_20261109_v1_study/`: all six frozen
random/topic CV fits plus the canonical fit with 8,000 originals, two epochs each.
Clean macro AUROCs are 0.996673/0.995652 versus WC120 0.965848/0.932745;
whitespace-only means fall to 0.847715/0.862552. All 35 condition rows, unfavorable
folds, paired intervals, ranking-error decompositions, measured costs and
independent audits are retained. A separate zero-fit diagnostic on the fixed
canonical checkpoint changes prefix100 AUROC 0.782456 -> 0.990948 solely by
preserving whitespace at identical word content. See the
[scientific results](../../docs/roberta_context_results_20261109.md),
[diagnostic](../../docs/roberta_prefix_whitespace_results_20261109.md) and
[frozen protocol](../../docs/roberta_context_protocol_20261109.md).

Neural commands use `/opt/anaconda3/envs/python_course/bin/python`; direct tested
packages are recorded in `requirements_roberta.txt`. Preparation reuses the
historical saved WC120 model for inference in its original environment and
requires the local ignored inputs. A clone alone is insufficient for a fresh
model-level reproduction; no fresh-clone or independent-environment fit is
claimed. Actual training commands and artifact hashes are in the study's
`execution_receipts.json`. No calibration/blend fitting or candidate export
occurred in this comparison.

The first contextual-model device measurement is complete in
`results/supervised/a_roberta_mps_20261109_v1/`: three FP32 RoBERTa MPS
configurations, each with five warmup and 20 timed full training updates.
All update/reload checks and independent evidence checks passed. This measures
device feasibility, with no held/test score or candidate export. See the
[device results](../../docs/roberta_mps_device_results_20261109.md) and use
`/opt/anaconda3/envs/python_course/bin/python` for neural training.

The user screenshot records WC120 **0.87399**, quote WC120 **0.87225** and
C60+richness **0.85812**. That was the earlier WC120 public lead. The sections below
preserve each study's original local evidence and export-time status. Current
scores, executed no-refit audits and follow-up research are in
[`docs/public_score_research_20261109.md`](../../docs/public_score_research_20261109.md).
The reused random validation set does not establish transfer to unseen domains,
generators or adversarial text.

Weak-crop status: **executed; stopped without export as declared**.
The [frozen protocol](../../docs/next_round_crop_protocol_20261109.md) specifies
one 0.1-weight intervention and at most 24 SVC fits. The topic proxy failed
quality checks, so only six fallback diagnostic random-CV fits ran. Their clean
mean changed 0.965186 -> 0.962823, prefix mean 0.855860 -> 0.906588, and random
screening passed. Topic, sealed-condition and calibrated-development stages
were not promoted. See the [execution record](../../docs/q5_crop_results_20261109.md)
and `results/supervised/a_q5_crop_20261109_v5_study/`. Current implementation is
`crop_model.py`, `crop_topics.py`, `crop_evaluation.py` and `run_q5_crop.py`;
machine-readable settings remain in `q5_crop_plan.json`.

The user-authorized adaptive revision is also complete:
`a_q5_crop_revised_20261109_v6_study`. Only the two challenged grouping limits
became diagnostics; prior groups/folds/reviews stayed fixed. Matched random and
topic CV each used six new fits. Topic prefix gain is +0.037687, with clean cost
0.003219, narrowly beyond the unchanged 0.003 operating budget. No later-stage
fits/export ran. [Revised results](../../docs/q5_crop_revised_results_20261109.md),
[revision protocol](../../docs/q5_crop_revised_protocol_20261109.md) and
`q5_crop_revised_plan.json` preserve the adaptation and uncertainty.

Reproduction prerequisite: crop/stress entrypoints and full verification require
the supplied CSVs plus the ignored local reference/model checkpoints pinned in
their plans. A clone alone is insufficient; no fresh-clone model-level rerun is
claimed. The revised result note records the exact `--plan` command. Numerical
CSV evidence and frozen sources are committed separately from model binaries.

## Four-group length/richness ablation (November 8)

Study: `a_q5_richness_20261108_v4_study`. This separately frozen follow-up
implements Route B2 after the user explicitly authorized four-group execution.
The original split, character (3,5) TF-IDF with 60,000 features, C=1, calibration
and seeds are unchanged. No quote normalization, C search, new split, external
data or adaptive feature/weight search is included.

| Group | Numeric blocks | Mean training-only CV AUROC | Original validation AUROC | Actual features |
|---|---|---:|---:|---:|
| R0: character | none | 0.963434 | 0.969550 | 60,000 |
| R1: character + length | length | 0.963743 | 0.969825 | 60,002 |
| R2: character + richness | richness | 0.967464 | 0.974508 | 60,003 |
| R3: character + both | length, richness | 0.967464 | **0.974838** | 60,005 |

The predeclared primary contrast **R3 minus R1** is **+0.005013**, conditional
paired 95% interval **[+0.00200635, +0.008241825]**. Its training-only CV gains
are positive in all three folds (mean +0.00372041). This supports incremental
prediction from the richness block conditional on these explicit length
features and this model geometry. It does not prove complete length
independence or a causal authorship mechanism.

Secondary contrasts: R1-R0 +0.000275, interval [-0.0005961, +0.00113635];
R2-R0 +0.004958, interval [+0.001469775, +0.008531525]; R3-R2 +0.000330,
interval [-0.00026605, +0.000961175]. The four-term interaction is +0.000055,
interval [-0.000968075, +0.0011081]. Additional length features and an interaction
are therefore not established benefits. Secondary intervals are descriptive,
without multiplicity or selection adjustment.

### Frozen features and fitting boundaries

- Tokens are lowercase Unicode letter-only words, using
  `(?u)\b[^\W\d_]+\b`; numeric/mixed/underscore tokens are excluded and
  apostrophes split words. Character TF-IDF still receives the original text.
- Length: `log1p(letter-token count)` and `log1p(raw Unicode character count)`.
- Richness: moving average type-token ratio over every contiguous 50-token
  window; Shannon unigram entropy divided by `log(token count)`; repeated
  bigram occurrence fraction `(number of bigrams - distinct bigrams)/number
  of bigrams`. Short MATTR documents use one window of their available tokens;
  empty and undefined short entropy/repetition cases return zero. There are
  no documents shorter than 50 letter tokens in either canonical partition.
- Each numeric block has its own `StandardScaler`, fitted only inside its
  permitted training/CV/calibration boundary, then weight `1/sqrt(d)` for its
  dimension d. The original character block has weight 1; no final row
  normalization is added. Expected training squared norm of a numeric block
  is `nonconstant_columns/d`. Adding blocks changes geometry and effective
  regularization; fixed C does not equalize capacity.

The document-local features are an adaptation, not a reproduction of a
published detector. [MATTR reference](https://doi.org/10.1080/09296171003643098)
and [training-only preprocessing guidance](https://scikit-learn.org/1.5/common_pitfalls.html)
inform the definitions and boundaries. These aggregate existing text rather
than introducing an external data source. All three richness statistics can
remain length-dependent. Training log-token-count correlations with the
three richness features are -0.0712, -0.2404 and +0.1837; the explicit length
block is a partial modeling control, not a complete removal of dependence.

### Candidate choice and diagnostics

The candidate was chosen using training-only CV **before** the validation
matrix. R2 narrowly leads R3 by just 0.0000004095 in mean CV, so the frozen
rule chooses R2; this near tie does not establish R2 superiority. R3 has the
higher validation point score, but no switch based on validation was made.
R2 minus the historical WC120 incumbent is **+0.004502**, interval
**[+0.00103495, +0.008509275]**; its CV gains against WC120 are positive in
two of three folds, with a mean gain of +0.00308297.

| Original-text length bin | n (human / AI) | WC120 incumbent | R2 |
|---|---:|---:|---:|
| Short | 646 (317 / 329) | 0.961560 | 0.967150 |
| Middle | 681 (389 / 292) | 0.971687 | 0.977022 |
| Long | 673 (294 / 379) | 0.978291 | 0.979080 |

Bins retain the original training whitespace-word-count tertiles, 174/240;
these subgroup point gains are descriptive. R2 original/quote/dash AUROCs are
0.974508/0.971734/0.974527. It remains quote-sensitive; the already-inspected
typography conditions do not demonstrate independent OOD robustness.

All four pipelines received three fresh raw-margin CV fits, with held margins
and fitted scaler statistics retained. The checksum-guarded historical R0
calibrated model was reused; three new calibrated 8,000-row models were fitted.
Their fit times were 38.11/54.41/57.17 seconds for R1/R2/R3. Fewer columns do
not imply faster fitting than the historical WC120 fit (33.66 seconds).
The condition matrix includes the four groups plus WC120, each under three
input conditions. Original Q1/Q5/tuning/quote evidence remains unchanged.

R2 passed all frozen export checks: positive validation interval lower bounds
against both R0 and WC120, CV consistency against WC120, length-bin regression
budget 0.01 and extra typography-loss budget 0.003. These are development
guardrails, not proof of private/public leaderboard improvement. The separate
eligible full-label refit exports at most one CSV:
`submissions/a_q5_richness_20261108_v4_c60_richness_full10000.csv`.
The old public-0.87399 WC120 file remains the incumbent until an external check.
The full-label refit completed in 67.79 seconds and exported 3,000 checked
probabilities (3,001 lines including the header), with exact model-reload
agreement. It has no held-out score; 0.974508 belongs to the 8,000-row model.
No Kaggle upload occurred.

Nine tests passed, including exact formulas, Unicode/short inputs, fold-only
fitting, complete calibration boundaries, independent-process serialization
and checkpoint reuse. The actual test stdout is archived in the study.
Independent verification passed, reconstructing all CV AUROCs from saved held margins,
fold scaler statistics from fitting IDs, matrix predictions, 15 AUROC intervals,
11 contrasts, length groups, numeric diagnostics and selection/export checks.
No full-size refits were performed by the verifier. Training/evaluation/export
took 469.01 seconds; the job including verification and plotting took 511.50
seconds, without interruption. All CV and new-model warning lists are empty.
A completed-run resume check skipped all fits and overwrites.

A new run uses the pinned environment, a NEW prefix, and the historical
C60/WC120 reference checkpoints required by the frozen plan. Those model
binaries are intentionally local and ignored by Git. A fresh clone must
reconstruct and verify the historical references before running this follow-up;
independent clean-environment reproduction has not yet been completed.

```sh
.venv/bin/python -m unittest experiments.supervised.test_q5_richness -v
.venv/bin/python -m experiments.supervised.q5_richness_job --prefix a_q5_richness_reproduction_001
```

Use `--resume` only with the same prefix and unchanged source, environment,
protocol, split and reference evidence. Atomic fold/model checkpoints are
retained. The job runs independent verification and plotting after training.
Figure: `report/figures/q5_a_q5_richness_20261108_v4_richness.png`.
Source snapshots, numeric diagnostics, held margins and paired uncertainty are
in the study directory; separate verification/delivery receipts are under
`results/supervised/`. This remains development research on a reused validation
set; intervals omit training variation and selection adjustment. Source/group
leakage and hidden-domain transfer remain unresolved.

## Quote canonicalization intervention (November 8)

Study: `a_q5_quotes_20261108_v3_study`. The earlier interruption occurred during
verification-code preparation, before this study started training. After
checking the historical evidence and completing the verifier, this separate
bounded run completed with the original 8,000/2,000 split, C=1, features,
calibration and seeds unchanged. Its only intervention maps curly quotation
marks/apostrophes to their ASCII equivalents during both training and prediction.
The stateless transformer is inside the complete calibrated pipeline; dash
handling is unchanged. The plan was frozen before v3 scores, after v2 diagnostics.

| Representation / training | Original AUROC | Quote mapping | Dash mapping |
|---|---:|---:|---:|
| C60 / raw | 0.969550 | 0.966299 | 0.969529 |
| C60 / quote normalized | 0.967798 | 0.967798 | 0.967816 |
| WC120 / raw | 0.970006 | 0.967153 | 0.970036 |
| WC120 / quote normalized | 0.968675 | 0.968675 | 0.968698 |

The primary matched WC120 quote-condition difference is **+0.001522**, with a
1,000-replicate conditional paired interval **[+0.0005494, +0.0025033]**.
The original-text cost is **-0.001331**, interval **[-0.0026960, -0.0000022]**.
Normalized WC120 improves quote-condition raw-margin AUROC in all three
training-only folds; mean quote CV changes from 0.961231 to 0.962077, while
mean clean CV falls from 0.964381 to 0.962077. The largest length-bin regression
against the incumbent is 0.003500, in short texts. This is a measured tradeoff,
not an across-the-board improvement or an established hidden-test gain.

All 12 training-fold fits, two new calibrated 8,000-row normalized fits, and
12 model/input-condition evaluations completed. The raw validation models were
reused with checksum guards, and their fresh CV clean scores reproduce v1.
Quote mapping changes 157 validation texts; dash mapping changes 177. Equal
raw/quote predictions for normalized models follow by construction. Dash was
already inspected in v2 and is an additional diagnostic, not a new holdout.
Intervals condition on this development split and fitted models and omit
training variability and selection adjustment.

WC120 normalized passed the predeclared **exploratory** export checks against
the incumbent: positive quote-interval lower bound, consistent quote CV gains,
clean point/CV losses within 0.003, length-bin losses within 0.01, and extra
dash loss within 0.005. The 0.003 budget is a practical exploratory preference,
not a statistical noninferiority margin. C60 normalized failed the clean-CV
budget and quote-interval check against the incumbent. At most one candidate
was permitted, with WC120 first in priority.

The eligible `exploratory_robustness_tradeoff` candidate was refit on all 10,000
labels and exported to
`submissions/a_q5_quotes_20261108_v3_wc120_norm_full10000.csv`: 3,000 test
probabilities, `id,label`, 3,001 lines including the header. Its full-label model
has no held-out score; 0.968675 belongs to the separate 8,000-row model. Reload
predictions match exactly. No Kaggle upload occurred, and the existing
public-0.87399 WC120 submission remains the incumbent pending an external check.

Five tests passed, covering the mapping, calibration/fold boundaries,
independent-process serialization and checkpoint reuse. The independent
artifact verifier reconstructed source/split/fold hashes, all matrix predictions,
bootstrap intervals, length bins, eligibility and full-label export; all checks
passed. Original Q1/Q5 evidence is unchanged. Training/evaluation/export took
346.71 seconds; the full job with verification and plotting took 379.63 seconds.
The completed-run `--resume` check skipped all fits and overwrites. Figure:
`report/figures/q5_a_q5_quotes_20261108_v3_quotes.png`.

Fresh reproduction uses a NEW prefix and the pinned environment:

```sh
.venv/bin/python -m unittest experiments.supervised.test_q5_quotes -v
.venv/bin/python -m experiments.supervised.q5_quotes_job --prefix a_q5_quotes_reproduction_001
```

Use the same prefix plus `--resume` only for identical interrupted settings.
The driver runs training, independent verification and plotting sequentially;
checkpoints reject changed source, protocol, split or reference evidence.
The study directory retains the frozen plan/source snapshots, CV progress,
comparison, predictions, bootstrap and selection. Separate verification and
delivery receipts are in `results/supervised/`; postprocessing source hashes,
test results and runtime summary are retained in the study directory.

## Bounded Q5 regularization follow-up (November 8)

Study: `a_q5_tuned_20261108_v2_study`. After the two user-reported public
scores (W60 0.85157; WC120 0.87399), a separate six-configuration protocol
compared C60/WC120 with SVM C in {0.1, 1, 10}. It preserved the canonical split,
features, seeds and calibration. The original Q5 plan/evidence were not changed.

| Representation | Mean training-only CV, C=0.1 | C=1 | C=10 |
|---|---:|---:|---:|
| C60 character | 0.956520 | **0.963434** | 0.960097 |
| WC120 word + character | 0.953765 | **0.964381** | 0.962976 |

All 18 candidate-fold fits completed. Features were fitted independently in
each of six representation/fold boundaries and cached only across the three
C values within that boundary. Each representation selected C=1 from training
CV before finalist validation. Both finalists were freshly fitted on 8,000
rows; validation AUROCs 0.969550/0.970006 exactly reproduce the old C60/WC120
probabilities in this environment. C60 minus WC120 is -0.000456, with a
conditional paired interval [-0.002611, +0.001900]; no clear superiority is
established. Independent-machine reproduction is still pending.

No new candidate passed the predeclared export rules. WC120 C=1 is the same
configuration as the existing submission, and C60 did not improve CV or have a
positive-lower-bound paired validation difference. The existing public-0.87399
WC120 CSV remains the incumbent; this follow-up made no new test predictions,
full-label refit, CSV export or Kaggle upload.

The fixed-model typography diagnostic measured:

| Representation | Original AUROC | Quote mapping | Dash mapping |
|---|---:|---:|---:|
| C60 C=1 | 0.969550 | 0.966299 | 0.969529 |
| WC120 C=1 | 0.970006 | 0.967153 | 0.970036 |

Quote mapping changed 157/2,000 documents (91 human, 66 AI); dash mapping
changed 177 (82 human, 95 AI). At threshold 0.5, quote mapping flipped 21 C60
and 15 WC120 decisions; dash mapping flipped none. These are observed local
sensitivities, not significance claims or proof of the public-gap mechanism.
Only evaluation text was changed in v2; the separate v3 intervention above
subsequently evaluated normalized training.
Formatting predictions and changed-subset diagnostics retain original IDs.

Five new tests passed (three tuning/selection/resume tests and two sensitivity
tests). The independent verifier reconstructed all CV summaries, source/split
guards, finalist probabilities, bootstrap intervals, nine typography conditions,
length diagnostics and export eligibility; all checks passed. The detached
job completed training/diagnostics in 155.98 seconds and all postprocessing in
174.30 seconds without interruption.

Fresh reproduction, with a NEW prefix:

```sh
.venv/bin/python -m unittest experiments.supervised.test_q5_tune experiments.supervised.test_q5_sensitivity -v
.venv/bin/python -m experiments.supervised.q5_tune_job --prefix a_q5_tuned_reproduction_001
```

`--resume` continues identical unfinished settings and rejects changed source,
protocol or reference evidence. It never expands the C grid. An eligible future
reproduction can export at most one candidate, after local selection, but never
uploads it. See `q5_tuning_plan.json` for the exact export guardrails, tie rules
and selection-adjustment limitations. The comparison figure is
`report/figures/q5_a_q5_tuned_20261108_v2_tuning.png`.

The measured quote sensitivity motivated the separate v3 train-and-infer
canonicalization ablation above. It did not establish that canonicalization
improves clean or hidden-test performance. No second intervention was added to
this bounded v2 run, and report prose remains in Overleaf.

## Q5 character-feature ablation (November 8)

Study: `a_q5_20261108_v1_study`. This is a separate, prospectively frozen
four-configuration study, not an extension of the Q1 classifier grid. All
models retain SVM C=1, the canonical 8,000/2,000 split, and the Q1 sigmoid
calibration protocol. No C, character range, feature weight, or classifier
search was added after observing Q5 scores.

| Configuration | Actual features | Mean training-only 3-fold AUROC | Validation AUROC | Full fit seconds |
|---|---:|---:|---:|---:|
| W60: original word (1,2) baseline | 60,000 | 0.936043 | 0.943876 | 5.28 |
| C60: character (3,5) | 60,000 | 0.963434 | 0.969550 | 27.43 |
| WC120: word + character | 120,000 | 0.964381 | **0.970006** | 33.66 |
| W120: larger word vocabulary reference | 120,000 | 0.936876 | 0.944597 | 5.47 |

All four training AUROCs are 1.0. Timings are measured in the existing Python
3.12.7/pinned environment, with one CPU thread; they include feature fitting
and calibration, excluding diagnostic CV. Each combination block has unit L2
norm; equal-weight concatenation is normalized again. W60 uses the original
single vectorizer without an extra normalization pass and exactly reproduces
all Q1 validation probabilities (maximum absolute error 0).

The predeclared primary difference WC120 minus W60 is **+0.026130**, with a
1,000-replicate paired stratified bootstrap interval **[+0.020083, +0.031971]**.
WC120 improves raw-margin AUROC over W60 in all three training-only folds.
The W120 difference is +0.000721, interval [-0.000562, +0.001859]; merely
doubling the word feature budget does not reproduce the observed character
gain in this study. This resource reference does not strictly isolate model
capacity. C60 already captures most of the gain: WC120 is only 0.000456 above
C60 on validation, so superiority of the combination over character-only
features has not been established.

At threshold 0.5, WC120 corrects 107 W60 errors and introduces 34 new ones,
reducing total errors from 258 to 185 (accuracy 0.8710 to 0.9075). Descriptive
length bins were fixed from training-only word-count tertiles, at 174 and
240 words. WC120 improves over W60 in all three bins. Changed-error IDs and
probabilities are saved without passage text; passage-level causal hypotheses
still require human review.

The four saved models reproduce their recorded probabilities exactly after
reload. Source/split/fold hashes, CV summaries, calibration and normalization
boundaries, the primary interval, and error-change counts passed verification.
Four Q5 tests and ten shared contract tests passed. This is verification in the
existing environment, not an independent clean-environment reproduction.

Training/evaluation/bootstrap took 146.05 seconds; the detached job including
verification and plotting completed in about 159.26 seconds without interruption.
Atomic per-fold/per-model checkpoints and a process lock permit identical
unfinished studies to resume. The local worker calls no model/API service.
Its log is `tmp/a_q5_20261108_v1.log`; job state is
`tmp/a_q5_20261108_v1_job.json`. These operational files are ignored by Git.

Fresh reproduction, using a NEW prefix:

```sh
.venv/bin/python -m unittest experiments.supervised.test_q5 -v
.venv/bin/python -m experiments.supervised.q5_job --prefix a_q5_reproduction_001
```

Use `--resume` with the same prefix only after an interrupted identical run.
Resume rejects changed training source, plan, environment or split and reuses
completed folds/models. `q5_job` runs training, saved-artifact verification and
plotting sequentially. It does not upload or fit on all 10,000 labels.
For a separate audit, use a new output path:

```sh
.venv/bin/python -m experiments.supervised.verify_q5 --prefix a_q5_20261108_v1 --output results/supervised/a_q5_20261108_v1_additional_verification.json
.venv/bin/python -m experiments.supervised.plot_q5 --prefix a_q5_20261108_v1
```

Evidence includes `q5_plan.json`, `cv_progress.json`, `cv_results.csv`,
`comparison.csv`, `bootstrap.json`, `baseline_gate.json`, `changed_errors.csv`,
`error_change_summary.json`, and `length_subgroups.csv` in the study directory;
each model has its own config/metrics/predictions/checksum receipt. The figure
is `report/figures/q5_a_q5_20261108_v1_comparison.png`.

WC120 is the leading Q5 development candidate; C60 is a lower-dimensional
alternative. Final team selection remains open. If selected, Alex should use
`results/supervised/a_q5_20261108_v1_wc120/validation_predictions.csv` for Q4.
The four-condition study itself created no full-data checkpoint or test
predictions. A separate post-study export is now available as described below;
the earlier word-only submission and all 8,000-row study artifacts are unchanged.

### Separate Q5 Kaggle candidate

After the user requested a file for manual Kaggle scoring, the selected WC120
configuration was refit on all 10,000 supplied labels, with no additional search.
The 60k word and 60k character budgets, equal-block/global L2 normalization,
SVM C=1 and training-only sigmoid calibration remain fixed.

- CSV: `submissions/a_q5_20261108_v1_wc120_full10000.csv`.
- Artifact directory: `results/supervised/a_q5_20261108_v1_wc120_full10000/`.
- Training order: canonical 8,000 training IDs followed by 2,000 validation IDs.
- Fit time: 41.49 seconds; full export/checks: 55.41 seconds, one CPU thread.
- Exactly 3,000 unique test IDs in original order, `id,label`, class-1 probabilities.
- Saved-model reload probabilities match exactly; original held-out artifacts
  are unchanged. Schema/ID/probability and checksum verification passed.
- No convergence warnings or Kaggle upload occurred. This full-label refit
  has no held-out evaluation; 0.970006 belongs to the original 8,000-row model.

Metadata records the CSV/model/test/export-code hashes and refit provenance.
The model binary remains local and ignored by Git. Regenerate after a fresh Q5
study with a NEW output path:

```sh
.venv/bin/python -m experiments.supervised.export_q5 --prefix a_q5_reproduction_001 --model wc120 --output submissions/a_q5_reproduction_001_wc120_full10000.csv
.venv/bin/python -m human_ai.check_predictions submissions/a_q5_reproduction_001_wc120_full10000.csv --kind submission
```

Use `--resume` only to continue identical interrupted export settings. The
exporter rejects changed training source/settings/environment and never fits
features to test passages or overwrites completed candidates.

The validation set already informed Q1 development. All intervals condition
on this split and fitted models, omit training variability and selection
adjustment, and do not establish hidden-domain/test-set gains. Character
features could also encode source shortcuts; near-duplicate/source leakage
remains unassessed. Report prose continues to be maintained in Overleaf.

## Expanded Q1 (current, November 8)

Study: `a_q1_expanded_20261108_v2_study`. Seven classifier families plus a
representation control, **64 distinct settings / 192 candidate-fold fits**.
All parameter choices use three training-only folds, a coarse search and one
predeclared refinement. The original 8,000/2,000 split and original v1 evidence
are unchanged. Settings are best among those actually tested, not global optima.

| Model | Representation | Best tested settings | Validation AUROC | Fit seconds |
|---|---|---|---:|---:|
| Logistic regression | TF-IDF | C=33.33333333333333 | 0.943699 | 1.98 |
| Multinomial NB | TF-IDF | alpha=0.3 | 0.905623 | 1.75 |
| Calibrated linear SVM | TF-IDF | C=1 | **0.943876** | 6.08 |
| Decision tree | TF-IDF | depth=15; min leaf=20 | 0.812888 | 2.76 |
| Random forest | TF-IDF | 400 trees; unlimited depth; max_features=0.02; min leaf=1 | 0.936392 | 35.37 |
| Histogram gradient boosting | SVD-128 | 300 rounds; rate=0.1; 15 leaves; L2=0 | 0.939490 | 5.43 |
| Neural network / MLP | SVD-128 + scaling | hidden=(128,); alpha=0.0001; rate=0.003 | 0.939616 | 4.76 |
| LR representation control | SVD-128 + scaling | C=0.3 | 0.922601 | 4.18 |

TF-IDF settings are unchanged from v1. SVD (128 components, 7 power iterations,
seed 42) and StandardScaler are fitted only on each applicable training fold.
The scaled SVD LR control and MLP share a representation; boosted trees use
the same SVD components without scaling. Comparisons with sparse models also
change representation, so they do not isolate the classifier alone.

All timings include full feature fitting and any calibration, not parameter
search. The expanded search took 223.10 s; search plus eight final fits and
evaluation took 297.47 s before bootstrap/secondary-seed checks. Peak process
RSS was about 965 MiB (cumulative high-water mark, not isolated model memory).
All runs used one CPU thread; no GPU or external weights were needed.

Word-only SVM remains the Q1 candidate; the separate Q5 candidate is above.
Its lead over retuned LR is just 0.000177,
paired conditional bootstrap interval [-0.000468, 0.000794]. LR is a faster
alternative, and the difference is unresolved. All intervals omit training
variation and model-selection correction. The original validation set was
already inspected in v1: this extension is a development comparison, not an
untouched final estimate. No expanded choice used Kaggle feedback.

MLP uses ReLU/Adam, at most 500 epochs, internal 10% stratified stopping data,
patience 20, tolerance 1e-4; stopping monitors accuracy. Feature transforms use
all fitting rows, while gradient updates use 90%; its training diagnostic
includes the internal stopping rows. Selected primary MLP stopped after 27
epochs. Seeds 42/7/142 yield validation AUROCs 0.939616/0.936834/0.940165;
seed 42 remains primary, without selecting the best seed. No candidate fit
had a convergence warning. One physical-core detection warning was recorded;
threadpool limits remained one. The original logger also emits a NumPy
deprecation warning when recording scalar iteration counts; pinned versions
completed correctly and all eight refits reproduced predictions exactly.

For new reproduction, using the Python 3.12 environment described below:

```sh
.venv/bin/python -m unittest experiments.supervised.test_expanded_q1 -v
.venv/bin/python -m experiments.supervised.expanded_q1 --prefix a_q1_expanded_reproduction_001
.venv/bin/python -m experiments.supervised.verify_expanded_q1 --prefix a_q1_expanded_reproduction_001 --output results/supervised/a_q1_expanded_reproduction_001_verification.json --refit
.venv/bin/python -m experiments.supervised.export_expanded_q1 --prefix a_q1_expanded_reproduction_001 --output submissions/a_q1_expanded_reproduction_001.csv
```

Original expanded candidate (already generated, so do not overwrite):
`submissions/a_q1_expanded_20261108_v2_svm_full10000.csv`. Its selected settings
were refit on all 10,000 supplied labels and its 3,000 test probabilities pass
the CSV checker and saved-model reload check. It was not uploaded to Kaggle.

Reports are written separately in Overleaf. Report prose, TeX sources, build
entrypoints and PDFs are not part of this code PR. The actual grids and
refinements remain in the study's JSON/CSV evidence. To regenerate only the
experiment figures:

```sh
.venv/bin/python -m experiments.supervised.plot_expanded_q1 --prefix a_q1_expanded_20261108_v2
```

Use the expanded study's `cv_results.json` / `cv_results.csv` for every exact
parameter combination, `cv_progress.json` for all fold scores and stopping
iterations, `chosen_parameters.json` for selections, `bootstrap.json` for
uncertainty, and `mlp_secondary_seeds.json` for stability. Each of the eight
run directories contains parameters, source/split provenance, metrics and
canonical 2,000-row validation predictions. The audit confirms all eight fresh
refits match saved probabilities exactly in this environment. Cross-machine
or clean-environment reproduction remains pending.

The single tree is a clear negative result here. The forest improves greatly
over it, but costs 35.37 s without beating SVM. Boosting/MLP beat the SVD LR
control while trailing sparse LR/SVM nominally; nonlinear modeling helps this
representation, while information discarded by compression is a plausible
limitation. These results do not establish that other tree, ensemble, or neural
representations would fail. Forest tree count and boosting rounds remain at
the largest values tested, so further resource/performance trade-offs are open.

## Initial v1 results and protocol (historical record)

Study: `results/supervised/a_q1_20261107_v1_study/`. The canonical split is
unchanged: 8,000 train (4,000/class), 2,000 validation (1,000/class). All models
use the same 60,000-feature word unigram/bigram TF-IDF representation.

| Model | Selected setting | Mean inner CV AUROC | Train AUROC | Validation AUROC | Fit seconds |
|---|---|---:|---:|---:|---:|
| Logistic regression | C=10 | 0.935842 | 1.000000 | 0.943436 | 1.78 |
| Multinomial Naive Bayes | alpha=0.1 | 0.886528 | 0.992115 | 0.906051 | 1.69 |
| Calibrated linear SVM | C=1 | 0.936043 | 1.000000 | 0.943876 | 5.84 |

Times include features and SVM calibration, exclude inner tuning. These are
single wall-time measurements on macOS ARM64, one CPU thread. Tuning across
all families took 5.44 seconds; final refits took 9.30 seconds combined. No
GPU was used. Peak process RSS was about 632 MiB (cumulative, not per model).

SVM is the current Q1 candidate, but its lead over LR is only 0.000440. The
exploratory paired bootstrap interval [-0.000713, 0.001660] does not establish
superiority. LR remains a faster alternative. NB's lower score is retained as
a negative result. Team-wide final selection remains open after Q3/Q5. No
Kaggle score has been measured or inferred from validation AUROC.

## Reproduction from repository root

Tested: **Python 3.12.7** with exact versions in `requirements.txt`. Use Python
3.12 for the pinned NumPy 1.26.4 environment; this machine's default Python
3.13 is not the modeling environment. Shared data utilities remain standard-library-only.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r experiments/supervised/requirements.txt
.venv/bin/python -m human_ai.prepare_data
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest experiments.supervised.test_q1 -v
.venv/bin/python -m experiments.supervised.run_q1 --prefix a_q1_reproduction_001
.venv/bin/python -m experiments.supervised.verify_q1 --prefix a_q1_reproduction_001 --output results/supervised/a_q1_reproduction_001_verification.json --refit
```

This machine executed the scripts with `/opt/anaconda3/bin/python`. The original
prefix is `a_q1_20261107_v1`. Use **new** prefixes and outputs for reruns; evidence
is never overwritten. The original selected candidate has already been generated:

```sh
.venv/bin/python -m experiments.supervised.predict --run-id a_q1_20261107_v1_svm --refit-all --output submissions/a_q1_20261107_v1_svm_full10000.csv
.venv/bin/python -m human_ai.check_predictions submissions/a_q1_20261107_v1_svm_full10000.csv --kind submission
```

For a fresh reproduction, substitute its selected run ID and a new output path.
`--refit-all` fits selected settings to all 10,000 supplied labeled rows, without
a new search or holdout estimate. Omit it to predict from a saved 8,000-row model
to a new CSV. Models are ignored `model.joblib` files; regenerate after cloning.
Scripts never upload to Kaggle or consume submission slots.

## Controlled protocol and interpretation

The plan in `q1_plan.json` was fixed before measuring validation scores:

- LR: C in {0.1, 1, 10}; L2, liblinear, tolerance 1e-5, max_iter 2000.
- Multinomial NB: alpha in {0.01, 0.1, 1}; force_alpha=True.
- Linear SVM: C in {0.1, 1, 10}; L2, dual='auto', tolerance 1e-5, max_iter 10000.

All three approaches are tuned with identical three-fold stratified splits
(seed 142), wholly within the 8,000 training IDs. Vocabulary/IDF are refit on
each inner training fold. Within a fold, identical features are reused across
classifiers to save time. All 27 candidate-fold scores and nine CV summaries
are retained. LR's best C is at the grid boundary; no global optimum is claimed.
No adaptive grid expansion followed the held-out comparison.

SVM tuning uses decision-score AUROC; probability calibration follows C selection.
The **entire TF-IDF + SVM pipeline** is inside CalibratedClassifierCV, preventing
calibration-fold vocabulary/IDF leakage. Calibration uses training-only folds
(seed 242), sigmoid, ensemble=False: one final full-training base estimator,
with the sigmoid learned from out-of-fold margins. Model seed is 42.

The common 2,000-row validation set compares the three selected families. It
is therefore a model-selection set, not an untouched estimate after selection.
Paired bootstrap uses 1,000 class-stratified replicates and seed 342. Intervals
condition on fitted models/split, omit training variability and selection
adjustment, and do not establish hidden-domain or hidden-generator performance.
Near-duplicate/source leakage remains unassessed; the existing split audit only
excludes exact whitespace-normalized duplicates. No test/unlabeled/starter data
or leaderboard feedback is used for fitting or selection.

NB is worse here, which does not establish that all NB variants are poor. Its
independence assumptions and correlated unigram/bigram features are possible
explanations, not tested causes. LR/SVM's train AUROC 1.0 versus validation about
0.944 suggests a generalization gap. Character features were still an unmeasured
hypothesis at this Q1 stage; the separate Q5 study above now tests them while
holding classifier/settings fixed.

## Evidence and integration

- Study bundle: frozen source/plan, versions, Git commit and dirty state, split
  hashes, exact fitting IDs, inner-fold IDs, CV scores, comparison, bootstrap.
- `..._lr`, `..._nb`, `..._svm`: all estimator parameters, metrics, local model,
  and checked `id,label,p_ai` validation predictions.
- `..._svm_full10000`: separate training ID order/settings, training-only
  diagnostics, fit time and local model; **no held-out score**.
- `..._verification.json`: source/fold audits, CSV metric reconstruction,
  reload and fresh-refit checks. All validation probabilities matched exactly.
- Report writing is handled separately in Overleaf, using the recorded code/results.
- `report/figures/q1_*.png`: comparison and CV-grid figures, regenerated with
  `python -m experiments.supervised.plot_q1 --prefix a_q1_20261107_v1`.
- `docs/lin_q1_handoff.md`: team transfer and honest AI-use notes.

Alex can use `..._svm/validation_predictions.csv` for Q4 independently. If team
selection changes, update Q4 and Q1.4. Gabriel can use the commands/versions
for clean-environment reproduction; that cross-environment check is still pending.

