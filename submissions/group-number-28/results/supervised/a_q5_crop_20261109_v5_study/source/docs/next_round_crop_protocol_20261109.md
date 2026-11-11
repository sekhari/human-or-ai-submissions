# Next-round protocol: one weak crop intervention

Status: **planned, not executed**. This document and
`experiments/supervised/q5_crop_plan.json` specify the next run. No clustering,
classifier training, candidate export or upload is part of this planning task.
The suggested run prefix is `a_q5_crop_20261109_v5`; a runtime must snapshot the
plan, source/input hashes and actual environment before any fitting begins.
The machine-readable plan also pins expected hashes of the reference model,
reference parameters, incumbent CSV, pilot plan and canonical split files;
verify these rather than silently accepting a changed reference.
The runner and model wrapper listed below are planned implementation work,
not commands that already exist.

## 1. Objective and interpretation

Find out whether a small amount of prefix-crop training improves the robustness
tradeoff while retaining original-text discrimination and surviving a lexical
topic shift. Keep WC120, public AUROC **0.87399**, as the incumbent.

The prior 0.5-weight pilot improved prefix AUROC by 0.059952 but lost 0.005340 on
clean development text. The next hypothesis is that **crop weight 0.1** can keep
some of that robustness benefit with a clean loss no worse than 0.003.
This is a hypothesis, not a linear interpolation or an expected result.

There is an important limit: test passages average more words than training
passages, so prefix cropping is not an established model of the hidden test
distribution. Passing the protocol would justify one exploratory robustness
candidate, not a claim that the public gap has been explained or repaired.
The canonical 2,000-row validation set has already been reused; it is development
evidence, not a newly untouched test. See the preceding
[diagnosis and pilot](public_score_research_20261109.md).

## 2. Exactly two experimental arms

| Item | B0: original training | B1: weak crop training |
| --- | --- | --- |
| Original >=150 whitespace words | weight 1.0 | original weight 0.9 + first-100-word crop weight 0.1 |
| Original <150 words | weight 1.0 | weight 1.0; no crop |
| Total weight per original ID | 1.0 | 1.0 |
| Predictive representation | Original WC120 | Identical fold-specific WC120 |
| SVM | Original C=1 and seed 42 | Same |
| Vocabulary and IDF input | Fitting-fold originals only | Same originals; crops do not fit the representation |

Cropping uses `text.split()` and joins the first 100 tokens with one ASCII
space. It changes available content, length and whitespace. Its authorship-label
preservation is a reasonable working assumption, not a pure length manipulation.

Keep word (1,2) and character (3,5) TF-IDF, at most 60,000 features each, existing
normalization, classifier settings and existing reference dependencies. The exact
representation and classifier parameters come from `q5_plan.json`; record its
hash and effective parameters at runtime. Do not change C, feature count,
tokenization, crop length/position, seeds, quote preprocessing or numeric features.
Weight 0.5 remains historical pilot evidence; it is not a third candidate.

In the raw-margin CV stages, fit one representation on each fitting fold's
originals and share it between B0 and B1. The saved transformer trained on all
8,000 rows must not be used in these folds. This differs deliberately from the
earlier fixed-feature pilot and prevents the held fold from influencing IDF.

## 3. Data use and the topic proxy

Keep the committed 8,000/2,000 split unchanged. Only the 8,000 original fitting
IDs participate in split construction, screening CV or development calibration. No unlabeled
learning or test-text fitting is included in this round. Do not access test text
until a model has passed every export gate.

Build a separate topic representation once, using the 8,000 originals:

1. Word-unigram TF-IDF: lowercase Unicode letter-only words of at least three
   letters, English stopwords, `min_df=10`, `max_df=0.6`, maximum 20,000 features,
   sublinear TF and L2 normalization.
2. Truncated SVD: 64 dimensions, seven iterations, seed 542; L2 normalize output.
3. KMeans: 12 clusters, 10 initializations, maximum 300 iterations, tolerance
   0.0001, Lloyd algorithm, seed 543.

This representation exists only to design a split. It may inspect the 8,000
texts jointly for that purpose; neither its vocabulary nor its SVD features
are reused in a predictive pipeline. Record this text-dependent split design
explicitly. Do not use labels to choose grouping features, K or seeds.

Before any candidate scores are calculated, freeze cluster assignments and
perform these checks:

- Each group has 100–2,400 documents.
- Two independent reviewers, blind to labels and model scores, inspect each
  group's 12 top unigram terms and five deterministically sampled local
  documents (seed 547). Record sampled IDs and a coarse content topic, mixed,
  or style/format judgment; save no passages. A disagreement counts as mixed.
  At least 9/12 groups must be content-interpretable, and at most one group
  may primarily reflect style/format.
- After assignments/reviews are frozen, reveal labels for diagnostics:
  topic/label normalized mutual information must be <=0.10, and every group's
  majority-class fraction must be <0.90.

These are conservative engineering gates, not standard statistical truths.
Genuine topics can correlate with the label. A failed gate means this particular
proxy does not support the intended interpretation; it is not evidence that
topic shift does not exist. Do not remove groups or retry clustering seeds.

Use `StratifiedGroupKFold(3, shuffle=True, random_state=544)` to keep each topic
wholly in one held fold. Each held fold must contain 2,000–3,334 originals,
35%–65% AI and at least 500 originals per class. The splitter attempts label
balance under nonoverlapping groups; it does not guarantee these checks, so
verify them explicitly. See the
[scikit-learn grouped splitter documentation](https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).

Construct **one matched random three-fold partition** by class-wise seeded
permutation (seed 545), allocating IDs using the topic folds' exact human/AI
counts. Its training/held sizes and class counts must match the corresponding
topic folds exactly. Every ID is held once in each scheme. Do not additionally
match topics or document length; describe those differences.

This matched partition is the primary random CV. An extra ordinary random
three-fold run would add six classifier fits without resolving a new question.
If grouping or fold-quality checks fail, run only a diagnostic stratified
three-fold partition with seed 546, report the grouping failure, and stop
promotion. Random diagnostic success alone does not authorize an export under
this protocol. No new grouping is selected to rescue the run.

## 4. Fixed evaluation matrix

| Condition | Definition | Population |
| --- | --- | --- |
| Clean | Original passage | All held originals |
| Quote + dash | Existing quote map followed by existing dash map | Same all-held IDs |
| Digits | Unicode decimal-digit runs replaced with `0` | Same all-held IDs |
| Whitespace only | Join all `text.split()` tokens with one ASCII space | Same all-held IDs |
| Prefix 100 | First 100 whitespace words | Original length >=150 only |
| Suffix 100 | Last 100 whitespace words | Same IDs as prefix |
| Middle 100 | Start at `floor((n_words-100)/2)`, take 100 words | Same IDs as prefix |

Score changes are paired by original ID and model arm. For any before/after
comparison restricted to crop-eligible documents, use exactly those same IDs
for both versions. Require at least 50 originals per class in each condition
and held fold; otherwise mark evidence insufficient.

Suffix/middle are new evaluation conditions, never training augmentations.
Freeze their definitions now, and reveal their scores from the cached fold
models only after both random and topic primary gates pass. They test transfer
beyond the trained prefix position. They are not independent datasets or a
fresh source holdout; short passages' windows can overlap. Whitespace-only
helps measure part of the format change without removing content, but still
does not separate pure length effects from content coverage.

## 5. Sequence, promotion gates and fit budget

All differences below mean **B1 minus B0 AUROC**. Finish all three folds of a
started stage before applying its aggregate gates. Do not stop after seeing
an unfavorable fold or retain only favorable conditions.

| Stage | Work | SVC fits | Next-stage requirement |
| --- | --- | ---: | --- |
| 0 | Plan snapshot, split hashes, topic proxy/review and matched assignments | 0 | Topic validity determines whether promotion is possible |
| 1 | Matched random three-fold CV, two arms, raw margins | 6 | Random gates pass and topic proxy is valid |
| 2 | Topic-disjoint three-fold CV, two arms, raw margins; then reveal suffix/middle for both schemes | 6 | Topic and new-condition gates pass |
| 3 | Two fully calibrated 8,000-label models; evaluate canonical 2,000-row development set once | 8 | Every calibrated gate passes |
| 4 | One eligible B1 refit with all 10,000 labels; check/export one file | 4 | Artifact checks pass; upload is separate |
| Maximum | Including conditional full-data export training | **24** | No extra candidates |

Topic clustering and sigmoid optimization are separate from the SVC count.
There are at most six shared raw-CV predictive representation fits, eight
calibrated-development representation fits and four final representation fits:
**18 predictive representation fits**, plus split-construction components.
Binary calibration fits at most three sigmoid calibrators. Stress prediction
and bootstrap do not add model fits.
The 24-fit limit counts competition-data SVC fits. Permit at most eight additional
tiny synthetic regression-test SVC fits, each with at most 30 original documents.
Do not add an unbudgeted real-data baseline fit before the stages: the complete
real-data reproduction check belongs to the eight fits in stage 3.

**Random-CV screening:**

- Clean: mean difference >=-0.003; no fold <-0.006.
- Prefix: mean difference >=+0.005; all three folds strictly positive.
- Quote/dash, digits and whitespace: each condition's mean >=-0.003;
  no condition/fold <-0.006.

**Topic-CV screening:**

- Clean: mean difference >=-0.003; no fold <-0.006.
- Prefix: mean difference >=+0.005; at least two folds strictly positive;
  no fold <-0.003.
- Quote/dash, digits and whitespace: each condition's mean >=-0.003;
  no condition/fold <-0.006.

**New-condition check, after both screens pass:** suffix and middle must each
have mean difference >=-0.003 in each split scheme, with no fold <-0.006.
This is a non-regression floor. Passing it alone does not demonstrate an
improvement on suffix or middle text.

If a gate fails or a required AUROC is undefined, save the complete current
stage and stop later stages. Keep WC120. Do not append crop weights 0.05/0.2,
new seeds, ensembles or another model family. A mechanically failed run may
resume only its identical incomplete work; a new scientific intervention
requires a separately documented next question.

## 6. Uncertainty and reporting

The primary CV summary is the equal-weight mean of three **within-fold** AUROC
differences, plus the three separate AUCs/differences and their range. Do not
pool raw margins from different fitted folds to compute a primary OOF AUROC:
their score scales need not be comparable. Preserve fold-tagged margin files
with an explicit non-probability schema.

Use 1,000 paired bootstrap replicates for descriptive conditional intervals:

- Random CV: class-stratified original-ID resampling within each held fold,
  reusing each draw across arms and variants; recompute the macro fold
  difference. Crop conditions use eligible IDs in that same draw.
- Topic CV: resample whole topics with replacement within each fixed held
  fold, including all their documents and variants. Report valid replicate
  count and skip single-class replicates without resampling until a desired
  interval is obtained. Fewer than 90% valid replicates means its interval
  is unavailable. With only 12 topics, treat this as sensitivity analysis.
- Canonical development set: class-stratified paired original-ID resampling
  with shared draws across conditions; calculate the clean and prefix
  paired percentile intervals used below.

For every scheme, skip undefined/single-class replicates, report the valid count
and never keep drawing to obtain a desired number of valid samples. Below 90%
valid replicates, its interval is unavailable. A gate requiring an unavailable
interval becomes insufficient evidence; a descriptive-only interval does not
itself decide promotion.

The intervals condition on trained models and chosen splits. They omit training
variation, prior development selection and multiple-comparison adjustment.
The three training folds overlap, so they are not three independent experiments
for an ordinary t-test. Report per-topic counts/AUCs where both classes exist;
single-class or small topics do not provide precise subgroup evidence. The
protocol uses practical decision gates, not a blanket significance claim.

## 7. Calibrated development and export gate

Only after both CV schemes and the new conditions pass, train B0 and B1 on all
8,000 fitting originals using identical three-fold sigmoid calibration:
`ensemble=False`, original-ID `StratifiedKFold` with seed 242, one job.
The calibrator receives original documents only. A custom base estimator creates
crops **inside its `fit()`**, after calibration has selected fitting originals:

1. Fit vocabulary/IDF on that fitting boundary's originals.
2. Create eligible crops and assign 0.9/0.1 weights internally.
3. Fit the SVM; predict margins only for the untouched calibration originals.
4. Learn one sigmoid from OOF original-document margins, then refit the base
   estimator on all permitted originals.

Do not expand the dataset before ordinary calibration CV: that would allow an
original and its crop to cross the fitting/calibration boundary. With
`ensemble=False`, three calibration fits plus a final full fit give four base
fits per arm. This follows the
[documented calibration procedure](https://scikit-learn.org/1.5/modules/generated/sklearn.calibration.CalibratedClassifierCV.html).

First require B0 to reproduce saved WC120 development probabilities with
maximum absolute error <=1e-10. A failure stops evaluation/promotion as an
implementation discrepancy; do not treat a changed baseline as an improvement.

On the reused 2,000-row development set, require all of:

- Clean point difference >=-0.003 **and paired 95% interval lower bound
  >=-0.003**.
- Prefix point gain >=+0.005 **and paired 95% interval lower bound >0**.
- Each other fixed condition's point difference >=-0.003.
- Clean performance in each of three original-length bins loses no more than
  0.010. Boundaries are word-count tertiles from the 8,000 fitting originals,
  using `digitize(right=True)`; require 30 per class per bin, and do not merge
  inconvenient bins after looking at outcomes.
- No convergence warning, data-boundary discrepancy or missing required result.

These interval checks are stronger than simply seeing a small negative clean
point estimate. They may reject a seemingly useful tradeoff; that is intentional
given the previous local/public ranking reversal. Report log loss, Brier score,
score correlations, feature counts and runtime as secondary diagnostics, not
additional knobs to optimize.

Passing permits at most one full-10,000-label B1 refit with the identical recipe
and calibration boundaries. It has **no local held-out score**. Export metadata
must label it `exploratory_robustness_tradeoff`, preserve WC120, and verify exact
3,000 test-ID coverage, class-1 probability direction, finite [0,1] values,
model-reload agreement and source/data/model/CSV hashes. Obtain an independent
artifact check. Actual upload is separate from this plan: check current team
slots/reset information at that time, without assuming the old Sunday allocation
still applies. Never tune crop weight to the resulting public score.

## 8. Implementation tasks and reviewable outputs

Planned code, not yet implemented:

- `crop_model.py`: cloneable, serializable base estimator with augmentation
  internal to `fit()`, exposing class order and margins/probability interface
  required by calibration.
- `run_q5_crop.py`: immutable plan snapshot, topic review receipts, matched
  folds, cached fold models, sequential stage gates and conditional export.
- `verify_q5_crop.py`: independently reconstruct fold counts, topic separation,
  weight conservation, AUCs/gates, source hashes and eligible export checks.
- `test_q5_crop.py`: meaningful regression checks for leakage boundaries, per-ID
  weights, baseline reproduction, original/crop calibration separation, matched
  splits, deterministic transformations and undefined-AUC handling.

Before competition-data fitting, validate those boundaries with small synthetic
sentinel documents, direct-pipeline/wrapper baseline equivalence and the existing
split/prediction contract tests. The full saved-model reproduction check waits
until the already budgeted stage 3. Do not
invent performance assertions or a success score. Expected artifacts include:

```text
results/supervised/<prefix>_study/
  plan.json, provenance.json, split_manifest.json
  topic_assignments.csv, topic_review.json, topic_quality.json
  random_folds.csv, topic_folds.csv, fold_counts.csv
  per_fold_metrics.csv, fold_margins_not_probabilities.csv
  bootstrap.json, gate_receipt.json, status.json
  source/  (exact executing code snapshots)
results/supervised/<prefix>_baseline_8000/       # only if stage 3 starts
results/supervised/<prefix>_weak_crop_8000/      # only if stage 3 starts
  validation_predictions.csv                   # id,label,p_ai, all 2,000 IDs
  condition_predictions.csv, metrics.json, config.json
results/supervised/<prefix>_weak_crop_full10000/ # only if eligible
submissions/<prefix>_weak_crop_full10000.csv    # at most one
submissions/<prefix>_weak_crop_full10000.metadata.json
```

All schemas identify original IDs and fold/condition/arm as appropriate. Never
write raw passages into new tracked outputs. Serialized models remain ignored
by Git. Snapshot the current dirty-tree state rather than deleting unrelated
work. The existing audits, pilot and incumbent submission remain immutable.

Run sequentially with one CPU thread and the recorded requirements; no new
dependencies, external model/API calls or GPU are needed. The preliminary
wall-time estimate is **15–45 minutes** for a fully eligible run, excluding
implementation/review, and is not a measured promise. Update the estimate after
the first fold using recorded feature-fitting, prediction and bootstrap times;
save resumable per-fold receipts rather than changing the protocol to meet an
estimate.

## 9. Decisions this plan makes

One candidate, one crop weight, two matched validation schemes, seven fixed
conditions, explicit clean-loss limits, at most 24 SVC fits and one eligible
submission file. NB weighting, richness variants, pseudo-labels, transformers
and ensembling remain outside this round so their effects cannot be mixed into
the crop question.

Still unknown: whether the 0.1 intervention meets these gates; whether the
label-free clusters form a useful topic proxy; whether any measured gain
transfers to unseen generators/adversarial edits or the private leaderboard.
Each has a recorded outcome path, including a valid negative result.
