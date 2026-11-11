# Adaptive continuation of the fixed weak-crop experiment

The user authorized revising the two challenged engineering thresholds and
continuing execution on November 9, 2026. This is an **adaptive continuation**
after the earlier quality checks and fallback-random scores were observed.
It must not be presented as an untouched confirmatory experiment.

New run: `a_q5_crop_revised_20261109_v6_study`.
Machine-readable protocol: `experiments/supervised/q5_crop_revised_plan.json`.
The complete earlier study and its frozen plan/source remain unchanged.

## Changes and preserved comparisons

Exactly two grouping constraints become reported diagnostics:

- At least nine content-interpretable groups: the observed 8 content, 3 mixed
  and 1 style/format groups are retained and described.
- A held-fold maximum of 3,334: retain the exact original held sizes
  2,606 / 3,354 / 2,040 and report their imbalance.

Do not recluster, change seeds, re-review topics, remove groups or draw a more
convenient topic split. Reuse the prior assignments, original-ID fold memberships
and historical blinded judgments byte-for-byte, with pinned parent hashes.
The remaining original structural/class-support checks continue to apply:
topic separation, canonical ID coverage, nonempty train/held folds, class
support/balance, cluster sizes, class-proxy diagnostics and style-group count.

The new random control uses seed545, assigning original IDs independently
within class to match each actual topic fold's exact human/AI counts. Therefore
training size and label composition match within every random/topic fold pair.
The old seed546 fallback split is historical evidence, not the new matched
control. Both new schemes require six SVC fits each.

Keep both model arms unchanged: original WC120 versus original weight0.9 plus
eligible prefix100 weight0.1, each original ID retaining total weight1.
Fit vocabulary/IDF using fitting originals only. All original random/topic,
sealed-condition and calibrated-development performance gates remain unchanged.
The maximum new competition-data fit budget remains24, conditional on stage
gates:6 random +6 topic +8 calibrated development +4 eligible full-data refit.
No grouping refit or new synthetic SVC fit is needed; unchanged implementation
checks are explicitly inherited from the prior run.

## Added interpretation checks, without extra model fits

Using the very same held topic-model predictions, report the fixed content
topics separately from mixed/style topics. Retain all fitting groups in every
model. Record class counts, subgroup AUCs and paired deltas by fold/condition;
empty or single-class populations have undefined AUC. Any descriptive macro
reports its number of contributing folds, and raw margins are never pooled
across fitted folds. This secondary analysis cannot rescue or veto a primary
performance gate and cannot establish real domain/generator transfer.

Record calibration OOF original-ID margins if that stage is reached, without
adding fits, so the training-only calibration boundary can be checked.
Only after all unchanged gates pass may one full-label exploratory candidate
be exported. There is no automatic Kaggle upload. WC120 remains the public
incumbent until separately supported selection.

Snapshot this revision and executing sources before new fits. Preserve a
separate revision/reuse receipt, complete stage metrics, stopped-stage reason
if applicable, and an independent zero-fit verification receipt. Report which
new outcomes were unknown when this revision was frozen.
