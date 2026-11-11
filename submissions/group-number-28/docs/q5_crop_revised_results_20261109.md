# Revised weak-crop experiment: measured topic transfer and clean cost

Run: `a_q5_crop_revised_20261109_v6_study`, November 9, 2026.
This is an **adaptive continuation** authorized after the previous stopping
diagnostics and fallback-random results were observed. It is not a pristine
confirmatory experiment. The original study, plan, topic assignments and reviews
remain unchanged.

## Authorized revision and actual execution

The [revised protocol](q5_crop_revised_protocol_20261109.md) changed exactly two
checks from promotion barriers to reported diagnostics:

- At least nine content-interpretable groups: observed eight are retained.
- Held-fold maximum 3,334: actual fold sizes 2,606 / 3,354 / 2,040 are retained.

No clustering refit, seed search, topic rejudgment or group deletion occurred.
The same 12 groups and topic-fold IDs were reused with pinned parent hashes.
The original remaining structural/class-support checks passed. Class-wise
permutation with seed545 created new matched random folds whose held and
fitting human/AI counts exactly match every corresponding topic fold.

Both arms stayed fixed: WC120/C=1, original training versus eligible original
weight0.9 plus prefix100 weight0.1, with total weight1 per original ID. Each
fold learned vocabulary/IDF only from its fitting originals; the two arms shared
that fold-specific representation. The old fallback split and its models were
not reused as the new matched control.

Exactly **12 new competition-data SVC fits** completed, six per split scheme,
without convergence warnings. No new synthetic SVC fit was performed; the
unchanged estimator/metric implementation checks were inherited with source
hash checks. Primary scoring and 1,000-replicate paired bootstrap were completed
for both schemes. Secondary group analysis reused the identical held topic-model
predictions and introduced no additional fits.

## Primary results

Each entry is the equal-weight mean of three **within-fold** AUROCs, not an AUC
computed from pooled cross-fold margins. Training sizes and class counts are
exactly matched between corresponding random/topic folds.

| Split | Condition | Baseline | Weak crop | Paired mean difference |
| --- | --- | ---: | ---: | ---: |
| Matched random | Original text | 0.965848 | 0.963368 | -0.002480 |
| Topic-disjoint | Original text | 0.932745 | 0.929526 | -0.003219 |
| Matched random | Quote + dash | 0.962794 | 0.960073 | -0.002721 |
| Topic-disjoint | Quote + dash | 0.931372 | 0.928046 | -0.003326 |
| Matched random | Digit runs to zero | 0.962375 | 0.960654 | -0.001720 |
| Topic-disjoint | Digit runs to zero | 0.927329 | 0.924694 | -0.002635 |
| Matched random | Whitespace only | 0.939918 | 0.943805 | +0.003887 |
| Topic-disjoint | Whitespace only | 0.901362 | 0.904371 | +0.003009 |
| Matched random | Prefix100 | 0.857019 | 0.907943 | +0.050924 |
| Topic-disjoint | Prefix100 | 0.816874 | 0.854561 | +0.037687 |

Each prefix comparison keeps exactly the same eligible original IDs in both
arms. All three topic-fold prefix gains are positive:
**+0.007842 / +0.041328 / +0.063890**. Clean differences are negative in all
three topic folds: **-0.001968 / -0.003802 / -0.003887**.

The baseline's original-text mean is 0.033103 lower under the topic split than
under its count-matched random control. This demonstrates that this particular
lexical-topic/style partition creates a harder evaluation. It does not identify
the competition's hidden domains, generators or adversarial edits.

## Uncertainty and the remaining performance decision

The matched-random screen passed all original point/fold performance checks.
The topic screen passed prefix, digit, whitespace and per-fold loss checks;
two mean-loss checks exceeded the unchanged practical budget:

| Topic condition | Observed mean loss | Budget | Excess above budget |
| --- | ---: | ---: | ---: |
| Original text | 0.003219 | 0.003000 | **0.000219** |
| Quote + dash | 0.003326 | 0.003000 | **0.000326** |

These are **near-boundary operational tradeoffs**, not proof that the method is
scientifically invalid or that its loss is decisively worse than 0.003.
Whole-topic paired conditional 95% bootstrap intervals are:

- Clean difference: **[-0.004383, -0.000828]**.
- Quote/dash difference: **[-0.004488, -0.000913]**.
- Prefix gain: **[+0.006565, +0.043198]**.

The clean and quote intervals span the -0.003 decision boundary. The measured
clean cost and prefix benefit are consistent across the three topic folds, but
the magnitude of the cost relative to that budget remains uncertain. Only 12
proxy groups exist; intervals condition on current models/splits and omit
training variation and adaptive-selection correction. All 1,000 replicates
were valid. Do not convert this result into a hidden-test or private-score claim.

The run honored the **remaining** unchanged performance stopping rule and did
not open suffix/middle conditions or proceed to calibrated development/full-data
training. No canonical-2,000 prediction, test-text prediction, candidate export
or upload was produced. The two revised grouping diagnostics no longer caused
the stop. WC120 remains the user-reported public incumbent at 0.87399.

## Content versus mixed/style sensitivity

These fixed groups come from the original combined blind judgments. Every
model still trains on all permitted fitting groups; this analysis only filters
its existing held predictions. It is secondary and does not decide promotion.

| Held subgroup | Originals | Clean mean difference | Prefix mean difference |
| --- | ---: | ---: | ---: |
| Content-interpretable topics | 4,741 | -0.003777 | +0.040148 |
| Mixed/style topics | 3,259 | -0.000276 | +0.006035 |

Both subgroups have defined AUCs and adequate class support in all three folds.
The original-text cost is also present within content-interpretable topics, so
it cannot be dismissed simply by excluding mixed/style judgments from the
interpretation. Prefix robustness improves within content topics as well.
These subgroup AUC differences are not additive: overall AUC also contains
cross-subgroup comparisons, so weighting subgroup differences by their sizes
does not reconstruct the overall difference. Neither subgroup comparison is a
causal estimate of topic membership.

## Artifacts and current conclusion

The run directory preserves the selected `plan.json`, exact `source/` snapshots,
`topic_reuse_receipt.json`, unchanged inherited grouping/review files, new
`random_folds.csv`, `fold_counts.csv`, raw fold feature/classifier receipts and
ID-aligned margins, random/topic metrics and gates, paired bootstrap results,
`topic_sensitivity.json`, `fit_receipts.json` and `completion.json`.
The machine-readable source plan is `q5_crop_revised_plan.json`.
No passages are stored in the new tracked artifacts; model binaries remain
ignored by Git.

Independent zero-fit verification passed:
`results/supervised/a_q5_crop_revised_20261109_v6_verification.json`.
It checked immutable parent artifacts, exact group/fold reuse and matched class
counts, all 12 fit receipts, original-only fold IDF, reloaded-model margins,
AUROCs/gates, both 1,000-replicate bootstrap runs and all 30 subgroup rows with
their descriptive summaries. The verifier read no test text and performed no
model fitting. Its executing source hash is included in the receipt.

This continuation establishes a more useful finding than the old categorical
topic-quality stop: the fixed augmentation's prefix benefit survives this topic
proxy, with a small but consistent clean loss. It supports an explicit tradeoff,
not a verified overall replacement for WC120. The old 0.003 loss budget should
be understood as an operating preference, not a natural boundary of model
validity. No further threshold or weight was changed after these new results.

## Execution command and reproduction prerequisites

The actual revised run used the pinned modeling environment:

```sh
/opt/anaconda3/bin/python -u -m experiments.supervised.run_q5_crop --plan experiments/supervised/q5_crop_revised_plan.json --prefix a_q5_crop_revised_20261109_v6 --action run
```

Use a new prefix for a separate reproduction; the recorded directory is
immutable. This runner also needs the supplied raw CSVs and the local reference
checkpoints named and hashed in the plan, including the original WC120
`model.joblib` and the parent study's `topic_proxy.joblib`. Model binaries are
intentionally ignored by Git. Obtain the original local checkpoints or reproduce
their parent studies under the recorded environment, then verify the required
hashes; do not bypass a mismatch. A Git clone by itself is not sufficient for
the checkpoint-dependent commands or the full model-reloading verifier.

The committed numeric CSVs, configurations, source snapshots and receipts
preserve the observed evidence without passages or model binaries. This upload
does not claim a fresh-clone, model-level reproduction has been completed.
