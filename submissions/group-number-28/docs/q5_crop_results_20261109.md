# Weak-crop execution: random screen passes, topic proxy invalid

Run: `a_q5_crop_20261109_v5_study`, executed November 9, 2026.
**Completed according to the declared stopping rule; no candidate exported.**
The public incumbent remains WC120 at the user-reported AUROC **0.87399**.
This is an engineering experiment record, not report prose.

## What ran

The [fixed protocol](next_round_crop_protocol_20261109.md) and
`experiments/supervised/q5_crop_plan.json` were snapshotted before fitting,
together with code, original split IDs, reference hashes and environment.
The original plan's `planned_not_executed` field is retained in that historical
snapshot; actual execution status is in `status.json` and `completion.json`.

The implementation checks passed: five boundary/weight/calibration tests using
exactly seven tiny synthetic SVC fits, six zero-fit metric/bootstrap checks,
and ten shared split/prediction contract tests. No additional real-data
reproduction fit was introduced.

Stage 0 learned the fixed, label-blind 12-cluster topic proxy from the 8,000
canonical fitting originals. Two independent reviewers each read the same
60 complete sampled documents and top-term lists without labels, candidate
scores or the other review. Reviews were frozen before class diagnostics.

The grouping failed two predeclared checks:

| Topic-quality check | Required | Observed |
| --- | --- | --- |
| Content-interpretable groups agreed by both reviewers | At least 9 of 12 | 8 content, 3 mixed, 1 style/format |
| Held fold size | 2,000–3,334 | 2,606 / **3,354** / 2,040 |

Other grouping checks passed, including cluster sizes, per-cluster class
mix and topic/label NMI (0.00078242 <=0.10). The oversize fold exceeded its
limit by 20 documents, but neither that limit nor the review requirement was
relaxed after observing the result. This is an invalid proxy under this
protocol, not proof that topic shift is absent.

As declared, the run used the fallback diagnostic random split:
`StratifiedKFold(3, shuffle=True, random_state=546)`. Its held sizes were
2,667 / 2,667 / 2,666, and every original ID was held once. This is **not**
the topic-matched random control, because no topic split qualified.

Exactly **six competition-data SVC fits** completed: baseline and weak crop
in each random fold. Each fold fitted WC120 vocabulary/IDF on its fitting
originals only, sharing that representation between the two arms. Long
originals had weight 0.9 and their first-100-word crops weight 0.1; other
originals had weight 1. Every original ID retained total weight 1.
All six SVC fits completed without warnings. Topic-disjoint classifier fits,
sealed suffix/middle evaluations, calibrated 8,000-row development evaluation,
full-10,000 refitting, test-text prediction and candidate export were skipped.

## Measured random-CV results

These values are equal-weight means of three **within-fold** AUROCs, using raw
decision margins. They are not pooled OOF AUROCs or scores on the canonical
2,000-row development set. Training size per fit was about 5,333 originals.

| Evaluation condition | Baseline | Crop weight 0.1 | Paired mean difference | Conditional paired 95% interval |
| --- | ---: | ---: | ---: | --- |
| Original text | 0.965186 | 0.962823 | -0.002364 | [-0.002988, -0.001738] |
| Quote + dash | 0.962423 | 0.959890 | -0.002533 | [-0.003208, -0.001883] |
| Digit runs to zero | 0.961795 | 0.960129 | -0.001666 | [-0.002240, -0.001122] |
| Whitespace only | 0.938952 | 0.943120 | +0.004168 | [+0.003140, +0.005196] |
| First 100 words, original >=150 | 0.855860 | 0.906588 | +0.050727 | [+0.046537, +0.055276] |

The prefix subset spans 6,420 held originals across the three folds. Each
interval uses 1,000 class-stratified paired original-ID bootstrap replicates
and averages within-fold differences. All replicates were valid. Intervals
condition on fitted models and splits, omit training/selection uncertainty and
do not establish hidden-domain performance or a family-wide significance claim.

Clean differences by fold were -0.001904 / -0.002817 / -0.002370.
Prefix differences were +0.051396 / +0.043639 / +0.057148.
Thus the diagnostic random screen passed all its point-estimate/fold gates:
clean mean loss <=0.003, no fold loss >0.006, prefix gain >=0.005 in all three
folds, and the other fixed conditions within their specified loss limits.
The random intervals are descriptive; they do not substitute for the skipped
calibrated-development or topic-evidence gates.

## What the result supports

The weak augmentation improves the targeted prefix condition consistently while
paying a small clean-discrimination cost in these random folds. It passes the
random screen; it does **not** pass the complete promotion protocol because the
topic proxy was invalid. No calibrated full-8,000 model or new canonical
validation score was produced.

The earlier crop-weight-0.5 pilot used a different fitting boundary and held
population. Do not directly compare its effect sizes with this three-fold run
as if crop weight were their only difference. Likewise, the current CV values
cannot be compared directly with a full-10,000 submission's public AUROC.

An additional observed sensitivity is worth preserving: keeping every word in
order and changing only whitespace lowered the baseline mean from 0.965186 to
0.938952, a difference of **-0.026234**. Crop training recovered 0.004168 under
that condition but did not remove the sensitivity. This shows format sensitivity
without truncating content. It does not establish that whitespace caused the
public gap; the measured hidden-test format mixture and labels remain unknown.

Independent checks found identical token sequences and exactly identical word
features in every held fold. Character-feature rows changed for
1,129 / 1,125 / 1,150 originals (3,404 of 8,000 total). The installed character
analyzer compresses runs of repeated whitespace, but isolated newline/tab
characters can remain until this explicit whitespace transformation. The
observed change is therefore not a word-tokenization or condition-alignment bug.

A future, separately specified hypothesis could compare original versus
full-whitespace-canonicalized training/prediction with the same WC120 model.
That would isolate a more targeted formatting intervention than cropping.
It was not added to this run, and no grouping seed, crop weight or quality gate
was retried to obtain an export.

## Artifacts and reproduction

The run directory contains `plan.json`, `provenance.json`, exact `source/`
snapshots, blind review packet/reviews, `topic_quality.json`, fallback folds,
per-fold feature/classifier receipts and ID-aligned predictions,
`random_per_fold_metrics.csv`, `random_gate.json`, `random_bootstrap.json`,
`fit_receipts.json` and `completion.json`. Model files are ignored by Git.
Raw passages are not included in the new tracked outputs.

The implementation is in `crop_model.py`, `crop_topics.py`,
`crop_evaluation.py` and `run_q5_crop.py`. Execution used the pinned historical
Python environment (`/opt/anaconda3/bin/python`). For a **new** reproduction
prefix, first prepare the topic packet:

```sh
python -m experiments.supervised.run_q5_crop --action prepare --prefix NEW_PREFIX
```

Obtain two independent blinded JSON reviews using the generated schema, then:

```sh
python -m experiments.supervised.run_q5_crop --action run --resume --prefix NEW_PREFIX --review-a REVIEW_A.json --review-b REVIEW_B.json
```

These commands require the original CSVs and the local reference WC120 model
whose exact hash is pinned in the plan. The model is intentionally ignored by
Git; a clone alone cannot run the checkpoint-dependent entrypoint or full
model-reloading verification. Reproduce the parent studies or obtain the
original local checkpoints and verify their hashes. The recorded study uses
its byte-exact executing source snapshot; the current runner also supports
the later adaptive revision. Fresh-clone model-level reproduction was not
performed during this work.

Do not rerun to search for favorable reviews or groupings. The existing run is
complete and should be inspected through its saved artifacts and independent
verification receipt rather than refitted. No Kaggle upload or new public
score belongs to this experiment.

Independent zero-fit verification passed and is recorded in
`results/supervised/a_q5_crop_20261109_v5_verification.json`. It reconstructs
topic checks/folds, original-only fold IDF, predictions from reloaded models,
all saved AUROCs/gates and the 1,000 paired bootstrap draws. The additional
`a_q5_crop_20261109_v5_whitespace_checks.json` preserves the whitespace
diagnostic above. Neither verifier fitted a model or accessed test text.

New shared knowledge: a fixed 0.1 crop intervention passes the random tradeoff
screen; the chosen topic proxy does not meet the planned evidence standard;
whitespace-only sensitivity is substantial. Cross-topic transfer, calibrated
development eligibility and hidden-test gain remain unestablished. WC120 stays
the incumbent.
