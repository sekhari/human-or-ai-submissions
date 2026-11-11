# Public-score diagnosis and next experiments — November 9, 2026

This is an engineering research note, not report prose or a new competition
result. The current incumbent remains **WC120, public AUROC 0.87399**.
The immediate issue is selection under distribution shift: higher scores on
the reused random development split have not consistently transferred to the
public set. The evidence does not identify a particular hidden attack.

## Recorded outcomes

The current user-supplied screenshot contains four distinct candidates and a
repeat WC120 upload with the same score. Scores were transcribed into
`submissions/leaderboard_log.csv`; upload timestamps and submission IDs remain
unknown. The screenshot hash and source-artifact hashes are in
`results/supervised/a_public_gap_20261109_v1_review/evidence.json`.

| Candidate | Source 8,000-label development AUROC | Public AUROC | Public change from WC120 |
| --- | ---: | ---: | ---: |
| Expanded word SVM | 0.943876 | 0.85157 | -0.02242 |
| WC120 | 0.970006 | **0.87399** | reference |
| Quote-normalized WC120 | 0.968675 | 0.87225 | -0.00174 |
| C60 + richness | **0.974508** | 0.85812 | -0.01587 |

Development scores belong to 8,000-label models; public scores belong to separate
10,000-label refits. Their difference is not a controlled estimate of domain
shift. The repeated upload provides no independent evidence. The supplied
`old_rubric.tex`, lines 97–98, describes unseen domains, unseen generators and
adversarially modified AI text, with 900 public / 2,100 private examples. This is
the local handout's description, not an independently verified current rule.
No public labels or paired public confidence intervals are available; a small
score difference should not be described as statistically significant.

The quote candidate already lost original-text development AUROC, so its public
direction is consistent with its documented tradeoff. The richness reversal is
more informative, but comparing it with WC120 changes both the richness block
and the presence of word features. Its entire public loss cannot be attributed
to richness without a C60 public control.

## Audits actually executed

All diagnostics retain the canonical split. No test text was used to fit a
competition model, no hidden labels were retrieved, and no submission was
created or uploaded during these audits.

**Submission and data integrity.** All five local candidate CSVs pass the shared
ID/schema/probability checker, with 3,000 predictions each. Original and expanded
word-SVM submissions are numerically identical. The four saved 8,000-row models
used in the stress audit reproduce their saved validation probabilities exactly
(maximum absolute error zero). Inspection found no obvious vocabulary/IDF/scaler
leakage in the calibrated pipelines; this is an audit conclusion, not a proof
against every possible implementation issue.

**Strong lexical overlap.** Across all 16,000 supplied rows, there are zero
normalized exact duplicate pairs and zero token-5-shingle Jaccard >= 0.8 pairs,
including train–validation and labeled–test boundaries. The exact prefix-filter
algorithm was checked against exhaustive synthetic pairs. This excludes strong
contiguous lexical duplication at this threshold; it does not exclude shared
templates, paraphrases, topics or generators. There is no duplicate-derived
reason to change the canonical split or invent source groups.

**Observable train/test shift.** These are unlabeled marginal diagnostics, not
evidence about which hidden examples are misclassified.

| Feature | Human training mean | AI training mean | Test mean | Train/test KS statistic |
| --- | ---: | ---: | ---: | ---: |
| Digit fraction | 0.012087 | 0.013237 | 0.007413 | 0.22007 |
| Newline fraction | 0.004435 | 0.009046 | 0.009911 | 0.14227 |
| MATTR-50 | 0.791133 | 0.794553 | 0.780615 | 0.09083 |
| Normalized token entropy | 0.841800 | 0.843726 | 0.830922 | 0.09490 |

Several test means lie outside both empirical class means. Exact population
label-prior shift alone would produce a convex combination of class means;
sampling uncertainty and dependence prevent treating empirical deviations as a
formal causal test. The low-digit bin occupies 23.4% of validation and 46.8% of
test, yet richness still helps within that validation bin. Thus these marginal
shifts alone do not reproduce the public failure.

The supplied unlabeled data is closer to labeled data on these measured
features: digit KS is 0.0264 and newline KS 0.0167. Learning from it may help,
but it is not established to represent the unseen test conditions.

**Paired stress evaluation.** Quote/dash canonicalization and replacing each
Unicode decimal-digit run with `0` are separate conditions on all 2,000
validation IDs. A third condition keeps the first 100 whitespace-delimited words
of the same 1,627 validation texts originally containing at least 150 words
(857 human, 770 AI). Each cropped score is compared with the original score on
exactly those IDs.

| Saved model | Original on 1,627 IDs | First 100 words | Paired change |
| --- | ---: | ---: | ---: |
| WC120 | 0.972229 | 0.876228 | -0.096001 |
| C60 | 0.971166 | 0.855590 | -0.115577 |
| C60 + richness | 0.977057 | 0.874123 | -0.102934 |
| Quote WC120 | 0.971407 | 0.873902 | -0.097506 |

WC120's conditional paired 95% interval for the change is
[-0.111115, -0.082764], based on 500 class-stratified bootstrap replicates.
Quote+dash and digit masking cost WC120 only 0.002818 and 0.003486,
respectively. All conditions and hashes were saved before stress predictions.
The confidence intervals omit model training variation and selection adjustment.

The crop changes available content, length and whitespace together. It is not
a pure length experiment or a recreation of a known Kaggle attack. In fact,
test texts average **216.97 words**, versus **208.71** in labeled training data;
test is not simply a collection of shorter passages. The approximately 0.096
crop loss resembling the public gap is not evidence of a common cause.
The result motivates a robustness experiment, not a diagnosis of hidden edits.

The next-round design was specified in
[the detailed weak-crop protocol](next_round_crop_protocol_20261109.md) and
`experiments/supervised/q5_crop_plan.json`. It has now been executed through its
declared invalid-topic stopping path: six diagnostic random fits, no export.
See [the execution record](q5_crop_results_20261109.md). The executed pilot below
remains a separate historical experiment.

## One controlled crop-training pilot

After observing that weakness, one exploratory pilot was executed, with its
plan saved before fitting. Both conditions use the saved 8,000-row WC120
vocabulary/IDF/normalization and identical LinearSVC parameters. The baseline
fits the original training documents with weight one. For each training text
with at least 150 whitespace words, the intervention assigns weight 0.5 to the
original and 0.5 to its first 100 words; other originals retain weight one.
There are 6,420 crops and 14,420 rows, with total weight 8,000 in each condition.
Only the classifier is refitted. No calibration or test prediction is performed;
these results use raw decision margins, not submission probabilities.

| Development condition | Original training | Crop training | Paired model difference |
| --- | ---: | ---: | ---: |
| Original text, all 2,000 IDs | 0.970006 | 0.964666 | -0.005340 |
| Quote+dash, all 2,000 IDs | 0.967188 | 0.961534 | -0.005654 |
| Digit masking, all 2,000 IDs | 0.966520 | 0.962003 | -0.004517 |
| First 100 words, same 1,627 IDs | 0.876228 | 0.936180 | +0.059952 |

Conditional paired 95% intervals are [-0.008038, -0.002776] for the clean loss
and [+0.049903, +0.070976] for the prefix gain. Baseline margin reproduction
against the saved classifier has maximum absolute error 9.55e-15; its AUROC
matches the saved calibrated model. Exactly two classifier fits completed
without warnings (0.82 / 1.05 seconds), with output/hash/AUROC verification.
Artifacts are in `results/supervised/a_public_gap_20261109_v1_crop_pilot/`.

This demonstrates that the targeted sensitivity is reducible, at a measurable
clean-performance cost. The clean loss exceeds the practical 0.003 budget used
in this research note. **Do not replace WC120 or export this pilot.** The target
was selected after the stress results, and the reused validation set is not an
untouched confirmation set. The fitted representation was held fixed to isolate
classifier training input; a production augmentation pipeline would need a
separate fold-contained evaluation of feature fitting and calibration.

## Methods supported by the research

[RAID (ACL 2024)](https://aclanthology.org/2024.acl-long.674/) reports failures
across unseen generators, generation settings and adversarial edits.
[M4 (EACL 2024)](https://aclanthology.org/2024.eacl-long.83/) independently
documents difficult transfer to unseen domains and generators.
[Xu et al. (EMNLP 2024)](https://aclanthology.org/2024.findings-emnlp.424/)
examines prompts, topics, tasks and text length as separate shift factors.
These sources support evaluating multiple conditions; none establishes a
particular fix or expected score for this competition.

The immediate recommendation is to keep WC120 and change the selection evidence
before expanding the model search. AUC depends on ordering, so moving a decision
threshold or applying one strictly increasing calibration map cannot repair the
single-model ranking. Probability blending can change rankings, but adding a
weaker model is not automatically helpful; ensemble weights need training-only
out-of-fold evidence and a small declared search budget.

1. **Content robustness with an explicit clean-loss budget.** The measured 0.5
   crop-weight pilot establishes a tradeoff. One possible next single-variable
   contrast is original-only versus a fixed smaller crop weight (for example
   0.1), with total weight per original ID still one; choose it before examining
   new results rather than sweeping weights on the same development set. Crop
   variants must stay with their original ID in future calibration/CV folds.
   Measure clean performance and every fixed stress condition; do not optimize
   only the condition that exposed the weakness. Hidden-test benefit remains
   a hypothesis even if this follow-up succeeds.
2. **Confirm a promising intervention with complete refitting inside folds.**
   Use only the original 8,000 training IDs for the next three-fold comparison;
   fit vocabulary, scaling, classifier and calibration inside each fitting
   boundary. Retain the canonical 2,000-row set for course comparisons and call
   it a reused development set. If adding topic-disjoint proxy folds, freeze
   label-free grouping before candidate results, report balance/training-size
   differences and matched random-fold controls. A topic proxy is not a real
   domain/generator label and cannot guarantee hidden-generator performance.
3. **Then test one alternative representation if the evidence warrants it.**
   A CPU-friendly option is NB log-count-ratio weighting, motivated by
   [Wang and Manning (ACL 2012)](https://aclanthology.org/P12-2018/).
   Compare binary counts with and without training-label-derived weighting;
   hold tokenization, budget and classifier fixed. Comparing only against TF-IDF
   would conflate binarization and weighting. This method may also amplify source
   shortcuts and has no established robustness advantage here.
4. **Coordinate the unlabeled study.** The first contrast should be vocabulary/
   IDF learning on fitting texts plus supplied unlabeled texts, with identical
   supervised labels, versus fitting texts alone. A later pseudo-label contrast
   is separate; high confidence under domain shift can reinforce wrong labels.
   Do not use test text for this fitting step. A generic pretrained encoder is a
   higher-cost alternative, with permitted weights and resources checked first;
   a published detector trained on external labeled corpora does not automatically
   satisfy the supplied-label-only rule.

Proposed follow-up guardrails should be fixed before its results: a clean-AUROC
loss budget of 0.003, at least 0.005 improvement in the targeted stress condition,
and positive mean benefit with reported fold variation. These are practical
decision criteria, not a guarantee of public improvement or a retrospectively
preregistered test. Do not repeatedly adjust them after seeing outcomes. A
single candidate may then be refit on 10,000 labels and checked externally,
subject to the team's upload-slot coordination. No upload is part of this note.

## Reproduction and artifacts

Use the recorded modeling environment (`/opt/anaconda3/bin/python` on this
machine, or an environment installed from the supervised requirements). Each
command requires a new output directory and refuses to overwrite a prior audit.

```sh
python -m experiments.supervised.audit_public_predictions --output results/supervised/NEW_predictions
python -m experiments.supervised.audit_overlap --output results/supervised/NEW_overlap
python -m experiments.supervised.audit_public_stress --output results/supervised/NEW_stress
python -m experiments.supervised.pilot_crop_robustness --output results/supervised/NEW_crop_pilot
```

The measured runs are the `a_public_gap_20261109_v1_predictions`, `_overlap` and
`_stress` directories under `results/supervised/`, plus `_crop_pilot` for the two
exploratory classifier fits. They contain source/input
hashes, aggregate diagnostics, ID-only overlap outputs and paired validation
predictions. Raw passages are not stored in the new artifacts. Existing
experiment evidence and checkpoints remain unchanged.

New shared knowledge: submission formatting is sound; strong lexical overlap
was not found; observable marginal shift exists; prefix robustness is weak and
can be improved with a measured clean-performance cost;
the public incumbent remains WC120. The hidden domain/generator/attack mixture,
the causal explanation for the richness reversal, and private-set performance
remain unknown.
