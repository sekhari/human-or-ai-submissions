# Research notes and proposed robustness study

Latest evidence: the November 9 user screenshot records quote WC120 public
0.87225 and C60+richness 0.85812, both below WC120's 0.87399. The historical
execution notes below retain their export-time status. See
[the November 9 diagnosis](public_score_research_20261109.md) for actual
submission-integrity, overlap, marginal-shift and paired stress diagnostics.

Original review status: research/design only, November 8, 2026. No new training
or submission was performed during that review. Subsequent execution is
recorded below. This is a technical experiment-planning note, not report prose;
unexecuted protocol choices remain proposals, not a preregistered run.

Subsequent execution: the separately frozen
`a_q5_tuned_20261108_v2_study` has now completed the six-configuration C search
and quote/dash diagnostics for C60, WC120 and the WC120 reference. Both selected
C=1, with no new export-eligible candidate; quote sensitivity was observed.
The subsequent `a_q5_quotes_20261108_v3_study` has now executed Route B1's
single quote-canonicalization intervention. WC120 quote-condition AUROC improves
from 0.967153 to 0.968675 (+0.001522, conditional paired interval
[+0.0005494, +0.002503275]), while original-text AUROC drops from 0.970006 to
0.968675. All three quote-CV differences are positive. This supports a specific
format-sensitivity tradeoff, not a general or hidden-test improvement.

The normalized WC120 model passed the predeclared exploratory loss budgets;
one candidate, `submissions/a_q5_quotes_20261108_v3_wc120_norm_full10000.csv`,
was refit on 10,000 labels and exported with 3,000 verified predictions.
Its metadata category is `exploratory_robustness_tradeoff`; no new Kaggle score
is recorded, and the old public-0.87399 WC120 file remains the incumbent.
Five tests and full artifact verification passed. See the supervised README
for exact commands and frozen provenance. The later authorized Route B2
execution is recorded below; broader W60, grouping and augmentation proposals
have not been executed.

The four-group `a_q5_richness_20261108_v4_study` now measures Route B2 with
unchanged C60/C=1/calibration. R0/R1/R2/R3 validation AUROCs are
0.969550/0.969825/0.974508/0.974838. The primary R3-R1 difference is +0.005013,
conditional paired interval [+0.00200635, +0.008241825], with positive gains in
all three training folds. Length-only, length-given-richness and interaction
intervals include zero. Thus richness provides incremental predictive information
under this explicit length adjustment and fixed geometry; causal independence
from length or hidden-domain transfer is not established.

Training-only CV selected R2 before validation (its mean differs from R3 by
only 0.0000004095). R2 improves against the historical WC120 incumbent by
0.004502, conditional interval [+0.00103495, +0.008509275], and passes the frozen
export rules. The larger R3 validation point estimate does not trigger a
post-validation model switch. One eligible full-label refit exported
`submissions/a_q5_richness_20261108_v4_c60_richness_full10000.csv`, with 3,000
checked probabilities, exact model-reload agreement and no held-out evaluation
for that 10,000-label model. Independent verification passed and has a separate receipt.
No new Kaggle upload or public score is claimed; the old public-0.87399
submission remains the incumbent.

## Objective and existing evidence

Prioritize experimental validity, informative controls and interpretable
decisions, following the user's report that the professor emphasizes design
and completeness. Leaderboard improvement remains an empirical secondary goal.

| Model | Recorded development AUROC | User-reported public AUROC |
| --- | ---: | ---: |
| W60 word SVM | 0.943876 | 0.85157 |
| WC120 word + character SVM | 0.970006 | 0.87399 |

The public gain is +0.022420, compared with +0.026130 on development validation.
Both models have a roughly 0.09 validation/public difference. This motivates
investigating validation representativeness, but does not identify its cause.
The local scores use 8,000-label models; the submissions use separate
10,000-label refits. Public labels and true source/generator groups are unknown.

Character-only C60 already achieves development AUROC 0.969550. Its small
overall difference from WC120 does not establish a reliable ranking. Preserve
all original Q1/Q5 evidence and the canonical 8,000/2,000 split.

## Primary research question

Does the measured character-feature benefit survive controlled surface-form
changes and meaningful group separation? If a weakness is identified, can a
single targeted intervention reduce it while retaining clean-data performance?

Distinguish three claims throughout:

1. Observed: a model changes under a particular diagnostic condition.
2. Hypothesized: this sensitivity may reflect an unstable shortcut.
3. Unestablished: the shortcut caused the Kaggle gap, or fixing it improves
   unseen-domain/private-leaderboard performance.

## Evidence informing the design

| Primary source | Useful finding or method | Proposed transfer and limitation |
| --- | --- | --- |
| [RAID, ACL 2024](https://aclanthology.org/2024.acl-long.674/) | Evaluates text detectors under multiple generation and attack conditions; robustness varies by condition. | Evaluate several targeted conditions separately. Do not assume the hidden competition set contains any particular RAID attack. |
| [CheckList, ACL 2020](https://aclanthology.org/2020.acl-main.442/) | Uses behavioral tests to expose failures hidden by aggregate evaluation. | Measure paired response to small typography changes. Authorship-label preservation needs justification, and unchanged model inputs are uninformative tests. |
| [WILDS, ICML 2021](https://proceedings.mlr.press/v139/koh21a.html) | Real deployment shifts can create a substantial gap between familiar and shifted evaluations. | Add a justified auxiliary split; it cannot diagnose this competition's hidden shift by itself. |
| [scikit-learn 1.5 grouped CV](https://scikit-learn.org/1.5/modules/cross_validation.html#cross-validation-iterators-for-grouped-data) | Group-aware folds prevent a group from appearing on both sides of a split. | Use genuine near-duplicate/template groups when available. Topic clusters are proxies, not true generator/domain labels. |
| [Deduplicating Training Data Makes Language Models Better, ACL 2022](https://aclanthology.org/2022.acl-long.577/) | Motivates auditing duplicated text and evaluation overlap. | An audit is justified; a gain from deleting duplicates in this classifier is not established by the language-model study. |
| [DetectRL, NeurIPS 2024 Datasets and Benchmarks](https://arxiv.org/html/2410.23746v3) | Separates domain/model/attack generalization and text-length evaluation. | Compare equal-ID conditions and matched-length subsets, rather than conflating different sample populations. |
| [El Attar et al., Findings IJCNLP-AACL 2026](https://aclanthology.org/2026.findings-aacl.10/) | Reports context dependence of many linguistic features and comparatively transferable lexical-richness signals in its multi-domain/model experiments. | Test a small, document-local feature block with a length control. Do not assume the paper's results establish gains here. Its appendix computes global hapax using entire train/evaluation splits separately; that evaluation-corpus-dependent operation is incompatible with this project's validation-fitting contract. |
| [Wang and Manning, ACL 2012](https://aclanthology.org/P12-2018.pdf) | Combines NB log-count-ratio feature weighting with a discriminative linear classifier. | A cheap secondary candidate, distinct from ordinary Multinomial NB. The original sentiment/topic results do not establish AI-text or cross-domain gains. |

## Phase A: diagnostics before another parameter search

### A1. Near-duplicate and template audit

- Audit the canonical train/validation boundary using reproducible lexical
  shingles or shared spans, followed by manual inspection of the nearest pairs.
- Freeze tokenization, similarity thresholds and sampling before evaluating
  model performance under any proposed grouping.
- Record cross-boundary pair counts, affected validation IDs, label conflicts,
  group-size distribution and a few reviewed examples without storing passages.
- Absence of convincing groups is a valid result; do not invent a source split.
- If removal is studied, include a same-count, class-matched random-removal
  control to separate overlap removal from reduced training sample size.
- A confirmed material leak can justify a separately versioned split repair
  and rerunning directly compared models; never silently replace the old split.

### A2. Fixed-model behavioral matrix

Use the saved 8,000-row W60, C60 and WC120 models, without tuning or refitting:

| Model | Original text | Quote typography change only | Dash typography change only |
| --- | --- | --- | --- |
| W60 | reference | diagnostic | diagnostic |
| C60 | reference | diagnostic | diagnostic |
| WC120 | reference | diagnostic | diagnostic |

The transformations must be specified precisely before execution, applied
symmetrically to both labels, and manually checked for unchanged authorship
and substantially unchanged content. Report the fraction of documents actually
modified and whether model inputs changed. Existing lowercasing/whitespace
processing can make proposed tests redundant; do not count built-in invariance
as a newly demonstrated robust mechanism.

Measure AUROC and its paired change, absolute probability change, threshold-0.5
error transitions, and label-specific effects. Resample original IDs together
with all their variants; variants are not independent observations. These are
local sensitivity tests, not simulations proven to match hidden test attacks.

For a separate length diagnostic, select the same sufficiently long documents
in each comparison, then compare their full text with a fixed-length prefix.
Report both class counts. Do not compare all original documents with only the
long-document subset and attribute the difference entirely to length.

### A3. Auxiliary grouped validation, conditional on A1

Only if there are credible groups, add group-disjoint CV inside the original
8,000 training IDs. Keep the canonical 2,000 validation rows unchanged.
Report training size, class balance and group-size differences relative to the
random-fold comparison. Grouped and random CV need comparable budgets; group
constraints may prevent exact size/balance matching and must be disclosed.

Refit the vocabulary, IDF, scalers, classifier and calibration solely within
each permitted fitting boundary. Never score the full-10,000-label checkpoint
on a newly drawn auxiliary holdout. Use group-aware uncertainty when sample
dependence makes ordinary document-level bootstrap inappropriate. Single-class
or tiny groups do not support a reliable subgroup AUROC.

## Phase B: choose one intervention using Phase A

### Route B1: targeted canonicalization if surface sensitivity is observed

This route has been measured in v3 with the single frozen quote mapping above.
Its constructed invariance and clean-data cost must both accompany the result.

Compare a fixed model with and without one deterministic typography rule,
using the rule consistently at training and prediction. Score both original
and transformed validation conditions, and one predeclared stress condition
not used to choose the rule. Normalizing only validation text in Phase A is
a diagnostic; it is not the same operation as training a normalized pipeline.

If augmentation is used instead, keep originals and variants together in every
training/CV/calibration group. Control added sample weight or include an
original-copy control, so duplication does not silently change effective
regularization. Start with canonicalization because it needs fewer moving parts.

### Route B2: interpretable richness features if representation is the issue

The user subsequently authorized this separate CPU-friendly adaptation, now
measured in v4; it is not a paper reproduction. The frozen richness block is
MATTR with all windows of min(50,n) tokens, unigram Shannon entropy divided by
log(n), and repeated-bigram fraction. The length block is log1p(letter-token
count) and log1p(raw character count). Lowercase Unicode letter-only tokenization,
empty/short-document rules, fold-fitted scaling and fixed block weights were
defined before the run in `q5_richness_plan.json`. No validation/test document
aggregates enter feature fitting. The three richness formulas are distinct but
strongly correlated (training MATTR/entropy correlation about 0.961); their
individual contributions were not isolated, and residual length dependence
remains possible. No training or validation document had fewer than 50 tokens.

| Configuration | Character representation | Length block | Richness block |
| --- | --- | --- | --- |
| R0 | C60 | absent | absent |
| R1 | C60 | present | absent |
| R2 | C60 | absent | present |
| R3 | C60 | present | present |

Primary contrast: R3 minus R1, testing incremental predictive information beyond
the explicit length block. R1 minus R0 diagnoses the value of length, and R2
minus R0 is secondary. Fix the classifier/search budget and declare block
scaling, normalization and weights; adding a block changes feature geometry,
so do not claim perfect capacity or regularization equivalence. Fit all learned
scaling within the training/calibration folds. Report clean, length-bin and
justified auxiliary-group results, along with time and actual feature counts.

The original plan prohibited automatically executing both B1 and B2. B1 was
completed first, and B2 was later explicitly authorized by the user with its
own frozen main comparison and stopping rule; it was not silently appended
to the B1 experiment.

### Secondary method, only if there is a remaining concrete question

An NB-SVM-inspired candidate requires a matched binary-count SVM control.
Compare ordinary binary counts against the same counts multiplied by a
training-label-derived NB log-count ratio. Retain existing TF-IDF results as
historical references, not the sole control: otherwise binarization and label
reweighting would change together. All ratio/vocabulary learning belongs inside
each training and calibration fold. Discriminative weighting can also amplify
source shortcuts; inspect auxiliary performance before claiming robustness.

## Coordinate Q3 rather than duplicating a teammate's work

If Gabriel's current work permits, distinguish two unlabeled interventions:

| Condition | Vocabulary/IDF text | Classifier labels |
| --- | --- | --- |
| U0 | labeled fitting rows only | true fitting labels |
| U1 | fitting rows plus supplied unlabeled data | true fitting labels |
| U2 | same representation as U1 | true labels plus one fixed pseudo-label round |

U1-U0 isolates unsupervised feature learning; U2-U1 isolates the additional
pseudo-label contribution. No validation/test rows join feature fitting.
Freeze selection thresholds, class-count caps/weights and stopping rules.
Confidence needs calibration, but high confidence is not proof that shifted
unlabeled examples are correctly labeled. See the
[SelfTrainingClassifier documentation](https://scikit-learn.org/1.5/modules/generated/sklearn.semi_supervised.SelfTrainingClassifier.html).

## Completion criteria and reporting boundaries

- Complete the diagnostic matrix and one chosen intervention; failure to find
  a weakness or gain is an admissible, informative result.
- Report a predeclared primary contrast, secondary diagnostics, class counts,
  conditional paired intervals, resource cost and all attempted settings.
- Preserve the old development scores; this follow-up was motivated by public
  feedback and is not an untouched test evaluation or a retroactive Q5 plan.
- Independent clean-environment reproduction remains necessary for final
  reproducibility claims. Each member records only their actual personal checks.
- Select at most one resulting candidate for a new Kaggle check after local
  selection. Do not use leaderboard success as the stopping condition.
- A plausible causal story is not a demonstrated generation mechanism. The
  strongest justified claim is the observed effect of the controlled change
  under the measured conditions.
