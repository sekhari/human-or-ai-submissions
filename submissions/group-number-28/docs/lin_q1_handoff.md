# Lin Q1 handoff

## Latest RoBERTa result (November 9)

Raw RoBERTa-base, refit on all 10,000 supplied labels, now has a user-reported
Kaggle Public Score of 0.96970 (+0.09571 versus WC120 0.87399). The screenshot,
CSV hash, actual seven-fit comparison, whitespace diagnostic and full-refit
verification are linked in [the public-result note](roberta_public_result_20261109.md).
The submission is `submissions/a_roberta_20261109_v1_raw_full10000.csv`.
This is the current reported public leader; the older study-time incumbent
statements below are historical. Private-score gains are not observed.

For held-out analysis, use the separate 8,000-trained model's predictions at
`results/supervised/a_roberta_context_20261109_v1_study/fits/canonical_8000/validation_predictions.csv`.
The full 10,000-label refit has no held-out score. Neural work uses
`/opt/anaconda3/envs/python_course/bin/python`; the measured recipe, hashes and
actual commands are archived with the results. Raw passages, pretrained caches
and fitted checkpoints remain local; original whitespace is preserved in the
submitted model's training and prediction.

## Four-group length/richness ablation (November 8)

`a_q5_richness_20261108_v4_study` measured the authorized second-round
four-group design with the canonical split, original C60 TF-IDF, C=1 and
calibration fixed. Length adds log1p token/character counts; richness adds
MATTR-50, normalized unigram entropy and repeated-bigram fraction. Tokens are
lowercase Unicode letter-only words under the frozen regex. Each numeric block
has a training-fitted StandardScaler and fixed dimension-based weight, entirely
inside CV/calibration; there is no final row normalization or parameter search.

| Group | Mean training-only CV AUROC | Validation AUROC |
| --- | ---: | ---: |
| R0: characters | 0.963434 | 0.969550 |
| R1: characters + length | 0.963743 | 0.969825 |
| R2: characters + richness | 0.967464 | 0.974508 |
| R3: characters + length + richness | 0.967464 | 0.974838 |

- Primary R3-R1: +0.005013, conditional paired interval
  [+0.00200635, +0.008241825]. All three training-fold differences are positive,
  with mean +0.003720413. This supports incremental prediction under the chosen
  explicit length adjustment and feature geometry, not causal length independence.
- Length-only gains +0.000275 and length-given-richness gains +0.000330 both
  have intervals including zero; the interaction interval also includes zero.
  The length-only model loses about 0.003241 against C60 in short texts.
- Training-only CV selected R2 before validation; its mean exceeds R3 by just
  0.0000004095. R3 has the larger validation point estimate. Neither comparison
  establishes superiority of R2 over R3, and no validation-based runner-up switch
  is permitted by the frozen export rule.
- Selected R2 exceeds the old WC120 validation AUROC by +0.004502, interval
  [+0.00103495, +0.008509275], with gains in two of three training folds and all
  three validation length bins. Original/quote/dash AUROCs are
  0.974508/0.971734/0.974527; quote sensitivity remains. All six frozen export
  checks pass. One separate 10,000-label refit generated
  `submissions/a_q5_richness_20261108_v4_c60_richness_full10000.csv` with 3,000
  checked probabilities and exact model-reload agreement. That refit has no
  held-out evaluation; independent verification passed and has its own receipt.
- Twelve CV fits, three new 8,000-row models and 15 validation conditions retain
  fold margins, scalar feature/scaler audits, paired bootstrap contrasts,
  length groups and source hashes. Nine tests passed before full-size execution.
  Richness formulas are distinct but strongly correlated; no auxiliary source
  groups or independent OOD evaluation were invented.
- The original public-0.87399 file remains the incumbent until an external
  check. No new Kaggle upload or public score is claimed. For any eventual
  Q4 analysis of R2, use
  `results/supervised/a_q5_richness_20261108_v4_c60_richness/validation_predictions.csv`,
  never a full-10,000-label model against the former validation rows.

## Quote-canonicalization follow-up (November 8)

`a_q5_quotes_20261108_v3_study` is complete and verified. It retained the
canonical split, C=1, feature budgets and calibration, adding one stateless
quote mapping inside the full feature/calibration pipeline. Four raw/normalized
pipelines received 12 training-only fold fits; two normalized models were then
fitted on the canonical 8,000 rows and evaluated in the 12-cell input matrix.

- Primary WC120 quote-condition AUROC: raw training 0.967153, normalized
  training 0.968675; paired difference +0.001522, conditional 95% interval
  [+0.0005494, +0.002503275]. Quote CV gains are positive in all three folds.
- Original-text WC120 AUROC decreases from 0.970006 to 0.968675 (-0.001331),
  with a conditional difference interval [-0.00269595, -0.0000022]. Short-text
  performance also decreases by about 0.003500. This is not an overall gain.
- WC120 passes the frozen exploratory tradeoff rules; C60 does not. The
  0.003 clean/CV loss budget is a practical preference, not a statistical
  noninferiority claim. Equal normalized-model outputs for original/canonical
  quotes are guaranteed by construction, not independent OOD evidence.
- One full-label exploratory candidate was generated:
  `submissions/a_q5_quotes_20261108_v3_wc120_norm_full10000.csv`, 3,000 checked
  probabilities after refitting on all 10,000 labels. Metadata labels it
  `exploratory_robustness_tradeoff` and records exact reload agreement and
  CSV/model/test hashes. The full-label refit has no held-out evaluation.
- No new Kaggle upload or score is recorded. Preserve the original WC120
  candidate with user-reported public score 0.87399 until an external check.
  If the new approach is eventually selected, Alex should analyze
  `results/supervised/a_q5_quotes_20261108_v3_wc120_norm/validation_predictions.csv`,
  never the full-label model against its former validation rows.
- Five tests and full saved-artifact verification passed. Provenance records
  commit `38a5c0b0d87c90e805c5ce9626e940ddec32ddb6` with actual uncommitted
  changes, exact source snapshots and original-evidence guards. Independent
  clean-environment reproduction and hidden-test transfer remain pending.

## Regularization follow-up (November 8)

`a_q5_tuned_20261108_v2_study` completed C60/WC120 x C={0.1,1,10} with 18
training-only candidate-fold fits. Both representations selected their original
C=1; fresh 8,000-row finalist fits reproduce C60 0.969550 and WC120 0.970006.
No new candidate met the frozen export rules, so the current public-0.87399
WC120 file remains unchanged. No new Kaggle upload occurred.

Quote normalization at evaluation changed 157 validation texts and reduced
WC120 AUROC by 0.002853 (15 threshold-0.5 decision flips); dash normalization
changed 177 texts with almost no ranking effect. C60 shows the same qualitative
pattern. These diagnostics motivated the later normalization ablation above;
they do not prove the cause of the public gap. The nine paired prediction sets
and counts remain saved in the regularization study.

Five new tests and the full artifact verifier passed. Original Q1/Q5 evidence
is unchanged. This public-feedback-motivated follow-up is development research,
not an untouched final estimate. Reproduction commands and the predeclared
eligibility rules are in the supervised README and `q5_tuning_plan.json`.

## Q5 update (November 8)

The prospectively frozen four-condition Q5 study is complete locally:
`a_q5_20261108_v1_study`. SVM C=1, the canonical split and calibration protocol
are unchanged. No adaptive parameter search followed the scores.

- W60 baseline: 0.943876; C60 characters: 0.969550; WC120 combination: 0.970006;
  W120 larger word vocabulary: 0.944597. All four train AUROCs are 1.0.
- Primary WC120 minus W60: +0.026130, conditional paired interval
  [+0.020083, +0.031971]; all three training-only CV folds agree in direction.
  W60 exactly reproduces Q1 probabilities. WC120 is only 0.000456 above C60;
  the combination has not been established as superior to character-only.
- At threshold 0.5, 107 old errors are corrected and 34 new errors appear;
  total validation errors fall from 258 to 185. Length-bin results and changed
  IDs/probabilities are recorded; causal failure explanations await review.
- WC120 is the leading Q5 development candidate; final team selection is open.
  If adopted, Alex's Q4 input becomes
  `results/supervised/a_q5_20261108_v1_wc120/validation_predictions.csv`.
  The old Q1 held-out file below remains valid for that historical model.
- Four Q5 tests and ten shared tests passed. Reloaded predictions exactly match
  for all four models, and the primary bootstrap interval was independently
  reconstructed from saved probabilities. Clean-environment/cross-machine
  reproduction by Gabriel remains pending.
- The detached local job completed training, verification and plotting in
  about 159.26 seconds without interruption. Checkpoints can resume identical
  unfinished steps; no external model/API service was used for training.
- The four-condition study did not fit all 10,000 labels. A later, separately
  authorized export now provides
  `submissions/a_q5_20261108_v1_wc120_full10000.csv`: WC120 refit on all supplied
  labels, 3,000 checked probabilities, exact saved-model reload agreement.
  Fit/export took 41.49/55.41 seconds. The exporter did not upload to Kaggle.
  This candidate has no held-out evaluation; continue using the original
  8,000-row WC120 predictions for Q4. The old word-only candidate is unchanged.
- These are development-validation results, not private/public Kaggle scores.
  Conditional intervals omit training variation and selection adjustment;
  source shortcuts and near-duplicate leakage remain unassessed.
- Codex assisted with Q5 design, implementation, execution, automated checks,
  interpretation and plotting. Lin's personal verification is not claimed;
  review the code/results and update the AI-use disclosure with actual checks.

## Current expanded study (November 8)

Q1 now covers LR, NB, SVM, decision trees, random forests, gradient boosting,
and an MLP neural network, plus an SVD LR representation control. The search
completed 64 configurations and 192 candidate-fold fits. Parameters were
selected using training-only coarse search and one predeclared refinement.
Full settings and results are in `experiments/supervised/README.md` and
`a_q1_expanded_20261108_v2_study`.

- SVM C=1 remains the provisional candidate: validation AUROC 0.943876.
  Retuned LR uses C=33.33333333333333 and achieves 0.943699.
- Decision tree: 0.812888; random forest: 0.936392; boosting: 0.939490;
  MLP: 0.939616.
- The first five models use sparse TF-IDF. Boosting uses SVD-128; MLP uses
  SVD-128 plus scaling. The scaled SVD LR control achieves 0.922601, so all
  differences cannot be attributed to the classifier alone.
- MLP seeds 7/142 achieve 0.936834/0.940165. The preselected seed 42 remains
  the primary result; the best seed was not selected after evaluation.
- Alex's current Q4 input is
  `results/supervised/a_q1_expanded_20261108_v2_svm/validation_predictions.csv`.
  SVM settings are unchanged; these predictions are byte-identical to v1.
- All eight models were refit in fresh instances and exactly reproduced their
  saved validation probabilities. Three additional representation/search/leakage
  tests passed, with zero model convergence warnings. Gabriel's independent
  clean-environment and cross-machine reproduction remains pending.
- Current full-data candidate:
  `submissions/a_q1_expanded_20261108_v2_svm_full10000.csv`.
  Its SHA-256 matches the initial candidate because the selected SVM is
  unchanged. A duplicate upload would consume a Kaggle slot without new evidence.
- Report writing is handled separately in Overleaf. Local report prose, TeX
  sources, build entrypoints and PDFs are excluded from this code PR. Exact
  search details remain in the study's JSON/CSV files for later writing.
- "Best" means best among tested settings, not globally optimal. Validation
  had already been inspected in v1; this is a development comparison. Bootstrap
  intervals are not selection-adjusted. No Kaggle scores informed selection,
  and Q5 was not part of this Q1 study. The later Q5 results are recorded above.
- AI also assisted with the added models, search, representation control, seed
  stability and updated report. Lin should review the disclosure below against
  Lin's actual verification; no personal team-member checks are claimed.

## Initial three-model study (historical record)

Q1.1-Q1.4 have measured experiments to support later Overleaf writing. Ownership is Lin:
Q1/Q5; Alex: Q2/Q4; Gabriel: Q3/Q6. This work did not perform Q5 or change raw
data, the canonical split, or teammates' experiment/report files.

### Results and handoff

- Initial SVM C=1: validation AUROC 0.943876; training AUROC 1.0; validation
  accuracy at threshold 0.5 is 0.871.
- LR C=10: 0.943436. Its 0.000440 gap to SVM has a paired interval crossing
  zero, so clear superiority is not established.
- NB alpha=0.1: 0.906051. This negative result uses the same features/split;
  it does not establish that every NB variant performs poorly.
- These scores use the fixed 8,000/2,000 split, not Kaggle public/private data.
- Alex's initial Q4 input is
  `results/supervised/a_q1_20261107_v1_svm/validation_predictions.csv`:
  2,000 rows, `id,label,p_ai`. Align by ID and state the error-analysis threshold.
  Update Q4 if team-wide selection changes; Q2 may retain its own fixed baseline.
- `..._svm_full10000/` includes the former validation labels and cannot provide
  an honest Q4 held-out evaluation.
- Gabriel's commands and exact dependencies are in
  `experiments/supervised/README.md`; tested Python is 3.12.7. Ten shared
  contract tests and four initial modeling tests passed. Fresh instances of
  all three models reproduced validation probabilities exactly in the existing
  environment; independent clean-environment/cross-machine checks remain pending.

### Initial local candidate

`submissions/a_q1_20261107_v1_svm_full10000.csv` contains 3,000 probabilities
after full-data refitting with selected Q1 settings. The checked schema is
`id,label`, where `label` is the AI probability. Metadata records provenance
and the CSV hash. No Kaggle upload, submission-slot use, or leaderboard score
was recorded. Team-wide final selection awaits Q3/Q5 evidence.

## Reproduction and development records

Experiments ran from scaffold commit
`17aba0e1f0c7b87fa597b4afb53e0cf8db61b2f9` with uncommitted changes. The actual
dirty state and executed-source hashes are retained, with source snapshots in
each study. The subsequent real archive commits are documented in
`docs/development_history.md`; no execution-time commits were fabricated.
Raw passages and model weights are excluded from Git.

The original local `old_rubric.tex` is unchanged. Q1 follows its criteria and
the supplied Project Expectations, but the latest template, final grading
threshold and deadline have not been independently verified.

## AI-use disclosure draft for Lin to review

Codex assisted with experimental design, Python implementation, execution,
automated verification, interpretation, plots, and the English Q1 draft. The
assistant checked source/split hashes, fold-local feature fitting, calibration
boundaries, probability/schema correctness, metric reconstruction and fresh
refits. Team members' personal review has not been observed or claimed. Lin
should review the implementation and explanations and record the checks Lin
personally performed or modified before adapting this disclosure for Q6.3.
Local sklearn models generated predictions; no external LLM API labeled
competition passages, and no hidden test labels were obtained.

## Decisions and unresolved hypotheses

Compare classifiers with matched representations, tune within training data,
compare families on the fixed validation set, and retain negative results.
Tracks run independently and exchange predictions by ID to reduce dependencies.
SVM has the nominal best score; LR is faster and the difference is unresolved.

The later Q5 study above now measures character-feature gains under the fixed
SVM protocol, including a larger-word-vocabulary resource reference. The added
value of combining words with characters over characters alone, near-duplicate/
source leakage, independent reproduction and hidden-test generalization remain
open. No extra search is needed to complete the four-condition Q5 study.
