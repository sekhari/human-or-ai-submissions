> Historical Alma log (2026-11-07). Current team integration status and updated Jianing score are in ../README.md.

# Random Forest experiment history

## Initial baseline — recorded from the existing completed run

Run: outputs/rf_20261107T193931_779056Z
Split: acf0a70e863835c0; 8,000 training rows / 2,000 stratified validation rows; seed 42.

| Representation | Train AUROC | Validation AUROC |
| --- | ---: | ---: |
| Word TF-IDF, 1–2 grams | 0.999748 | 0.932584 |
| Word counts, 1–2 grams | 0.999762 | 0.944298 |

Both used 200 trees, unrestricted depth, min_samples_leaf=2, max_features=sqrt.
This is a retrospective snapshot of already completed work, not a claim that Git tracked its earlier development.

## Follow-up plan — fixed before running new experiments

1. Reuse the exact validation IDs. Analyze baseline errors and performance in fixed word-length bins (<=150, 151–250, >250).
2. Compare word counts, style-only, word counts + style, binary word presence, and character TF-IDF. Re-run word TF-IDF as a reference.
3. Select the representation using three-fold stratified CV on the 8,000 training rows only. Learn vocabulary/IDF separately within every fold.
4. On the CV-selected representation, compare four predefined RF settings: current parameters; leaf=5; leaf=10/depth=30; leaf=5/max_features=0.1.
5. Evaluate the candidate representations and the CV-selected tuned model on the unchanged validation set. Record all results, including negative results.
6. Report paired bootstrap uncertainty for the selected model's validation AUROC difference from the count baseline; describe it as exploratory because the validation set has already informed the development direction.

Hypotheses: style features may supply information beyond word identity; binary counts may reduce repetition sensitivity; character features may capture complementary surface patterns; regularization may improve generalization.
These are hypotheses, not established explanations for the Kaggle gap.
No test labels, external supervised labels, Kaggle leaderboard optimization, or teammate prediction files are used.
Unlabeled-data learning, final Kaggle retraining/submission, and teammate ensembling remain separate work.

## Follow-up execution — 2026-11-07

Implementation commit: ca78d50. Baseline snapshot/plan commit: 5214c50.
Successful run: outputs/rf_improvement_20261107T200156_163772Z.
Notebook: Alma_RF_Improvements.ipynb (executed outputs retained).

An initial attempt (rf_improvement_20261107T200133_105496Z) stopped while printing Unicode snippets to a Windows GBK console, before CV training. The runner was corrected to emit UTF-8. Its partial diagnostic folder remains, and it is not a model result.

### Observations before new training

The count baseline's AUROC was 0.883867 for <=150 words, 0.953816 for 151–250 words, and 0.961903 for >250 words.
Confidence-selected baseline errors included non-English passages, a technical abstract, and recipe-like passages. These examples are not an estimate of how common each failure type is.
The style transformer uses English-word regexes, which is a limitation for the observed multilingual text.

### Controlled results

| Representation | Mean training CV AUROC | CV standard deviation | Validation AUROC |
| --- | ---: | ---: | ---: |
| Word counts | 0.936462 | 0.007166 | 0.944298 |
| Word TF-IDF | 0.927524 | 0.010645 | 0.932584 |
| 21 style statistics | 0.985673 | 0.000598 | 0.989913 |
| Counts + style | 0.978003 | 0.001915 | 0.981946 |
| Binary word presence | 0.922607 | 0.008501 | 0.931258 |
| Character TF-IDF | 0.960135 | 0.002231 | 0.961611 |

Training-only CV selected style-only. This selection was not made by ranking the validation scores.
The fixed RF settings comparison selected the original settings (CV 0.985673).
leaf=5 scored 0.984164; leaf=10/depth=30 scored 0.980901; leaf=5/max_features=0.1 scored 0.979265.
Increasing regularization did not help in these tested configurations.

### Interpretation

- The main observed improvement came from representation, not RF tuning.
- Style-only outperformed counts+style, so adding word features did not improve this experiment.
- The selected model's train AUROC was 0.999968, validation AUROC 0.989913, gap 0.010055.
- Versus baseline, validation AUROC increased by 0.045615. The paired bootstrap 95% interval was [0.035675, 0.055108].
- Length-bin AUROCs improved to 0.982366, 0.992830, and 0.988700, respectively.
- At threshold 0.5, false positives increased from 39 to 49, while false negatives decreased from 206 to 47. Better AUROC does not guarantee every threshold-specific error count improves.
- This does not prove unseen-domain robustness or predict the Kaggle score. The validation set already informed the development direction.
- A potential explanation for counts+style being worse is how random feature selection treats 21 style columns among 20,021 columns; this remains an untested hypothesis.

### Verification and pending work

All six holdout AUROCs were independently recomputed from saved probabilities using a rank/pair calculation.
Sample IDs, labels, split membership, training-only CV folds, CV selection, and saved notebook execution were checked.
See the run's verification.json and EXPERIMENT_REPORT.md.

No test model was fit, no submission CSV was generated, and nothing was uploaded to Kaggle or GitHub.
Next: review errors and the English-tokenization limitation, then generate a candidate Kaggle CSV by refitting the CV-selected pipeline on all 10,000 labeled examples. A Kaggle result is needed before claiming that the local-to-test gap improved.
The required unlabeled-data experiment and learning curve are still separate tasks.

## Kaggle CSV export — 2026-11-07

At the user's request, refit the CV-selected style-only RF (unchanged 21 features and parameters) on all 10,000 labeled training rows. Export notebook: Alma_RF_Kaggle_Export.ipynb.
Output: submissions/rf_style_20261107T200909_355010Z/submission_rf_style.csv.
Verified exactly 3,000 rows, columns id,label, IDs in template order, and finite P(AI) in [0,1]. Provenance/hashes are in submission_metadata.json. At the time of this export, no Kaggle upload had occurred. The previous 0.989913 validation score belongs to the 8,000-row model; the refit does not have a new held-out score.


## Team comparison update — 2026-11-07

This corrects the earlier approximate scores and premature description of a finalized model choice. The project planning document identifies the third teammate as Abishek (called Ali in conversation); his full official name and NetID are not yet confirmed.

## Current Kaggle Public Scores (2026-11-07)

Transcribed from the team's Kaggle screenshot supplied by Alma. These are Public Scores, not private/final leaderboard results or local validation scores.

| Contributor | Kaggle submitter | Submission file | Public Score |
| --- | --- | --- | ---: |
| Abishek (called Ali in conversation) | ac3388 | stack_nbsvm_stylo.csv | 0.92847 |
| Liyan (Alma) Yang | Liyan Yang | submission_rf_style.csv | 0.89682 |
| Jianing Li | Jianing Li | submission_Jennie_TF_MiniLM_SVM_C1.csv | 0.84249 |

Abishek's submission is currently the leading candidate and the team's intended direction. He is still improving it; the final model/version is not yet locked. The screenshot does not show a final submission selected. Model internals cannot be established from filenames alone.

The project planning document names the third teammate **Abishek**. His Kaggle display name is **ac3388**; full official name and Cornell NetID remain unconfirmed (the display name alone does not verify a NetID).

Alma's RF artifacts remain reproducible comparison materials. Abishek's and Jianing's code and the combined report still need integration. Earlier no-upload statements describe the state at those experiments. Local validation measurements are unchanged.
