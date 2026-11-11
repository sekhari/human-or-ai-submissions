# Kaggle candidate files

Use `<run_id>.csv` with `id,label` and 3,000 probabilities for class 1 (AI).
Check with `python3 -m human_ai.check_predictions submissions/<run_id>.csv --kind submission`.
The checker does not upload. Coordinate team upload slots in iMessage; record
actual uploads in `leaderboard_log.csv`. No fabricated example scores are included.

Current highest reported public candidate, **Public Score 0.96970** in the
user-provided Kaggle screenshot (+0.09571 versus WC120 0.87399):
`a_roberta_20261109_v1_raw_full10000.csv`. It refits the measured raw-text
RoBERTa-base recipe from the generic checkpoint on all 10,000 supplied labels:
FP32 MPS, two epochs, seed 42, 512 tokens, microbatch 4/accumulation 4 and
AdamW learning rate 2e-5.
Original whitespace is preserved in fitting and prediction. This recipe has
the highest measured clean random/topic and canonical development AUROCs among
the evaluated candidates, with known whitespace sensitivity; its source-model
0.999297 development AUROC is not a held-out score for this full-label refit.

All 3,000 probabilities and IDs pass the submission checker. Reloading the final
checkpoint reproduces every test margin and probability exactly, and the
independent zero-fit audit passes. Metadata and complete execution evidence
are in the companion `.metadata.json` and
`results/supervised/a_roberta_20261109_v1_raw_full10000/`. The companion metadata
preserves its export-time unscored/not-uploaded state. The subsequent user
screenshot and observed score are recorded separately in
[`roberta_public_result_20261109.md`](../docs/roberta_public_result_20261109.md)
and `leaderboard_log.csv`; private performance is unknown.

Previous Q5 incumbent: `a_q5_20261108_v1_wc120_full10000.csv`, a separate
word-plus-character SVM refit on all 10,000 supplied labels. Its metadata
records 3,000 verified test IDs, checksums and exact model-reload agreement.
The exporter did not upload it. The source 8,000-row model's validation AUROC
is 0.970006; the full-label candidate itself has no held-out score.

The later six-configuration regularization follow-up selected the same WC120
C=1 settings and exported no new candidate. WC120 retained its earlier lead
at 0.87399 until the later RoBERTa 0.96970 result; see the upload log.

Scored exploratory candidate (public AUROC **0.87225**, user screenshot):
`a_q5_quotes_20261108_v3_wc120_norm_full10000.csv`. It uses the fixed WC120
SVM C=1 with quote canonicalization inside training and prediction, refit on
all 10,000 supplied labels. It contains 3,000 verified class-1 probabilities
in original test-ID order, with exact saved-model reload agreement.

This file is labeled `exploratory_robustness_tradeoff`: its source 8,000-row
model improves the quote-change validation condition by 0.001522 while losing
0.001331 on original text. Its public score is 0.00174 below WC120; private
performance is unknown. It did not replace the then-leading 0.87399 WC120.
The exporter did not upload this candidate. Its companion
`a_q5_quotes_20261108_v3_wc120_norm_full10000.metadata.json` records the source
study, eligibility checks, CSV/model/test hashes and the absence of a held-out
score for the full-label refit.

Scored four-group-study candidate (public AUROC **0.85812**, user screenshot):
`a_q5_richness_20261108_v4_c60_richness_full10000.csv`. It combines the
original C60 character TF-IDF with three document-local richness features and
fixed, training-fitted scaling inside the complete calibrated C=1 SVM pipeline.
Training-only CV selected this model before validation; its source 8,000-row
model achieves validation AUROC 0.974508 versus WC120's 0.970006 (paired
difference +0.004502, conditional interval [+0.00103495, +0.008509275]).
The four-group primary contrast independently supports a richness increment
after the explicit length adjustment; see the supervised README for limits.

All frozen export guardrails passed. This separate refit uses all 10,000 labels
and contains 3,000 test probabilities in original ID order (3,001 lines with
the header). Schema, ID, finite-probability, exact model-reload and independent
artifact checks passed. The companion metadata records CSV/model/test hashes,
eligibility and the source study. No held-out score belongs to the full-label
refit. The later user screenshot records public AUROC 0.85812, below the
then-leading public-0.87399 WC120. RoBERTa now has the highest reported public
score; see the current result above and the
[November 9 research note](../docs/public_score_research_20261109.md) for
the score reversal, executed integrity/shift/stress audits and next experiments.

