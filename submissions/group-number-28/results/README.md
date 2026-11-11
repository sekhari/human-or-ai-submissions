# Recorded results

One immutable folder per actual run: `<track>/<run_id>/`.
See `docs/experiment_contract.md` for schemas. Q1 now has recorded evidence under
`supervised/a_q1_20261107_v1_*`; see the supervised-track README. Commit small evidence artifacts, not model
binaries or full passages. Avoid a shared mutable results spreadsheet/file.

Current expanded comparison: `supervised/a_q1_expanded_20261108_v2_*`.
The study contains 64 configurations / 192 fold fits across seven classifier
families and one representation control. Eight model runs retain held-out
predictions; a separate full-data SVM produced the historical word-only CSV.
The verification JSON checks all selected settings, stored scores and exact
fresh-refit reproduction. Initial v1 artifacts remain unchanged.

Q5 evidence is in `supervised/a_q5_20261108_v1_*`: four frozen representation
conditions, 12 training-only CV fits, exact reproduction of the old baseline,
four held-out prediction files, paired bootstrap intervals, descriptive error/
length diagnostics, and saved-model verification. WC120 validation AUROC is
0.970006 versus W60 0.943876. A separate post-study refit now lives in
`supervised/a_q5_20261108_v1_wc120_full10000/` and exports a checked 3,000-row
candidate under `submissions/`. It includes all supplied labels, so its metrics
have no held-out score. Final team selection and independent-environment
reproduction are still pending. Original Q1/Q5 study evidence is unchanged.

The separate `supervised/a_q5_tuned_20261108_v2_*` follow-up records six C
configurations, 18 candidate-fold fits, two fresh C=1 finalist models, paired
intervals and nine typography diagnostic conditions. Both representations
selected the original C=1, and no new candidate passed the export rules.
Verification passed; the negative tuning result and quote sensitivity are
preserved without replacing the existing public-0.87399 submission.

The measured quote intervention is in
`supervised/a_q5_quotes_20261108_v3_*`: 12 training-only CV fits, two new
normalized 8,000-row models, and the four-pipeline by three-input-condition
matrix. Its primary WC120 quote-condition difference is +0.001522, conditional
paired 95% interval [+0.0005494, +0.002503275]. Original-text AUROC drops by
0.001331 to 0.968675; the normalization is a measured tradeoff, not a blanket
improvement. Quote invariance is guaranteed by the preprocessing construction.

Five tests and complete saved-artifact verification passed. A maximum of one
eligible candidate was exported through a separate 10,000-label refit:
`supervised/a_q5_quotes_20261108_v3_wc120_norm_full10000/`, with its CSV and
metadata under `submissions/`. That refit has no held-out score and no recorded
Kaggle upload/score. The earlier public-0.87399 incumbent and all prior evidence
remain unchanged. Provenance retains the actual commit/dirty state, frozen
executing-source snapshots, input/source hashes and checked model reloads.

The four-group study is recorded under
`supervised/a_q5_richness_20261108_v4_*`. It preserves C60, the canonical split,
C=1 and calibration, adding two fixed length features and/or three document-local
richness features. Twelve training-only folds precede three new calibrated
8,000-row fits; checksum-guarded C60 and WC120 models supply historical controls.
The validation matrix contains all four ablation groups plus the WC120 reference
under original, quote and dash conditions (15 evaluations).

Original-text AUROCs are C60 0.969550, +length 0.969825, +richness 0.974508,
and +length+richness 0.974838. The primary R3-R1 difference is +0.005013,
conditional paired interval [+0.00200635, +0.008241825], and all three training
fold differences are positive. Length-only and length-given-richness intervals
include zero; no length/richness interaction is established. Richness statistics
are mathematically distinct but strongly correlated in these data.

Training-only CV selected `c60_richness` before validation, ahead of the combined
model by only 0.0000004095 in mean CV; this does not establish superiority between
them. That selected model improves validation AUROC against WC120 by 0.004502,
conditional interval [+0.00103495, +0.008509275], and improves all three length-bin
point estimates against WC120. Its CV gains against WC120 are positive in two
of three folds. It passed the frozen export rules and produced one separate
full-label refit in `supervised/a_q5_richness_20261108_v4_c60_richness_full10000/`.
The matching CSV/metadata under `submissions/` records 3,000 checked probabilities,
exact model-reload agreement and no held-out evaluation for that 10,000-label
model. Nine tests passed before the run; independent artifact verification
passed and has its own receipt. Prior evidence and the public-0.87399 incumbent remain unchanged,
with no new Kaggle upload or public score claimed.

