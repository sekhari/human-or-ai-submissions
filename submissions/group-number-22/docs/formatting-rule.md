# Conditional formatting rule

Issue #11 transferred from Gtpatel02 to rdevasia25 at the user's request. The
GitHub assignee, owner label, work plan, context, and notebook owner now agree.
Issue #10 was merged before implementation. No prior #11 branch or PR was present.

## Fixed protocol

Notebook 11 compares all eight requested candidates on the exact five training
folds from #10. Use its word/character/NB feature settings, solver settings, and
seed 42. Reference and rule-all share their identical learned model and margin
reference, so their comparison changes only the rule. Each other candidate gets
a fresh full feature/model fit in each fold. There are 40 policy-fold evaluations
and 35 distinct learned CV fits, not 40 separate classifier fits.

The pattern is literal two consecutive newlines. It is narrower than notebook
13's blank-line regex. Preserve raw passages and use notebook 02's loaders.
Before specialist filtering, count human matches in the original current fit
partition. Any human match disables the rule for that fit, even if specialist
filtering removes the counterexample. Never derive fold rules from evaluation
labels. No global normalization changes the source text.

Soft offset 2 is primary. Hard offset 1000 is an ablation with separate training
OOF selection. Both use equal standardized-margin averaging, greedy minimum gain
.0005, at most five members, grid-order ties, and fewer members within .0005 of
the best path. Both choices are saved before validation. Validation cannot choose
which scoring path to report as primary.

All-fit margins use eligible unlabeled reference rows; specialists additionally
exclude literal blank-line rows. Store population mean/std and ordered reference
ID hashes. Features and supervised NB ratios fit inside the current training
partition only. No test-batch score normalization or calibration is used.

The hard path maps the pre-offset base margin through a fixed arctan function:
flagged scores lie in [.99,1), others in (0,.99). This avoids sigmoid saturation
near offset 1000. Observed score order and ties are checked without labels. An
incompatible ordering fails rather than silently rank-normalizing a batch. Those
bands are imposed by the rule; they are not measured probability calibration.

## Measured results

| Candidate | Soft OOF | Hard OOF | Soft validation | Hard validation | Used fit rows |
|---|---:|---:|---:|---:|---:|
| reference | 0.973497 | 0.973497 | 0.977144 | 0.977144 | 8000 |
| rule_all_wordchar16 | 0.993817 | 0.993821 | 0.995015 | 0.995015 | 8000 |
| rule_no_blank_wordchar16 | 0.991252 | 0.991269 | 0.992074 | 0.992074 | 5374 |
| rule_no_blank_wordchar64 | 0.991787 | 0.991800 | 0.992507 | 0.992507 | 5374 |
| rule_no_blank_casechar64 | 0.992417 | 0.992426 | 0.993638 | 0.993638 | 5374 |
| rule_no_blank_case_svm1 | 0.992386 | 0.992392 | 0.993440 | 0.993440 | 5374 |
| rule_no_blank_nbword4 | 0.982654 | 0.984525 | 0.981579 | 0.982271 | 5374 |
| rule_no_blank_word13_16 | 0.982893 | 0.983753 | 0.982084 | 0.982341 | 5374 |

| Path | Members | Training OOF AUROC | Validation AUROC |
|---|---|---:|---:|
| soft | rule_all_wordchar16, rule_no_blank_case_svm1 | 0.994690 | 0.995390 |
| hard | rule_all_wordchar16, rule_no_blank_case_svm1 | 0.994691 | 0.995390 |

The hard path gives no validation AUROC gain over soft in this run. Soft remains
primary under the fixed plan. Both train AUROCs are 1.0, not held-out performance.

Both train/validation paths, weaker candidates, fold variation, rule state,
reference statistics, attempted greedy additions, and fit costs are retained.
Original fitting pool: 8,000 verified rows. Actual kept rows and class counts are
reported per candidate; the top-level run count describes the full experiment
pool. Training human literal-blank matches: 0.
Original training literal-blank rows: 2626.

The #10 reference OOF difference is 0, and its validation
difference is 0. Search took 541.148 seconds;
the recorded complete run took 932.511 seconds. These are local
split measurements, not Kaggle results. Validation was reused in earlier work,
so confirmation is conditional rather than an untouched final test.

The rule encodes a formatting correlation in the fit data. It can fail on other
human text with blank lines or on differently formatted AI text. The observed
validation subgroup table is descriptive only. Empty or single-class group AUROC
is null, not a fabricated score. #12 must measure formatting and seed sensitivity;
#11 does not infer anything about hidden Kaggle subgroup labels.

## Reproduce and hand off

```bash
uv run --locked jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=3600 notebooks/11_formatting_rule.ipynb
uv run --locked python tools/check_formatting_rule.py
uv run --locked python tools/check_text_handoff.py 11_formatting_rule
uv run --locked python tools/check_text_handoff.py 11_formatting_rule --mode hard
uv run --locked python tools/check_notebooks.py
```

Fresh-kernel execution, synthetic training-only eligibility/vocabulary checks,
hard range/order checks, shared-reference checks, no-fit/no-write helper import,
notebook schema/syntax, plot inspection, and diff checks passed. Both portable
ensembles reproduce all 2,000 validation probabilities exactly in clean Python
processes, with all private artifact hashes verified. No notebook globals,
fitting, or downloads are needed for loading.

Safe record: `results/11_formatting_rule/20261110T214338421386Z.json`.
The private pointer `artifacts/11_formatting_rule/latest.json` names primary soft
and hard models, candidate models, ID-aligned raw/standardized margins and flags,
reference stats, selections, hashes, and `refit_spec.json`. These artifacts remain
ignored. Detailed failure traces and progress also stay private. No upload occurs.
Teammate review is requested in the PR.

For #12/#17, set `FORMATTING_RULE_HELPERS_ONLY=1` before
`%run ./11_formatting_rule.ipynb` to define factories without fitting or writing.
`FormattingArm(candidate).fit(fit_frame, eligible_reference_frame)` rebuilds one
candidate from raw verified rows. Only text enters the classifier; IDs record
provenance. Start fresh for every split or solver condition. Never reuse a shared
reference across different data, settings, or solver seeds.
When #17 mixes this arm with style scores, apply its required outer mean/std
normalization on eligible unlabeled reference rows and freeze those statistics.
Notebook 11 standardizes components but does not add that outer blend scaling.

An arm's `components(texts)` returns native margins, standardized margins, and
active flags. An ensemble's `components(texts)` returns the mean standardized
margin and active flag weight. `decision_function(texts, offset=0)` gives the rule-off score. Use offset 2 or
1000 for soft or hard scores on that same fixed ensemble. Its configured `predict_proba`
uses the proper mapping. Compare native component margins when #12 asks for raw
margin changes, and keep all saved reference statistics frozen at evaluation.
