# RoBERTa contextual experiment — 2026-11-09

**RoBERTa improves mean clean ranking against WC120 under both existing split schemes.** The paired clean mean differences are +0.030826 for matched random and +0.062907 for the topic proxy. All seven prescribed fits and all five evaluation conditions completed; unfavorable folds are retained.

Whitespace-only regressions versus WC120: matched random -0.092203; topic proxy -0.038810; reused canonical validation -0.100720. Joined prefix100 regressions versus WC120: matched random -0.044863; reused canonical validation -0.093772. These losses remain part of the reported model-family tradeoff.

This is one fixed model-family comparison using supplied labels. It does not isolate a causal effect of pretraining or architecture, and no hidden-test score was measured or inferred. No quality threshold stopped or promoted a fit, no fusion weights were learned, and no submission was exported.

## Frozen recipe and boundaries

Generic `FacebookAI/roberta-base` revision `e2da8e2f811d1448a5b465c236feacd80ffbac7b`, FP32 SDPA, 2 epochs, constant learning rate 2e-05, AdamW weight decay 0.01, gradient clip 1, seed 42, 512 fixed padded tokens, microbatch 4 with 4 accumulation steps (effective batch 16). Partial updates retain every original and weight each microbatch by its actual share of the update. Every fit starts from the pinned generic checkpoint; device-test weights are not reused.

Training uses `/opt/anaconda3/envs/python_course/bin/python` on this M4 Pro. The last prescribed epoch is evaluated without held-loss early stopping or epoch selection. The six CV fits reuse the exact matched random/topic memberships within the canonical 8,000 originals; the additional fit uses the preserved 8,000/2,000 development boundary. No test/unlabeled data or external authorship labels enter this experiment.

Primary AUROC uses the raw AI-minus-human logit margin. Probabilities are computed from the FP32 logits after conversion to CPU float64 and are recorded as secondary outputs. WC120 uses the preserved within-fold SVM margins for CV and the historical calibrated probability on canonical validation. No raw-margin/probability average or held-cohort score transform is used.

## Main comparisons

| Evaluation | Condition | WC120 AUROC | RoBERTa AUROC | Delta | Paired 95% interval | Positive folds |
| --- | --- | --- | --- | --- | --- | --- |
| Matched random | clean | 0.965848 | 0.996673 | +0.030826 | [+0.027132, +0.034447] | 3/3 |
| Matched random | quote_and_dash | 0.962794 | 0.996799 | +0.034005 | [+0.030258, +0.037872] | 3/3 |
| Matched random | digit_runs_to_zero | 0.962375 | 0.996873 | +0.034498 | [+0.030450, +0.038193] | 3/3 |
| Matched random | whitespace_only | 0.939918 | 0.847715 | -0.092203 | [-0.100699, -0.083334] | 0/3 |
| Matched random | prefix_100 | 0.857019 | 0.812155 | -0.044863 | [-0.056958, -0.032244] | 0/3 |
| Topic proxy | clean | 0.932745 | 0.995652 | +0.062907 | [+0.033758, +0.104661] | 3/3 |
| Topic proxy | quote_and_dash | 0.931372 | 0.995601 | +0.064228 | [+0.033737, +0.107372] | 3/3 |
| Topic proxy | digit_runs_to_zero | 0.927329 | 0.995654 | +0.068324 | [+0.039547, +0.105565] | 3/3 |
| Topic proxy | whitespace_only | 0.901362 | 0.862552 | -0.038810 | [-0.059689, +0.039216] | 1/3 |
| Topic proxy | prefix_100 | 0.816874 | 0.818225 | +0.001351 | [-0.038515, +0.041773] | 1/3 |
| Reused canonical validation | clean | 0.970006 | 0.999297 | +0.029291 | [+0.022923, +0.035713] | 1/1 |
| Reused canonical validation | quote_and_dash | 0.967188 | 0.999271 | +0.032083 | [+0.025418, +0.038694] | 1/1 |
| Reused canonical validation | digit_runs_to_zero | 0.966520 | 0.999262 | +0.032742 | [+0.025816, +0.039235] | 1/1 |
| Reused canonical validation | whitespace_only | 0.947669 | 0.846949 | -0.100720 | [-0.120741, -0.083552] | 0/1 |
| Reused canonical validation | prefix_100 | 0.876228 | 0.782456 | -0.093772 | [-0.120125, -0.069601] | 0/1 |

CV values are equal-weight means of three within-fold AUROCs; scores from different fitted models are never pooled for a primary AUROC. Canonical validation is one reused development set, not an untouched confirmation. Prefix100 uses the same original IDs with at least 150 whitespace words, evaluating their first 100 words joined with one ASCII space. It both shortens the passage and normalizes its separators, so a prefix100 decline cannot be attributed solely to reduced context.

Intervals condition on these fitted models and fixed/adaptively reused splits. Matched random and canonical draws resample class-stratified original IDs; topic draws resample whole frozen topic clusters separately inside each held fold. The same original multiplicities are paired across models and conditions. Undefined replicates are counted and not redrawn. The 1,000-replicate intervals are descriptive, without training-variation, adaptive-selection or multiplicity adjustment; overlap among fitting folds prevents treating folds as independent replications.

## All prescribed fold and condition results

| Fit | Condition | Held n | WC120 AUROC | RoBERTa AUROC | Delta |
| --- | --- | --- | --- | --- | --- |
| random_0 | clean | 2606 | 0.966804 | 0.993766 | +0.026962 |
| random_0 | quote_and_dash | 2606 | 0.963158 | 0.994280 | +0.031122 |
| random_0 | digit_runs_to_zero | 2606 | 0.963419 | 0.994435 | +0.031015 |
| random_0 | whitespace_only | 2606 | 0.940932 | 0.882121 | -0.058811 |
| random_0 | prefix_100 | 2087 | 0.857559 | 0.826240 | -0.031320 |
| random_1 | clean | 3354 | 0.963167 | 0.998094 | +0.034927 |
| random_1 | quote_and_dash | 3354 | 0.960731 | 0.997975 | +0.037244 |
| random_1 | digit_runs_to_zero | 3354 | 0.959802 | 0.997983 | +0.038180 |
| random_1 | whitespace_only | 3354 | 0.936328 | 0.809247 | -0.127081 |
| random_1 | prefix_100 | 2699 | 0.855827 | 0.783602 | -0.072225 |
| random_2 | clean | 2040 | 0.967573 | 0.998160 | +0.030588 |
| random_2 | quote_and_dash | 2040 | 0.964493 | 0.998143 | +0.033649 |
| random_2 | digit_runs_to_zero | 2040 | 0.963903 | 0.998202 | +0.034299 |
| random_2 | whitespace_only | 2040 | 0.942496 | 0.851778 | -0.090718 |
| random_2 | prefix_100 | 1634 | 0.857670 | 0.826624 | -0.031046 |
| topic_0 | clean | 2606 | 0.915379 | 0.993426 | +0.078047 |
| topic_0 | quote_and_dash | 2606 | 0.915375 | 0.993463 | +0.078088 |
| topic_0 | digit_runs_to_zero | 2606 | 0.916020 | 0.993269 | +0.077249 |
| topic_0 | whitespace_only | 2606 | 0.876128 | 0.779845 | -0.096283 |
| topic_0 | prefix_100 | 1938 | 0.806409 | 0.747911 | -0.058498 |
| topic_1 | clean | 3354 | 0.956378 | 0.998537 | +0.042159 |
| topic_1 | quote_and_dash | 3354 | 0.952155 | 0.998363 | +0.046208 |
| topic_1 | digit_runs_to_zero | 3354 | 0.950761 | 0.998564 | +0.047803 |
| topic_1 | whitespace_only | 3354 | 0.920674 | 0.875928 | -0.044746 |
| topic_1 | prefix_100 | 2593 | 0.854195 | 0.820969 | -0.033226 |
| topic_2 | clean | 2040 | 0.926478 | 0.994992 | +0.068513 |
| topic_2 | quote_and_dash | 2040 | 0.926588 | 0.994976 | +0.068388 |
| topic_2 | digit_runs_to_zero | 2040 | 0.915207 | 0.995127 | +0.079921 |
| topic_2 | whitespace_only | 2040 | 0.907285 | 0.931884 | +0.024600 |
| topic_2 | prefix_100 | 1889 | 0.790018 | 0.885795 | +0.095777 |
| canonical_8000 | clean | 2000 | 0.970006 | 0.999297 | +0.029291 |
| canonical_8000 | quote_and_dash | 2000 | 0.967188 | 0.999271 | +0.032083 |
| canonical_8000 | digit_runs_to_zero | 2000 | 0.966520 | 0.999262 | +0.032742 |
| canonical_8000 | whitespace_only | 2000 | 0.947669 | 0.846949 | -0.100720 |
| canonical_8000 | prefix_100 | 1627 | 0.876228 | 0.782456 | -0.093772 |

The other conditions normalize quote/dash characters, replace Unicode digit runs with zero, or normalize whitespace. They are dependent sensitivity variants, not separate domains or evidence that the hidden test uses those transformations. No previously sealed suffix/middle condition was opened.

## Ranking complementarity

| Clean fit | Repair credit | Regression credit | Net / all pairs | Spearman rho | Both inverted | AI-human pairs |
| --- | --- | --- | --- | --- | --- | --- |
| random_0 | 3.23% | 0.53% | +0.026962 | 0.7215 | 1528 | 1697448 |
| random_1 | 3.63% | 0.13% | +0.034927 | 0.7557 | 1564 | 2812104 |
| random_2 | 3.21% | 0.15% | +0.030588 | 0.7946 | 384 | 1039244 |
| topic_0 | 8.28% | 0.48% | +0.078047 | 0.6741 | 3050 | 1697448 |
| topic_1 | 4.31% | 0.09% | +0.042159 | 0.7661 | 1578 | 2812104 |
| topic_2 | 7.28% | 0.43% | +0.068513 | 0.7427 | 772 | 1039244 |
| canonical_8000 | 2.95% | 0.03% | +0.029291 | 0.8330 | 453 | 1000000 |

Each AI-human pair earns 1 for correct order, 0.5 for an exact tie and 0 for an inversion. Repair/regression columns are positive/negative changes in that credit divided by all pairs; their difference reconstructs the displayed AUROC delta. A pair may be half-repaired or half-regressed. The same documents contribute many pairs, so pair counts are not independent sample sizes. Joint inversions and rank correlation describe error overlap; they do not establish that a deployable blend would gain. A probability combination would need training-only calibration and a separate evaluation boundary for any learned combination weights.

## Tokenizer coverage and natural length

| Fit | Boundary | Originals | Truncated | Truncated fraction | Full tokens median | Full tokens p95 | Token mass retained |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random_0 | training | 5,394 | 39 | 0.72% | 280.0 | 469.3 | 99.84% |
| random_0 | held | 2,606 | 20 | 0.77% | 273.0 | 469.8 | 99.91% |
| random_1 | training | 4,646 | 33 | 0.71% | 276.0 | 466.8 | 99.90% |
| random_1 | held | 3,354 | 26 | 0.78% | 280.0 | 473.0 | 99.82% |
| random_2 | training | 5,960 | 46 | 0.77% | 277.0 | 471.1 | 99.86% |
| random_2 | held | 2,040 | 13 | 0.64% | 280.0 | 464.0 | 99.88% |
| topic_0 | training | 5,394 | 29 | 0.54% | 284.5 | 473.0 | 99.88% |
| topic_0 | held | 2,606 | 30 | 1.15% | 265.0 | 456.8 | 99.83% |
| topic_1 | training | 4,646 | 41 | 0.88% | 289.0 | 474.0 | 99.90% |
| topic_1 | held | 3,354 | 18 | 0.54% | 259.0 | 460.3 | 99.82% |
| topic_2 | training | 5,960 | 48 | 0.81% | 262.0 | 460.1 | 99.82% |
| topic_2 | held | 2,040 | 11 | 0.54% | 316.0 | 479.0 | 99.97% |
| canonical_8000 | training | 8,000 | 59 | 0.74% | 278.0 | 470.0 | 99.87% |
| canonical_8000 | held | 2,000 | 5 | 0.25% | 279.5 | 465.0 | 99.93% |

Tokenizer counts include special tokens; truncation is full token count above 512 and kept tokens are min(full count,512). Token mass retained is an aggregate count ratio, not a guarantee that omitted content is irrelevant. Fixed 512-token padding still incurs computation for shorter passages. Held coverage is constructed after the final training update. Natural-length subgroup boundaries are whitespace-word tertiles learned only on the corresponding fitting originals; all subgroup counts and undefined single-class results remain in the linked numeric evidence.

## Measured resource cost

| Fit | Train n | Held n | Updates | Epoch loops, min | 5-condition inference, s | Evaluation total, s | All native attempts, min | Final successful process, min | MPS driver max, GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_0 | 5,394 | 2,606 | 676 | 14.53 | 300.58 | 307.14 | 20.07 | 5.30 | 7.496 |
| random_1 | 4,646 | 3,354 | 582 | 12.67 | 408.21 | 416.96 | 19.84 | 19.84 | 7.472 |
| random_2 | 5,960 | 2,040 | 746 | 16.29 | 247.41 | 252.70 | 20.77 | 20.77 | 7.358 |
| topic_0 | 5,394 | 2,606 | 676 | 15.26 | 304.27 | 311.00 | 20.71 | 20.71 | 7.616 |
| topic_1 | 4,646 | 3,354 | 582 | 12.48 | 391.99 | 400.40 | 19.40 | 19.40 | 7.472 |
| topic_2 | 5,960 | 2,040 | 746 | 16.48 | 256.24 | 261.81 | 21.10 | 21.10 | 7.327 |
| canonical_8000 | 8,000 | 2,000 | 1000 | 22.08 | 240.58 | 245.72 | 26.46 | 26.46 | 7.471 |

The seven fits executed 5,008 optimizer updates. Their completed epoch loops total 109.79 minutes (109.72 minutes inside synchronized training-update timers). Five-condition inference totals 35.82 minutes; the broader evaluation windows total 36.60 minutes. There were 8 native process attempts. Their total measured wall time was 148.35 minutes; the final successful attempts total 133.58 minutes. Failed/interrupted attempts, if any, remain included in the former total.

The reconstructed wall-clock span from the first attempt's inferred start (2026-11-09T23:27:00.053738+00:00) to the last finish (2026-11-10T01:59:45.184982+00:00) was 152.75 minutes. It includes gaps between attempts, including a mechanical-repair pause if one occurred. Start times are inferred from each recorded UTC finish minus its measured process duration; this is a timestamp-span reconstruction, not another training timer.

There were 1 failed/interrupted native attempt(s), affecting random_0. The per-fit epoch-loop costs describe the completed prescribed epochs; the native all-attempt costs also retain failed-attempt time. A recovery-only successful process can be short because its training epochs were already saved. See the execution receipts and recovery evidence for the mechanical cause and exact continuation boundary.

Recovery `evaluation_cpu_transfer_001` for `random_0`: MPS-to-CPU simultaneous float64 conversion fails after completed epoch2; fix CPU transfer before dtype conversion. The preservation receipt records 2 completed epochs and 676 completed optimizer updates, 0 new model fits required, seed/recipe changed: false, held-quality scores available before repair: false. The registered repair scope is `evaluation_cpu_transfer_cast_only`; old/new worker hashes and the saved checkpoint hash are preserved in the linked lineage.

The final `random_0` process records 0 new optimizer updates in that process; its full epoch-loop totals retain the saved completed training. This distinguishes recovery execution cost from scientific training cost.

These timing columns overlap and must not be added: evaluation includes held tokenization and condition inference, while native process wall time includes setup, training, epoch checkpointing, evaluation and final save/reload. The native-process sum excludes gaps between subprocesses; the reconstructed span includes those gaps. Both exclude the earlier model download and classical-baseline preparation and the later analysis/auditing/report generation. Neither is the entire research-session duration.

Across the sampled training windows, the largest MPS driver allocation was 7.616 GiB, MPS tensor allocation 5.906 GiB and process RSS 2.152 GiB. The separate whole-process RSS high-water mark was 2.438 GiB. These are maxima over separate observations, not jointly additive components. Unified-memory measurements overlap; do not sum RSS and MPS memory. The sampler covers training and epoch checkpoint windows and stops before held inference and final save/reload. These are observed sampled extrema rather than exact GPU peaks; the process lifetime RSS has broader scope.

No unsupported-op CPU fallback warning was observed in the captured Python warnings or scanned native logs across these exercised paths. Generic fallback was disabled before importing PyTorch. Ordinary CPU tokenization, dispatch, scalar diagnostics and AdamW step bookkeeping remain expected; this is not an operator-by-operator GPU profile or a zero-CPU claim.

## Verification, interpretation and remaining unknowns

The supplied independent zero-fit audit passed 524,299 checks for all seven fits. It inspects saved evidence without a fresh fit or fresh checkpoint inference.

The saved numerical checks cover finite forward/backward updates, MPS gradients and optimizer moments, changed encoder/classifier parameters, and final checkpoint reload agreement on two clean held inputs after training. This checks numerical consistency, not independent-environment reproduction or full-model task correctness.

The topic split remains a lexical-topic/style proxy with only 12 existing clusters, including mixed/style groups; it is not a true source, generator or deployment-domain holdout. These are development comparisons after earlier diagnostic exploration, one recipe and one seed. They cannot establish private-leaderboard transfer or eliminate source/near-duplicate leakage. Better local AUROC would not guarantee a higher competition score; unfavorable local results do not rule out other encoders or training recipes.

New shared assets are seven independently bounded contextual checkpoints, paired per-original predictions on five conditions, verified training receipts, tokenizer coverage and measured experiment cost. Decisions about follow-up architectures, crop handling or calibrated combination can now use this evidence. Hidden-domain transfer, training variation and the value of a properly evaluated combination remain unverified.

## Machine evidence

- [Frozen plan](../results/supervised/a_roberta_context_20261109_v1_study/plan.json).
- [Executing-source provenance](../results/supervised/a_roberta_context_20261109_v1_study/provenance.json).
- [All per-fit metrics](../results/supervised/a_roberta_context_20261109_v1_study/analysis/per_fold_metrics.csv).
- [Macro comparisons](../results/supervised/a_roberta_context_20261109_v1_study/analysis/aggregate_metrics.json).
- [Paired bootstrap intervals](../results/supervised/a_roberta_context_20261109_v1_study/analysis/paired_bootstrap.json).
- [Pair-credit and rank-correlation evidence](../results/supervised/a_roberta_context_20261109_v1_study/analysis/pair_complementarity.csv).
- [Clean natural-length and truncation subgroups](../results/supervised/a_roberta_context_20261109_v1_study/analysis/clean_subgroups.csv).
- [Tokenizer coverage](../results/supervised/a_roberta_context_20261109_v1_study/analysis/coverage_summary.json).
- [Native process costs and log hashes](../results/supervised/a_roberta_context_20261109_v1_study/execution_receipts.json).
- [Analysis receipt](../results/supervised/a_roberta_context_20261109_v1_study/analysis/analysis_receipt.json).
- [Final independent verification](../results/supervised/a_roberta_context_20261109_v1_study/FINAL_AUDIT.json).
- [Mechanical source amendments](../results/supervised/a_roberta_context_20261109_v1_study/mechanical_amendments.json).
- [random_0 preserved pre-repair evidence](../results/supervised/a_roberta_context_20261109_v1_study/mechanical_recovery_001/preservation_receipt.json).

Model binaries and native logs are local ignored artifacts. The committed-size evidence records numeric results, IDs, hashes and frozen sources without raw passages. No automatic Kaggle or GitHub action is performed by this report utility.
