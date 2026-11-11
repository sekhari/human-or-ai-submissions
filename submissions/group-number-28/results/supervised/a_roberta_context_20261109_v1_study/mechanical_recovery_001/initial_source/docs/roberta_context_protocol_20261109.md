# Fixed contextual-encoder comparison — November 9, 2026

The user authorized scientific execution after the successful M4 Pro device
measurement. This protocol is fixed before any scientific RoBERTa training or
held-model prediction. It evaluates whether one generic contextual encoder,
fine-tuned on supplied labels, improves ranking and contributes different useful
errors from WC120. It does not isolate the causal effect of pretraining from all
other differences between the model families.

Machine-readable plan and prepared ID/baseline files:
`results/supervised/a_roberta_context_20261109_v1_study/plan.json`.

## Fixed recipe and complete fitting scope

Use `/opt/anaconda3/envs/python_course/bin/python`, the tested FP32 MPS/SDPA
path, and the pinned generic `FacebookAI/roberta-base` revision
`e2da8e2f811d1448a5b465c236feacd80ffbac7b`. Its model-card license is MIT.
Each fit starts from this generic checkpoint, with a newly initialized
two-class head; device-test or preceding-fold weights are never reused.

- Two complete epochs; seed 42; shuffle training originals deterministically.
- Fixed padding/truncation at 512 tokenizer tokens, microbatch 4, gradient
  accumulation 4 (effective batch 16).
- AdamW learning rate 2e-5, constant schedule, weight decay 0.01 on all trainable
  parameters, foreach/fused false; gradient clipping norm 1.0. These preserve
  the measured optimizer/device path rather than introducing a new search.
- Retain every training original in each epoch. For a final partial effective
  batch, weight each microbatch mean loss by its actual size divided by the
  actual effective-batch size. No underweighted partial updates or dropped IDs.
- Final prescribed epoch only. No held loss between epochs, early stopping,
  epoch selection, architecture/learning-rate/seed search or score-based retry.

Run all seven fits, regardless of favorable or unfavorable quality:

| Fits | Fitting originals | Held originals | Total updates, two epochs |
| --- | --- | --- | ---: |
| Three exact matched-random folds | 5,394 / 4,646 / 5,960 | 2,606 / 3,354 / 2,040 | 2,004 |
| Three exact lexical-topic folds | 5,394 / 4,646 / 5,960 | 2,606 / 3,354 / 2,040 | 2,004 |
| Canonical development fit | 8,000 | 2,000 | 1,000 |

The existing crop-revision folds, topic assignments and canonical split are
reused exactly, with hashes. Topic groups are lexical content/style proxies,
not verified source or generator labels. The topic construction saw all 8,000
development texts, and these splits have already been inspected in earlier
studies. Neither CV nor the reused canonical development set is newly untouched
confirmation. No previous grouping/clean-score thresholds are reinstated.

## Measurements and comparisons

After the final epoch, infer on the same five primary held conditions used by
the historical fold baseline: clean, quote-and-dash, digit-runs-to-zero,
whitespace-only, and prefix100 for originals with at least150 whitespace words.
Definitions are reused from `crop_evaluation.make_conditions`, without opening
suffix/middle conditions. Every condition remains paired by original ID.

Use the class1-minus-class0 **raw logit margin for ranking/AUROC**, avoiding
artificial ties when FP32 softmax probabilities saturate. Also save P(AI=1), with
finite/orientation checks. The six historical WC120 raw-margin predictions are
reused; the saved 8,000-trained WC120 model supplies canonical held inference
without refitting. Its cached clean/stress predictions reproduce exactly.

Primary evidence: all three within-fold clean AUROCs and their equal-weight
mean in each scheme, plus paired RoBERTa-minus-WC120 differences. Do not pool raw
scores from separate fitted folds to create a primary AUC. Secondary evidence:
the four fixed sensitivity conditions, natural-length bins learned only from
each fit's training texts, full tokenizer-length coverage and truncation
subgroups, within-fold rank correlation, and corrected/newly damaged
human-versus-AI pair rankings (ties count0.5). Different errors alone do not
prove a useful fusion.

Compute1,000 paired bootstrap replicates using original IDs within each random
held fold, and whole frozen topic clusters within each topic held fold. Use
seeds548/549/550 for random/topic/canonical. Keep undefined single-class draws,
report valid counts, and do not redraw to obtain a desired result; intervals
are unavailable below90% valid draws. These are descriptive conditional
intervals, omitting model-training variation, selection and multiplicity
adjustment. Overlapping training folds are not independent replications.

No direct averaging of SVM margins with encoder probabilities, held-cohort rank
normalization, learned blend weights, calibration fits or candidate export is
part of this comparison. A later deployable blend needs training-only OOF
calibration/weight fitting assessed outside that fitting boundary.

## Execution evidence and recovery

Record exact fitting/held IDs, source/model/plan hashes, versions, complete
epoch coverage and optimizer updates, finite losses/gradient norms, device
placement, CPU fallback warnings, observed MPS/RSS memory, training/evaluation
time and checkpoint reload agreement. Tokenizer coverage of held inputs is
computed only after training; no held text informs model fitting. Raw passages,
token sequences, checkpoints and caches remain local and ignored.

Stop only for mechanical, numeric, data or device failures. Preserve failed
attempts and native logs; resume the identical unfinished recipe from a complete
epoch checkpoint when supported. Record any repair explicitly, without changing
the scientific comparison because of scores.

The measured device rate projects about108.44 minutes of training for all seven
fits. This excludes full-data tokenization, inference on five conditions,
checkpointing and analysis. End-to-end duration and thermal stability are now
measurements to collect, not claims established by a20-step benchmark.

## What this round will settle

New assets will include a fixed contextual candidate, complete paired held
predictions, ranking-complementarity evidence and measured full-process cost.
Whether any observed development gain transfers to the private competition
distribution will remain unknown. Public WC120 score0.87399 remains the current
external reference; no hidden score is inferred from this experiment.
