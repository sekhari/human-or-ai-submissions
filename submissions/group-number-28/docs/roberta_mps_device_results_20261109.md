# RoBERTa-base MPS device measurement — November 9, 2026

**Complete: full FP32 forward, backward and AdamW updates work on this M4 Pro.**
Each of three fresh processes ran five warmup updates and 20 timed optimizer
updates. Encoder and classifier parameters changed, losses/gradient norms were
finite, and checkpoint reload reproduced the two training-example evaluation
probabilities exactly. This is a device benchmark; no validation/test AUROC or
new competition candidate was produced.

Study: `results/supervised/a_roberta_mps_20261109_v1/`.
Worker: `experiments/supervised/benchmark_roberta_mps.py`.
Independent, zero-fit evidence check: `experiments/supervised/verify_roberta_mps.py`.

## Environment and input boundary

- Apple M4 Pro, 48 GiB unified memory, 14 CPU cores, macOS 15.7.4.
- User-selected interpreter: `/opt/anaconda3/envs/python_course/bin/python`,
  Python 3.11.15; PyTorch 2.11.0, Transformers 5.4.0, Hub 1.8.0,
  tokenizers 0.22.2, NumPy 2.4.3, safetensors 0.7.0, psutil 7.0.0.
- MPS built/available: true. The observed recommended MPS memory limit was
  36 GiB; this is not a measurement of free physical memory.
- Generic `FacebookAI/roberta-base`, pinned revision
  `e2da8e2f811d1448a5b465c236feacd80ffbac7b`. Seven selected checkpoint/tokenizer
  files total 501,538,643 bytes (about 502 MB decimal), including
  498,818,054 bytes of safetensors weights. Download receipt includes hashes.
  The cache is local under ignored `tmp/huggingface/hub/`.
- FP32, SDPA attention, seed 42, AdamW learning rate 2e-5, weight decay 0.01,
  foreach/fused false, gradient clipping norm 1.0. Ten PyTorch CPU threads were
  observed; CPU thread count was not tuned.
- Real text from deterministic pools of canonical training originals only:
  128, 128 and 400 originals respectively. No held/test text was passed into
  the model. Fixed padding and truncation at 256/512 tokenizer tokens.
  The randomly initialized classification head is expected when converting
  a generic masked-language checkpoint to this two-class task.

## Measured training performance

An update includes zeroing gradients, CPU batch slicing, transfer to MPS,
forward/loss, backward, clipping and optimizer update. MPS synchronization
surrounds the complete update. Numeric checks/logging and memory sampling also
contribute to the separately measured complete 20-update loop.

| Run | Microbatch | Max tokens | Accumulation | Effective batch | Mean update, s | 20-update loop, s | Documents/s, loop | Effective tokens/s, loop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `b2_l256_a1` | 2 | 256 | 1 | 2 | 0.114534 | 2.304521 | 17.357 | 3,945 |
| `b4_l512_a1` | 4 | 512 | 1 | 4 | 0.353485 | 7.082531 | 11.295 | 3,164 |
| `b4_l512_a4` | 4 | 512 | 4 | 16 | 1.298115 | 25.984943 | 12.315 | 3,537 |

Effective tokens are attention-mask counts, including special tokens and
excluding padding. Fixed padded tokens still incur model computation. Different
lengths/pools prevent treating document throughput as an isolated batch-size
comparison. The accumulated run directly measures the proposed effective batch
16; no multiplication of a forward-only or single-microbatch time was used.
Its 20 timed updates executed 80 forward/backward microsteps on 320 documents.

Per-phase fields in `steps.csv` are host enqueue durations, not separate GPU
phase execution times. Model/tokenizer setup, tokenization, evaluation and
checkpoint I/O/reload are outside training timing. Total fresh-process wall
times were 13.71/16.84/39.17 seconds, including warmup and those short-run checks;
their residual overhead is not a full-dataset evaluation measurement.

## Memory and CPU fallback observations

These are **observed maxima sampled during warmup and training**, not exact
peaks and not checkpoint/reload peaks. GiB means bytes divided by 2^30.

| Run | MPS tensors, GiB | MPS driver allocation, GiB | Process RSS, GiB | Whole-process RSS high-water mark, GiB |
| --- | ---: | ---: | ---: | ---: |
| `b2_l256_a1` | 2.240 | 3.473 | 0.794 | 2.083 |
| `b4_l512_a1` | 4.023 | 6.272 | 0.954 | 2.078 |
| `b4_l512_a4` | 5.894 | 7.343 | 0.999 | 2.246 |

RSS and MPS allocation overlap in unified memory; **do not add the columns**.
The lifetime RSS high-water mark has a different scope, including setup and
save/reload. Memory sampling was every 0.1 seconds plus per-update observations,
with no sampling errors. System swap was already 799,801,344 bytes at the start
and stayed unchanged in all three sampled runs. Minimum system-available memory
was about 7.35 GiB in the accumulated run; background applications were not
controlled. Nominal 48 GiB must not be treated as unused memory.

`PYTORCH_ENABLE_MPS_FALLBACK=0` was set before importing PyTorch. Parameters,
gradients and AdamW floating moment tensors were verified on MPS. Python
warnings and native stdout/stderr contained no unsupported-op fallback matches.
The supported conclusion is **no unsupported-op CPU fallback was observed on
these exercised paths**. Ordinary CPU tokenization, dispatch, scalar diagnostics
and AdamW step counters remain expected. This is not a claim of zero CPU work or
an operator-by-operator GPU profile. PyTorch has explicitly registered CPU
fallbacks for some operations even when generic fallback is disabled.

The accumulated timed loop used 3.073 process CPU-seconds over 25.985 wall
seconds. This measures ordinary process CPU activity; it does not independently
attribute operators to devices.

## Time projections and their boundary

Use the accumulated run's complete-loop rate:

`25.984942792 / 20 = 1.299247140 seconds per optimizer update`.

For two epochs and effective batch 16, assuming accumulation resets each epoch
and the final partial batch is retained, updates are
`2 * ceil(number_of_training_originals / 16)` per fit. Each scientific fit must
start from the generic checkpoint, not these device-test weights.

| Proposed work | Optimizer updates | Estimated training-loop time |
| --- | ---: | ---: |
| Canonical 8,000-original fit, two epochs | 1,000 | 21.65 min |
| Six existing matched-random/topic CV fits, two epochs each | 4,008 | 86.79 min |
| Six CV fits plus the canonical 8,000-original fit | 5,008 | 108.44 min (1 h 48 min) |
| Optional later 10,000-label submission refit, two epochs | 1,250 | 27.07 min additional |

The six fitting sizes are 5,394 / 4,646 / 5,960 for each of the two schemes,
reconstructed by the independent verifier from the existing frozen
`a_q5_crop_revised_20261109_v6_study/fold_counts.csv`. This projects one encoder
recipe, not multiple architectures/seeds or calibration fits. If both the
canonical fit and the later submission refit are included, total projected
training-loop time is 135.51 minutes.

**These are training components of a full experiment, not measured end-to-end
experiment durations.** Add model loading, full-data tokenization, held-condition
prediction, metrics/calibration/blending, checkpoint writes and verification.
The exact evaluation schedule has not been executed. Twenty updates do not
establish long-run thermal behavior or full-epoch memory stability; no timing
confidence interval is claimed. A planning allowance is not a measured result.

## Evidence and next decision

All three independent checks passed (429/429/430 checks) without loading a model
or running another fit. They reconstruct timing/token math and sampled memory,
check canonical training IDs and their per-update order hashes, source/model
hashes, device/update checks, reload consistency and native/Python fallback
logs. The reload comparison covers two training examples; it does not establish
held-out performance or full inference throughput.

- `device_plan.json`, `model_download.json`, `execution_sources.json` and
  `execution_receipts.json`: frozen settings, provenance and actual commands.
- Each run: `steps.csv`, `memory_samples.csv`, `metrics.json`,
  `validation_receipt.json`, `fallback_receipt.json`, `warnings.json` and
  `independent_verification.json`.
- `time_projection.json`: original post-run projection;
  `measurement_summary.json`: consolidated measured values and derived scope
  totals. Neither replaces the per-step evidence.
- `model.pt` and native logs remain ignored local artifacts. Selected IDs and
  numeric outputs contain no raw passages. The weights are for device checks
  only and must not warm-start the scientific comparison.

Decision: the measured FP32 512-token, microbatch-4/accumulation-4 route is
practical enough to evaluate locally; these observations do not require a cloud
GPU or prove another model would be faster/better. Freeze the scientific recipe
and tokenizer-length coverage before comparing RoBERTa with WC120 on all
prescribed folds. Standalone score, ranking complementarity and hidden-domain
transfer remain unknown. A calibrated blend would require its own training-only
calibration boundary and additional cost.
