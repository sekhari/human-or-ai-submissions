# M4 Pro: refine the new-signal experiments

Public feedback update: the separate full-label raw RoBERTa candidate scored
**0.96970**, according to the user-provided Kaggle screenshot, versus previous
WC120 0.87399. See the [public result](roberta_public_result_20261109.md). The
historical measurement/proposal sections below preserve their original scope;
private-score transfer remains unknown.

Status: **the MPS device benchmark and seven-fit scientific comparison are
complete**. Matched-random clean macro AUROC is 0.996673 versus WC120 0.965848;
topic-proxy clean macro is 0.995652 versus 0.932745. Whitespace normalization
reduces the encoder's corresponding means to 0.847715 and 0.862552. The separate
equal-word diagnostic changes canonical prefix AUROC from 0.782456 to 0.990948
by preserving original whitespace, with no new training. See the
[full comparison](roberta_context_results_20261109.md) and
[prefix diagnostic](roberta_prefix_whitespace_results_20261109.md). These are
development results, not measured leaderboard gains.

The earlier device measurement's three FP32 configurations passed
five warmup plus 20 timed forward/backward/AdamW updates and checkpoint reload
checks using `python_course`. The 512-token, microbatch-4/accumulation-4 run
measured 12.315 documents/s across the complete timed loop and a sampled maximum
MPS driver allocation of 7.343 GiB. No unsupported-op CPU fallback was observed
on the tested paths. See the [device results](roberta_mps_device_results_20261109.md)
for provenance, limits and time projections.

The user confirmed an M4 Pro Mac and prefers comparing potential before
committing resources. The initial read-only local probe verified **48 GiB unified memory,
14 CPU cores, arm64, macOS 15.7.4 and Python 3.12.7**. The inspected environment
did not contain PyTorch or Transformers. At that initial review stage no
package/model was downloaded and no MPS operation was tested.
The observation receipt is
`results/supervised/m4pro_feasibility_20261109/hardware_and_status.json`.

User-selected training environment: **`python_course`**, executable
`/opt/anaconda3/envs/python_course/bin/python`. A subsequent read-only check
found Python 3.11.15, PyTorch 2.11.0, Transformers 5.4.0 and Hugging Face Hub
1.8.0 already installed there. All future model training and MPS probes must
use this environment. The earlier missing-package statement refers only to
the initially inspected Anaconda base interpreter; it does not describe the
selected training environment. No RoBERTa-base cache was found in its configured
Hugging Face cache at the time of that check. The subsequent authorized device
test downloaded the pinned generic model (about 502 MB) to an ignored workspace
cache and confirmed actual MPS execution. The original read-only receipt remains
unchanged.

## Updated choice

Make **RoBERTa-base fine-tuning plus WC120 complementarity** the first main
candidate, with a local causal-LM probability head as an independent signal
branch. DistilRoBERTa is a throughput/reference option rather than a CPU-driven
size ceiling. Keep DeBERTa-v3-base as a second architecture challenger if the
first route provides useful evidence. This ordering is an engineering judgment,
not a claim that RoBERTa or DeBERTa will achieve a particular hidden score.

The executed comparison now supports the raw encoder's additional ranking
signal. The next focused training variable is whitespace normalization with the
remaining recipe held fixed; its quality effect is still untested. Preserve the
raw model as a measured tradeoff rather than rejecting or selecting it using an
arbitrary clean-loss/stress threshold. One scientific round required 109.79
minutes of training loops and 36.60 minutes of evaluation; its elapsed span,
including a documented evaluation-conversion repair, was 152.75 minutes. The
separate zero-fit prefix diagnostic took 87.75 seconds. Sampled training/epoch
checkpoint driver allocation reached 7.616 GiB; unified-memory measures overlap.

[Apple's PyTorch guide](https://developer.apple.com/metal/pytorch/) describes
Apple Silicon GPU training through MPS. The machine meets its published OS and
architecture prerequisites. Actual backend availability and the tested FP32
RoBERTa operator path have now passed the runtime check. Follow
[PyTorch's MPS notes](https://docs.pytorch.org/docs/2.11/notes/mps.html), not CUDA
configuration instructions. Shared memory capacity alone does not establish
training speed or peak-memory requirements.

| Model/route | Role | Why it is worth evaluating |
| --- | --- | --- |
| `FacebookAI/roberta-base` | First full contextual classifier | Generic self-supervised encoder; introduces representation absent from existing TF-IDF/SVD models |
| `microsoft/deberta-v3-base` | Second encoder challenger | Different attention/pretraining architecture; test compatibility and task benefit rather than extrapolating NLU rankings |
| `distilbert/distilroberta-base` | Throughput/frozen-feature reference | Smaller contextual model if measured full-base throughput is unattractive |
| `openai-community/gpt2` | Probability-feature implementation reference | Small generic causal LM for checking shifted token likelihoods, masking and rank statistics |
| `Qwen/Qwen2.5-0.5B` base | Modern probability-signal challenger | Another pretrained next-token distribution, without training a large model or prompting it for authorship labels |

Model cards: [RoBERTa](https://huggingface.co/FacebookAI/roberta-base),
[DeBERTa](https://huggingface.co/microsoft/deberta-v3-base),
[DistilRoBERTa](https://huggingface.co/distilbert/distilroberta-base),
[GPT-2](https://huggingface.co/openai-community/gpt2),
[Qwen base](https://huggingface.co/Qwen/Qwen2.5-0.5B).
Use a generic base language checkpoint, not a ready-made detector trained on
external AI/human labels. Record model revision/license before execution and
use only the supplied labels for supervised fitting.

## Memory: feasibility evidence, not a benchmark

For ordinary FP32 AdamW, weights, gradients and two optimizer moments alone
require roughly 16 bytes per trainable parameter:

| Encoder size | Persistent-state theoretical lower bound |
| --- | ---: |
| DistilRoBERTa, about 82M | 1.22 GiB |
| RoBERTa-base, about 125M | 1.86 GiB |
| DeBERTa-v3-base, about 184M total | 2.74 GiB |

These exclude activations, temporary buffers, optimizer implementation overhead,
framework caches and the OS. They support investigating these sizes on this
48-GiB machine; they are not peak-memory or batch-size guarantees. DeBERTa's
86M backbone figure excludes approximately 98M embedding parameters. Do not
confuse quantized-inference memory with full-parameter training memory.

## First measurement, before scientific model selection

Use the selected `python_course` neural-model environment to preserve the current
classical requirements. Pin installed/tested versions and checkpoint revision. First
check `torch.backends.mps.is_built()` and `.is_available()`, then run a complete
classification forward, loss, backward and AdamW update. A successful tensor
allocation alone is insufficient.

Begin in FP32 and a fresh process with microbatch 2, fixed length 256. After it
works, measure microbatch 4, fixed length 512. For each configuration, use five
warmup and 20 timed updates, synchronizing MPS before/after timing. Record:

- Finite loss/gradients, device placement and any CPU fallback/operator warning.
- Documents and effective tokens per second, including gradient accumulation.
- MPS allocated/driver memory and process RSS; record observations rather than
  treating nominal 48 GiB as free memory.
- Save/reload inference agreement in evaluation mode.

Use fixed padding for the first measurements. Test mixed precision only after
the FP32 path works; do not presume CUDA kernels or precision behavior.
If a model needs fallback, measure the actual complete path. A fallback is an
engineering observation, not proof that the model family lacks scoring value.
No task-validation labels choose the hardware configuration.

Estimate training cost from measured throughput. A proposed initial recipe of
2 epochs, effective batch 16 and 8,000 originals is about 1,000 optimizer updates,
plus evaluation. For the existing two three-fold schemes, the six fits would
require approximately 4,008 updates with their actual sizes. Gradient
accumulation makes each optimizer update contain several microsteps, so a
forward-only timing is not a training-time estimate. The completed direct
effective-batch-16 measurement projects 21.65 minutes for the 8,000-original
two-epoch training loop and 86.79 minutes for the six CV training loops. These
exclude setup, evaluation, checkpoint and calibration costs and do not establish
full-epoch thermal behavior; see the linked measurement record.

## Scientific comparison after feasibility

The first encoder recipe can start from 2 epochs, learning rate 2e-5,
weight decay 0.01, seed 42, effective batch 16 and maximum 512 tokens. These are
proposal values to freeze before task outcomes, not known optimal settings.
Record tokenizer-length coverage; do not silently turn every long passage into
the prefix100 condition from the previous studies. Use the final prescribed
epoch rather than selecting an epoch on outer held labels.

Evaluate standalone and combination behavior using the already defined,
count/class-matched random and topic-proxy folds. An initial implementation
trial may establish feasibility, but scientific conclusions require reporting
the complete prescribed comparison, including unfavorable folds. A frozen
embedding plus logistic-regression probe is a useful reference; its failure
does not rule out supervised fine-tuning of the same encoder.

Keep clean/topic AUCs, natural-length performance, paired rank errors and cost
as continuous evidence. Do not require standalone clean AUC >0.970006 or revive
the arbitrary 0.003 loss rule. Preserve tradeoffs that are not dominated by
another model, and assess whether a new signal repairs WC120's difficult
rankings without creating more damaging new errors.

**Score-scale control is required.** Existing v6 outer-fold WC120 files contain
raw SVM margins. Do not average those directly with encoder softmax probabilities.
Either first report error/rank complementarity without a deployable blend claim,
or produce training-only OOF sigmoid calibration inside each outer fitting
boundary before one fixed probability combination. Learned combination weights
also require an evaluation boundary outside the weight fitting; using all 8,000
labels for weights and then scoring topic-held subsets would leak those labels.
Do not replace this with a percentile transform fitted on held score cohorts.

## Independent probability branch

For a generic causal LM, compute shifted next-token log probabilities on ordinary
passages without a chat template or generated authorship answer. Mask padding,
exclude undefined first-token targets and normalize summaries by valid token
count. Extract a small fixed feature block: NLL mean/variation/quantiles and
probability-rank fractions. Fit scaling and a simple supervised head inside
fitting boundaries. Compare model-specific features through the head, not raw
perplexity numbers from incompatible tokenizers.

GPT-2 is a useful cheaper implementation/throughput reference. Qwen 0.5B is a
separate modern-model test, not an assumed improvement; vocabulary and logits
cost can matter as much as parameter count. MLX-LM is an optional generative
inference path on this Mac, but the first encoder can use standard
PyTorch/Transformers without model-conversion work. The
[official MLX-LM repository](https://github.com/ml-explore/mlx-lm) focuses on
local LLM inference/fine-tuning; backend changes should not be mixed into the
first scientific comparison unnecessarily.

## What is now shared and what remains unknown

Known: M4 Pro with 48 GiB executes complete FP32 RoBERTa training updates at the
measured cost, so the local contextual encoder route is feasible. Existing classifiers
do not already cover pretrained contextual representation. Selection should
measure new signal and complementarity rather than only another clean-score
threshold.

Known: complete two-epoch FP32 encoder fits and five-condition evaluation run
locally at the measured costs; clean rankings improve in all six existing CV
folds, while whitespace sensitivity is substantial and variable across folds.
The same 100-word contrast demonstrates a large input-format effect without
changing lexical content or model parameters.

Unknown: exact unsampled memory peaks, other model/precision paths, training
variation, whether normalization during fitting retains the gains, whether a
properly calibrated combination helps, and transfer to hidden domains/private
scores. Causal-LM features remain an unexecuted independent branch.
