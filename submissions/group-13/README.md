# Group 13: human versus AI text

**Team:** Sara Wang (sw2529), Jiale Lin (jl4898), Justin Morrill (jm3234).

**Kaggle team:** Group 13.

This folder contains Justin's selected experiments and portable training/prediction entrypoints for integration into the team submission. The strongest reported submission combines RoBERTa, T5 and Qwen at family weights **20/40/40**, with public AUROC **0.98786**. The logistic stack scored **0.98784**. Both recipes, two other strong blends, and the earlier logistic regression, linear SVM and scratch BiGRU experiments are included.

The repository contains source, configuration and aggregate results only. Download competition data separately. Model weights, individual predictions, environments and caches are excluded. No script uploads to Kaggle.

## Setup

Run commands from `submissions/group-13`. Python **3.12** was used.

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux, or `.venv\Scripts\Activate.ps1` in Windows PowerShell. For logistic regression and SVM:

```bash
python -m pip install -r requirements.txt
```

For the neural models, install a PyTorch build appropriate to your hardware, then:

```bash
python -m pip install -r requirements-neural.txt
```

The original pretrained runs used PyTorch **2.9.1 with ROCm 7.2** in Linux/WSL on an AMD GPU with 24 GB VRAM. The tested build identified itself as `2.9.1+rocm7.2.0.lw.git7e1940d4`. Use PyTorch >=2.6,<3 with the pinned Transformers/PEFT packages. CUDA PyTorch is appropriate on NVIDIA; do not replace an already working GPU PyTorch installation with a CPU wheel. `--device cuda` is PyTorch's device name for both CUDA and ROCm. Other hardware may change floating-point results.

Obtain the competition files from Kaggle and extract them locally:

```text
data/train.csv       # id,text,label; 10,000 rows, 0=human, 1=AI
data/test.csv        # id,text; 3,000 rows
```

The selected final recipes do not use the additional unlabeled examples. The starter 2,000 rows are a subset of train.csv and must not be appended. Preserve original CSV row order for the recorded random split. The source competition ZIP had SHA256 `1e602e9811ec88a1c13140c393eed78732679a025c53f8d5f99a357e08ad58f9`.

All output directories must be new: the entrypoints refuse to overwrite existing runs. `artifacts/` and `outputs/` are ignored by Git.

## Logistic regression and SVM

Both selected baselines use lowercase word **and punctuation** tokens, unigrams plus bigrams, minimum document frequency 2, L2-normalized TF-IDF and C=10. Vocabulary and IDF are fitted on training rows only. Logistic regression uses L2 regularization and liblinear. Linear SVM uses squared hinge loss; a three-fold sigmoid calibration fitted within the training subset supplies probabilities. Calibration includes the vectorizer inside each fold and uses seed 44 when the main seed is 42.

Train and evaluate on the stratified 80/20 split (8,000/2,000 rows, seed 42):

```bash
python train.py --train data/train.csv --model logistic --seed 42 --out artifacts/logistic-validation
python train.py --train data/train.csv --model svm --seed 42 --out artifacts/svm-validation
```

Each run writes validation predictions, AUROC/log loss, split IDs and a fitted model locally. The portable commands were rerun during export and reproduced the original AUROCs: **0.959476** for punctuation logistic regression and **0.962387** for punctuation SVM.

Refit on all labeled rows and produce a Kaggle CSV:

```bash
python train.py --train data/train.csv --model logistic --seed 42 --full --out artifacts/logistic-full
python predict.py --test data/test.csv --model logistic --artifacts artifacts/logistic-full --out outputs/logistic.csv
python train.py --train data/train.csv --model svm --seed 42 --full --out artifacts/svm-full
python predict.py --test data/test.csv --model svm --artifacts artifacts/svm-full --out outputs/svm.csv
```

These baseline numbers are local validation scores, not Kaggle scores. Full refits have no independent local validation score.

## Scratch neural network

The scratch model learns a 128-dimensional word/punctuation embedding, runs a bidirectional GRU with 64 units per direction, pools its 128-dimensional token states, then applies a 64-unit hidden classification layer and one binary logit. Mean pooling averages valid token states; learned attention pooling scores each state with a small network and takes their softmax-weighted average. Padding is excluded. The vocabulary is learned from training text only; rare/unseen tokens map to UNK. Text is truncated to 768 tokens.

Defaults are AdamW at 0.001, batch 32, embedding dropout 0.1, classifier dropout 0.3, weight decay 0.0001, gradient clipping 1, and validation-AUROC early stopping with patience 3 and a maximum of 20 epochs.

```bash
python train.py --train data/train.csv --model gru --pooling mean --seed 42 --device cuda --out artifacts/gru-mean-validation
python train.py --train data/train.csv --model gru --pooling attention --seed 42 --device cuda --out artifacts/gru-attention-validation
python train.py --train data/train.csv --model gru --pooling mean --strong-dropout --seed 42 --device cuda --out artifacts/gru-regularized-validation
```

Use `--device cpu` for a CPU run. `--strong-dropout` sets embedding dropout 0.4, classifier dropout 0.6 and weight decay 0.01. The recorded mean validation AUROCs across seeds 42/43/44 were 0.942873 for mean pooling, 0.942205 for attention pooling, and 0.945497 for the stronger-regularized mean model. These are **averages of run scores**, not ensemble scores.

For a full refit, select the epoch count from the validation run's `bigru_<pooling>_seed42/summary.json` (`best_epoch`). Replace `N` below with that integer; keep pooling and regularization identical:

```bash
python train.py --train data/train.csv --model gru --pooling mean --strong-dropout --seed 42 --full --epochs N --device cuda --out artifacts/gru-full
python predict.py --test data/test.csv --model gru --artifacts artifacts/gru-full --device cuda --out outputs/gru.csv
```

## Best pretrained ensemble

There are nine trained members, shared by all final recipes:

| Family | Members | Learning rate | Epochs | Micro batch |
|---|---|---:|---|---:|
| RoBERTa-base | Seeds 42,43,44 | 3e-5 | 3,2,3 | 16 |
| RoBERTa-large | Two fits, seed 42 | 1e-5 and 2e-5 | 3 each | 8 |
| T5-v1.1-base | Seeds 42,43 | 1e-4 | 3 each | 8 |
| Qwen2.5-1.5B | Seeds 42,43 | 1e-4 | 2 each | 4 |

All use max length 512, effective batch 32 via accumulation, AdamW, matrix weight decay 0.01 (zero for bias/vector parameters), a linear schedule with 10% warmup, and gradient clipping 1. RoBERTa-base seed 43 stops after epoch 2 on a three-epoch schedule. BF16 autocast is used when supported. Large RoBERTa and Qwen use gradient checkpointing. Exact source revisions and seeds are frozen in `configs/models.json`; `--seed` changes baseline initialization and random split, not these member seeds.

RoBERTa fine-tunes a two-class sequence head and encoder. T5 fine-tunes the encoder-decoder with dropout zero; one decoder-start position predicts the two vocabulary tokens `human` and `AI`. Their logits are normalized over those two classes. Qwen is a causal decoder: the final real EOS state feeds a new two-class head. Its base weights remain frozen; rank-8 LoRA adapters (alpha 16, dropout 0.05) train in query/value projections, alongside a head with dropout 0.1. Qwen inputs are truncated to 511 tokens and get an explicit EOS, with right padding.

Define R, T, Q as follows:

1. Average the three RoBERTa-base probabilities to get B and the two large probabilities to get L. Then `R = sigmoid(0.5*logit(B) + 0.5*logit(L))`, with probabilities clipped at 1e-12 for this operation.
2. T is the arithmetic mean of the two T5 probabilities.
3. Q is the arithmetic mean of the two Qwen probabilities.
4. The best submission is `0.20*R + 0.40*T + 0.40*Q`.

Validate the fixed member recipes on the random holdout:

```bash
python train.py --train data/train.csv --model ensemble --device cuda --cache-dir .hf-cache --out artifacts/ensemble-validation
```

This writes `recipe_metrics.json` for the fixed blends. The bundled stack is deliberately excluded from these holdout metrics because its final coefficients used all training labels. Epoch choices and recipes were selected during earlier development; a rerun of this familiar holdout is not a fresh unbiased estimate.

Train all nine members on all labeled rows, then generate the best submission:

```bash
python train.py --train data/train.csv --model ensemble --full --device cuda --cache-dir .hf-cache --out artifacts/ensemble-full
python predict.py --test data/test.csv --model ensemble --artifacts artifacts/ensemble-full --recipe best --device cuda --cache-dir .hf-cache --out outputs/submission.csv
```

Weights download from Hugging Face on first use. Pinned sources: [RoBERTa-base](https://huggingface.co/FacebookAI/roberta-base) and [RoBERTa-large](https://huggingface.co/FacebookAI/roberta-large) (MIT); [T5-v1.1-base](https://huggingface.co/google/t5-v1_1-base) and [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) (Apache 2.0). No external labeled training data, paid classification API or hidden test labels are used. Remote model code execution is disabled. After caching sources, add `--offline` to prevent further downloads.

Allow several GB for downloads and additional space for fitted members. Nine-member training is substantial GPU work, potentially hours; it is not needed to read the results or test baseline code. `--micro-batch` can reduce training memory (choose a divisor of 32), but changes numerical reproduction. CPU pretrained training requires explicit `--device cpu` to avoid accidentally starting a very slow run.

## Other leading submissions

Use the same `artifacts/ensemble-full` directory; training is shared. Only `--recipe` and output filename change:

```bash
python predict.py --test data/test.csv --model ensemble --artifacts artifacts/ensemble-full --recipe stack --device cuda --cache-dir .hf-cache --out outputs/stack.csv
python predict.py --test data/test.csv --model ensemble --artifacts artifacts/ensemble-full --recipe t5-qwen --device cuda --cache-dir .hf-cache --out outputs/t5-qwen.csv
python predict.py --test data/test.csv --model ensemble --artifacts artifacts/ensemble-full --recipe equal --device cuda --cache-dir .hf-cache --out outputs/equal.csv
```

| Recipe | Definition | Reported public AUROC |
|---|---|---:|
| `best` | 20/40/40 R/T/Q | **0.98786** |
| `stack` | Logistic stack of R/T/Q logits | 0.98784 |
| `t5-qwen` | 0/50/50 | 0.98709 |
| `equal` | Equal thirds | 0.98641 |
| `midpoint` | Mean of `best` and `stack` predictions | Not reported |

The public scores were reported by the user after manual uploads; they were not independently fetched. They cover approximately 30% of the test rows; private evaluation uses the remaining 70%. The 0.00002 gap between the top two is too small to establish that one generalizes better. Strong grouped local stack scores did **not** yield a public improvement. Repeated public-score-guided choices also risk fitting the public subset.

The frozen stack formula in `configs/frozen_stack.json` is:

```text
sigmoid(-1.9093563623409995
        + 0.39571407289809585 * logit(R)
        + 0.23702473770047208 * logit(T)
        + 0.27049267789642517 * logit(Q))
```

Clip R/T/Q to [1e-6,1-1e-6] first. These coefficients come from logistic regression with C=0.1, balanced class weights and standardized logit features, algebraically converted back to unstandardized logits. They are not percentage blend weights.

## Rebuild the stack without base-model leakage

`configs/stack_splits.json` freezes four partitions of training IDs into `base_train`, `meta`, and `evaluation`, keeping the recorded source/style proxy groups apart. They are proxy groups inferred from available text, not known generator identities. Every split trains nine base members on `base_train` only, fits the stack on `meta` only, then compares it with the fixed blend on `evaluation`.

```bash
python train.py --train data/train.csv --model stack --device cuda --cache-dir .hf-cache --out artifacts/stack-oof
python predict.py --test data/test.csv --model ensemble --artifacts artifacts/ensemble-full --recipe stack --stack-config artifacts/stack-oof/stack.json --device cuda --cache-dir .hf-cache --out outputs/refitted-stack.csv
```

This is **36 additional base fits** and requires the complete original training IDs. Fold checkpoints are removed after verified excluded predictions to limit disk use; add `--keep-fold-models` to retain them. Fold-level scores are saved in `evaluation.json`. For the final stack, predictions made while each ID was excluded from base fitting are pooled across splits, averaging family probabilities per ID before applying logit. Each of the 10,000 unique IDs contributes one final fitting row. This final stack refit has no independent local score. These data had already influenced earlier development, so the grouped checks are stress tests, not a pristine final holdout.

## Original experiments and result records

The four files in `experiments/` preserve the original experiment code unchanged. They cover word-only versus punctuation tokenization, unigrams/bigrams/both at varying C, SVM comparisons, and mean versus attention BiGRU. Their longer module comments explain the models and checks. To rerun the complete historical sequence, place the original ZIP in `data/`, enter `experiments/`, and run in dependency order:

```bash
python logistic_regression_experiments.py --zip ../data/cornell-tech-cs-5785-combined-2026-ai-vs-human-text.zip
python punctuation_experiments.py --zip ../data/cornell-tech-cs-5785-combined-2026-ai-vs-human-text.zip
python svm_experiments.py --zip ../data/cornell-tech-cs-5785-combined-2026-ai-vs-human-text.zip
python neural_network_experiments.py --zip ../data/cornell-tech-cs-5785-combined-2026-ai-vs-human-text.zip --device cuda
```

The later historical scripts check earlier output files, so this order matters. The portable root entrypoints avoid those historical dependencies when running a single selected model. `records/` contains aggregate sweep results, grouped checks, submission hashes and provenance. The original word-only logistic best AUROC was 0.935516; punctuation improved it substantially. The scratch network did not outperform the sparse SVM here, despite its greater flexibility.

## Verification and limits

Quick checks, requiring no competition data or downloaded models:

```bash
python -m unittest discover -s tests -v
python tests/check_neural.py
```

The second command requires neural dependencies but runs tiny random networks on CPU. It checks BiGRU padding/attention/gradients, all transformer output shapes and backward passes, Qwen padding invariance and frozen-base versus LoRA gradients.

During export we also reran both full 8,000/2,000 baseline evaluations and generated checked 3,000-row predictions; ran the scratch GRU train/save/load/predict path on a 20-row smoke sample; and reconstructed all five ensemble recipes from the original saved member predictions. Maximum probability differences from the original CSVs were below 2.1e-13. `records/export_verification.json` stores aggregate comparisons only. Smoke results are functionality checks, not model-quality estimates.

The portable transformer trainer consolidates the original recipes and was checked against the original code/configuration. The complete nine-model GPU training and 36-fit stack rebuild were **not rerun just for this export**. The formula comparison verifies the blend reconstruction, not byte-identical end-to-end retraining. Different hardware, libraries, batch sizes and random-number behavior can change retrained predictions; the reported public scores belong to the original CSV hashes.

Prediction output is exactly `id,label`, where `label` is probability of AI. The command validates row count, finite [0,1] values and original ID order, and writes a provenance manifest. `--allow-small-test` and training `--limit` exist solely for smoke tests.

## AI assistance and development history

ChatGPT/Codex was used extensively for model explanations, experiment design, code generation and debugging, experiment execution, analysis, and documentation, including this export. The user directed the work and manually submitted files to Kaggle and reported the public results. Automated verification performed during export is listed above; this does not claim that every line or scientific conclusion has already been independently checked by every teammate. The team should reconcile this disclosure with the report's Q6.3 and describe its own manual review accurately.

Original experiments were developed locally before this repository import. `records/source_provenance.json` records hashes of the preserved original scripts. Commits here represent the actual curation, portable-entrypoint integration, and documentation work; they do not reconstruct or invent earlier Git history. Coordinate the course's single team PR with the other contributors before submission.
