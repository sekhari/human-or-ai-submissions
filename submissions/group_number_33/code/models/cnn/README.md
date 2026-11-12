# CNN experiments

All CNN source code and documentation live in this package:

```text
models/cnn/
  model.py          # TextCNN and the shared epoch/prediction functions
  neural.py         # Tokenization, embeddings, training, and checkpoints
  generalization.py # Splits, experiment runner, and submission generation
```

Run commands from the repository root using `python -m models.cnn`. Input datasets and
downloaded embeddings stay in `data/`. All CNN-generated splits, checkpoints,
metrics, histories, analysis reports, and submission CSVs go into `models/cnn/outputs/`
by default. This output directory is ignored by Git; `--output-dir` can override
it. Older standalone histories/submissions are in `models/cnn/outputs/legacy/`, and
initial review results are in `models/cnn/outputs/review/`.
The default experiment matches the submitted configuration: raw text with
newline tokens, 768-token sequences, 100-dimensional random embeddings, and the
fixed stratified 8,000/2,000 split. Model settings live in `neural.CNNConfig`;
`model.py` contains no separate training program or conflicting settings.

## Neural generalization benchmark

`generalization.py` trains the raw-text PyTorch `TextCNN` by default. An optional
cleaned-text variant remains available for controlled comparisons:

- `cnn_raw`: preserve case, punctuation, URLs, numbers, and explicit newline tokens.
- `cnn_clean`: run the existing cleaner, preserving case, then use the same tokenizer.

Both use token IDs, learned embeddings, convolution widths 3/4/5, max pooling,
dropout, and a binary output. TF-IDF is used only to construct topic groups;
it is not fed into either CNN. No logistic regression is used. The architecture
and optimizer defaults are configured in `neural.py`. Both benchmark variants use a sequence
limit of 768 tokens (configurable) to accommodate newline tokens. Runs truncate
longer sequences, and horizontal whitespace runs are not encoded. No external
data or pretrained models are required. The dependencies are the existing
PyTorch, NumPy, pandas, and scikit-learn environment.

Run these commands from the repository root in your terminal. Use `python3`
instead of `venv/bin/python` if your environment is already activated.

First, create and inspect the fixed splits:

```sh
venv/bin/python -u -m models.cnn prepare
```

This writes `models/cnn/outputs/splits.csv`, `topics.csv`, and
`split_metadata.json`. The random split uses the same stratification, seed, and
validation IDs as `preprocess.py`. The second protocol clusters texts into 25
topic proxies and assigns entire clusters to five held-out folds. Inspect the
keywords and human/AI counts in `topics.csv` before interpreting the results.
Topic construction uses word TF-IDF, SVD, and clustering on all labeled texts
without using labels as clustering features; labels are used to balance folds.
These are split-design features, not detector features. Neither `test.csv` nor
`unlabeled.csv` is used. Each detector fits its own vocabulary only on its
training fold.

To run the submitted configuration on the required fixed random holdout:

```sh
venv/bin/python -u -m models.cnn benchmark
```

This trains on 8,000 rows and monitors AUROC on the fixed 2,000-row validation
partition after each epoch. It saves each new best checkpoint and stops after
three epochs without improvement. The validation score therefore participates
in epoch selection; the hidden Kaggle test set remains the final evaluation.
To compare raw and cleaned text explicitly, add
`--models cnn_raw cnn_clean`.

For a supplementary generalization experiment, run the raw CNN on a topic holdout:

```sh
venv/bin/python -u -m models.cnn benchmark --protocol topic --fold 0 --models cnn_raw
```

To extend that supplementary experiment, evaluate all five topic folds. These
use different validation rows and do not replace the required fixed-holdout
comparisons. Skip repeating the random split with `--protocol topic`:

```sh
venv/bin/python -u -m models.cnn benchmark --protocol topic --fold all
```

Topic evaluations use the selected topic fold directly for early stopping and
reporting. The remaining topic folds supply the training rows. As with the
random benchmark, the saved model is the checkpoint with the highest validation
AUROC before patience is exhausted.

Use `--epochs 15 --patience 3` to cap the selection phase at 15 epochs. Other
options include `--max-length`, `--max-vocab`, `--batch-size`, `--learning-rate`,
`--seed`, and `--device auto|cpu|mps|cuda`. Automatic device selection prefers
CUDA, then Apple MPS, then CPU. Seeds are set for each fit; results can still
vary across devices and library versions.

Results go into a new `models/cnn/outputs/cnn_benchmark_<protocol>_<fold>_<timestamp>/`:

- `metrics.csv`: training/validation AUROC, training time, log loss, selected epoch,
  and topic summaries for each evaluation.
- `summary.csv`: mean training/validation AUROC, mean training time, minimum fold
  AUROC, and median selected epoch for each model/protocol.
- `validation_predictions.csv`: probabilities and original IDs for error analysis.
- `topic_metrics.csv`: performance and class counts for each held-out topic.
- `run_metadata.json`: split fingerprint, versions, and model settings.

Each model/evaluation subdirectory also contains `history.csv`,
`embedding_report.json`, and `model.pt`. Checkpoints
include weights, token vocabulary, preprocessing mode, and model settings.
Metrics and predictions are saved after every completed evaluation, so results
from earlier evaluations survive an interrupted run. Training cannot resume an
unfinished epoch from a checkpoint; completed checkpoints support inference.

Prefer a model that performs consistently across topic folds, then inspect its
weak topics. Topic summaries require at least 20 examples of each class per
topic; small or single-class topics remain visible but are excluded from those
summaries. Minimum topic AUROC can be noisy. The five folds have different sizes,
so the summary reports an unweighted mean of fold AUROCs rather than pooled
predictions from independently trained classifiers.

Topic groups approximate unfamiliar domains. They are not verified source
domains and cannot measure transfer to unseen AI generators, since generator
labels are unavailable. A topic holdout is not guaranteed to be harder than the
random holdout or to predict the Kaggle score. Keep these split IDs fixed while
comparing models; reserve a fresh evaluation if you tune extensively against them.
The script rejects stale splits when the training file or scikit-learn version
changes. Recreating splits with different settings changes the benchmark.

The competition PDF (page 2, Validation) requires a fixed stratified 2,000-row
validation set and up to 8,000 training rows for controlled experiments. It
explicitly permits training the final Kaggle model on all 10,000 labeled rows
after selecting the approach. For the current development stage, keep that
holdout and generate submissions from existing benchmark checkpoints. For the
random protocol, `random_-1_cnn_raw_random/model.pt` was fitted on the 8,000
outer-training rows and excludes the shared 2,000-row holdout. Use the path from
the desired benchmark run:

```sh
venv/bin/python -u -m models.cnn submit \
  --checkpoint "models/cnn/outputs/<your-benchmark-run>/random_-1_cnn_raw_random/model.pt"
```

Replace `<your-benchmark-run>` with the actual directory name. This command is
inference only: it loads saved weights, vocabulary, preprocessing, and dimensions;
it does not read labeled training data or run an optimizer. Full-data refitting
has its own explicit `train-full` command below.

This produces a new `models/cnn/outputs/cnn_prediction_cnn_raw_random_<timestamp>/`
directory containing `submission_cnn_raw.csv` and metadata recording checkpoint
and test-file fingerprints. Predictions use `id,label` columns, original test-row
order, and AI probabilities. The script does not upload to Kaggle.

The random protocol preserves the required shared holdout. Topic-based protocols
use different held-out rows, may train on shared validation IDs, and can have
more than 8,000 training rows. Treat them as supplementary stress tests, not as
the controlled model-comparison or learning-curve experiments required by the
report. Pretrained comparisons for the report must also use `--protocol random`.

## Final training on all labeled data

Use the fixed-holdout benchmark to compare configurations and choose an epoch
count first. Then refit the selected model on all 10,000 labeled passages:

```sh
venv/bin/python -u -m models.cnn train-full --epochs 12
```

The defaults match the CNN that scored approximately 0.94 on the public
leaderboard when trained on 8,000 rows: raw text, random 100-dimensional
embeddings, 768-token sequences, and seed 12. The 12 epochs come from that
configuration's earlier validation run. Replace it with the best epoch from
the benchmark you intend to use. This command trains a fresh network on
every row of `data/train.csv` for exactly 12 epochs, then predicts the 3,000
passages in `data/test.csv`. It does not use the test passages for training and
does not require prepared splits.

A new `models/cnn/outputs/cnn_full_cnn_raw_random_<timestamp>/` directory contains:

- `model.pt`: fitted weights, vocabulary, and configuration.
- `refit_history.csv`: training loss by epoch.
- `training_ids.csv`: every labeled ID used for this fit.
- `run_metadata.json` and `embedding_report.json`: configuration and provenance.
- `submission_cnn_raw.csv` and its metadata: test probabilities in original ID order.

Upload that CSV to Kaggle to measure the full-data model's public AUROC. There
is no held-out validation AUROC for this fit; a training-set score would not
establish which model generalizes best. Required report comparisons still use
the fixed 8,000/2,000 benchmark. The command does not upload automatically.
The full-data training duration is saved as `training_seconds` in
`run_metadata.json` and the submission metadata.

To refit a different selected configuration, use `--model cnn_clean` or
`--embedding-mode frozen|finetune` with the corresponding vector options. For
example, to train the fine-tuned GloVe variant for an already selected 8 epochs:

```sh
venv/bin/python -u -m models.cnn train-full \
  --model cnn_raw --embedding-mode finetune --epochs 8 \
  --embedding-dim 100 --embeddings data/embeddings/glove.6B.100d.txt \
  --lowercase-embedding-fallback
```

Select each configuration's epoch count using its fixed-holdout experiments.
The command requires `--epochs` and does not choose settings from test predictions.
Existing benchmark checkpoints and submissions are preserved.

## Pretrained embedding comparison

The CNN also supports pretrained word vectors, using the same tokenizer and
training-only vocabulary. Use these experiments if the competition rules permit
external pretrained vectors. The default remains random, trainable embeddings.

For a first experiment, use 100-dimensional vectors from the
[Stanford GloVe release](https://github.com/stanfordnlp/GloVe).
[Keras's official example](https://keras.io/examples/nlp/pretrained_word_embeddings/)
also documents the 100-dimensional file in this archive. Run the download and
extraction in your terminal; the archive is approximately 822 MB:

```sh
mkdir -p data/embeddings
curl -fL --retry 3 -o data/embeddings/glove.6B.zip https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
unzip -n data/embeddings/glove.6B.zip glove.6B.100d.txt -d data/embeddings
```

Prepare splits once with `python -m models.cnn prepare` if needed. Then compare all
three embedding treatments on the fixed random holdout, using only raw text initially to
avoid mixing the embedding and cleaning experiments:

```sh
venv/bin/python -u -m models.cnn benchmark \
  --protocol random --models cnn_raw \
  --embedding-dim 100 \
  --embeddings data/embeddings/glove.6B.100d.txt \
  --lowercase-embedding-fallback \
  --embedding-modes random frozen finetune
```

- `random`: initialize vectors randomly and train them, providing a dimension-matched baseline.
- `frozen`: copy available pretrained vectors and freeze the entire embedding table; train the CNN layers.
- `finetune`: copy the same pretrained vectors and train both embeddings and CNN layers.

All three use the same dimensions, splits, model capacity, optimizer learning
rate, and seed. The random baseline must also use 100 dimensions; comparing
pretrained 100-dimensional vectors against the old 128-dimensional baseline
would change two factors at once. Each treatment trains on the same training
partition and independently stops according to the same validation partition.
For an additional
topic stress test, replace `--protocol random` with `--protocol topic --fold all`.

The file reader loads vectors only for tokens in that fit's training vocabulary.
Exact token matches take priority. GloVe 6B is uncased; the optional lowercase
fallback initializes, for example, `Hello` using `hello` when no exact vector
exists. Token IDs remain case-sensitive, but differently cased tokens can start
with identical vectors. Frozen embeddings retain that similarity; fine-tuned
embeddings can diverge. Omit the flag to require exact matches.

Tokens without a vector, including some punctuation and newline tokens, keep
their seeded random initialization. They are also fixed in the frozen treatment.
Padding stays zero. The code prints vocabulary coverage and records exact and
lowercase match counts, vector-file SHA-256, and dimensions in
`embedding_report.json` alongside the checkpoint. This is vocabulary coverage,
not the percentage of running text
tokens covered. A dimension mismatch fails before the benchmark starts training.

Scores in `metrics.csv` and `summary.csv` identify both `model` and
`embedding_mode`. Compare fixed-holdout AUROC across treatments for the required
report, and topic-fold results separately. The code adds no
automatic performance assumptions about pretraining.

To generate a submission from a pretrained benchmark model, pass its saved
checkpoint. The treatment and dimensions are loaded from that checkpoint:

```sh
venv/bin/python -u -m models.cnn submit \
  --checkpoint "models/cnn/outputs/<your-benchmark-run>/random_-1_cnn_raw_finetune/model.pt"
```

Saved checkpoints contain the learned embeddings and vocabulary. Loading them
with `models.cnn.neural.load_predictor(path)` does not require the original GloVe file.
