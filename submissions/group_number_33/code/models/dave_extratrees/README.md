# Dave's retained model: 145 style features + 800 ExtraTrees

For the course submission, start with the [team README](../../../README.md). Its separate `train.py` and `predict.py` entrypoints reuse this exact recipe. The combined runner documented below is retained for historical reproduction; its commands run from `submissions/group_number_33/code/`, not the instructor repository root.

This directory reproduces Dave's selected AI-versus-human text classifier. It converts each passage into **145 handcrafted writing-style and structural features**, then uses an **800-tree ExtraTreesClassifier** to predict the probability of AI authorship.

Dave reported that this version improved the Kaggle public AUROC by **0.004** over his preceding submission. This is a reported change, not an exact public score or a prediction of private-leaderboard performance. The local historical validation AUROC is **0.9956835**.

This is the retained ExtraTrees method. The earlier MultinomialNB/TF-IDF method is a separate experiment. The code here runs the fixed selected recipe; it does not run a new hyperparameter search.

The report's training-size and unlabeled-data experiments are organized in
[`q4_q5/`](q4_q5/README.md). They reuse this recipe and the fixed 2,000-row
validation set, with results under `outputs/q4_q5/`.

## 1. What data do we use?

| File | Role in this method |
|---|---|
| `train.csv` | 10,000 labeled passages used for validation experiments and final training. Columns: `id`, `text`, `label`. |
| `test.csv` | 3,000 passages to predict for Kaggle. Columns: `id`, `text`. |
| `sample_submission.csv` | Defines the submission format and expected ID order. |
| `unlabeled.csv` | Not used to train this selected model. |
| `train_starter_2000.csv` | Not used here; it is a subset of `train.csv`, not additional data. |

Labels are **0 = human-written** and **1 = AI-generated**. The model receives features calculated from `text`; `id` only matches predictions to passages, and `label` is the training target. Neither is included among the 145 features. No external LLM API, external labels, or hidden test labels are used.

## 2. The complete workflow

```text
train.csv: 10,000 labeled passages
    |
    +-- Fixed stratified split, random_state=12
    |       |
    |       +-- 8,000 training passages --> 8,000 x 145 feature matrix
    |       |                                  |
    |       |                           Train the selected ET800 model
    |       |                                  |
    |       +-- 2,000 validation passages --> 2,000 x 145 feature matrix
    |                                          |
    |                                  Predict and measure AUROC
    |
    +-- Final submission training: same recipe, all 10,000 labeled passages
            |
            +-- 10,000 x 145 feature matrix --> Refit ET800
                                                   |
test.csv: 3,000 passages --> 3,000 x 145 features --> AI probabilities
                                                   |
sample_submission.csv: ID order ----------------> submission.csv
```

The validation model and the final submission model are **two separate fits**. The final model's use of all 10,000 labels does not turn its training results into a validation score.

## 3. Text processing: preserve the original writing

We keep capitalization, punctuation, digits, spaces, tabs, and line breaks. We do not remove stopwords, stem or lemmatize words, or lowercase the entire original text. These details are inputs to the style statistics, so deleting them would remove information we intend to measure.

The extractor performs lightweight parsing:

- **Words:** a regular expression recognizes Unicode word characters and internal straight or curly apostrophes, so contractions can remain single tokens.
- **Sentences:** heuristic units split on sentence-ending punctuation or newlines; empty units are dropped. This is a practical counting rule, not a full linguistic sentence parser.
- **Paragraphs:** blank lines separate paragraphs.
- **Repetition and vocabulary diversity:** tokens are internally case-folded, so `Hello` and `hello` count as the same word for these statistics. Original capitalization remains available for other features.

The final classifier uses **no TF-IDF representation, learned vocabulary, embedding, or feature scaler**. Some features are naturally ratios, such as punctuation count divided by character count. Those ratios are calculated within each passage.

## 4. How one passage becomes 145 numbers

Each feature is a fixed calculation. One passage becomes one row; each of the 145 columns has the same meaning for every passage.

| Feature family | Examples | What it describes |
|---|---|---|
| Size | Character, token, sentence, paragraph, and line counts | Passage length and organization |
| Diversity and repetition | Unique-token ratio, repeated-token fraction, repeated adjacent words and word sequences, entropy | How varied or repetitive the passage is |
| Word and sentence lengths | Means, standard deviations, minima, maxima, medians, and other percentiles | Length patterns and their variation |
| Character use | Uppercase, lowercase, digit, non-ASCII, and mixed letter/number proportions | Character-use habits |
| Punctuation | Counts and rates for periods, commas, quotes, brackets, dashes, and other symbols | Punctuation patterns |
| Layout and formatting | Blank lines, indentation, whitespace runs, headings, lists, quotations, and code fences | Layout and document structure |

For example, `Hello hello!` has 12 characters, 2 tokens, 1 unique case-folded token, a unique-token ratio of 0.5, and 1 exclamation mark. These are several columns within the full feature vector.

The number **145** is the total after expanding these rules into individual measurements. It is not an automatically discovered optimal number. We defined the measurements, while the classifier learns how to combine them using labeled training examples.

These features were chosen to test whether writing style and structure provide useful signals while limiting dependence on particular topic words. They are interpretable and produce a compact, fixed-size input. They are hypotheses to evaluate, not universal rules about AI writing. Formatting and data-source differences can still cause misleading correlations.

The extractor is **stateless**: it learns no vocabulary or corpus-wide statistics from the training or validation set. Every passage is processed independently with the same rules. The implementation is reused unchanged from [`StyleFeatures`](../text_generalization/style_search.py), and each completed run exports the ordered feature names.

## 5. What is ExtraTrees, and how is ours configured?

**ExtraTrees** means **Extremely Randomized Trees**. It combines many decision trees. At a split, it considers randomly selected candidate features and randomly generated thresholds, then uses the training labels to choose a useful split. Different trees learn different rules. For a new passage, the classifier averages their predicted AI probabilities.

The retained settings are fixed in [`recipe.json`](recipe.json):

| Parameter | Value | Meaning |
|---|---|---|
| `n_estimators` | `800` | Train 800 trees. |
| `max_features` | `0.8` | Consider a random subset of about 80% of features at a split. |
| `min_samples_leaf` | `2` | Require at least two samples in a leaf. |
| `max_depth` | `None` | No explicit depth limit. |
| `criterion` | `gini` | Use Gini impurity to evaluate candidate splits. |
| `bootstrap` | `True` | Draw each tree's training sample with replacement. |
| `max_samples` | `0.8` | Make 80% as many bootstrap draws as there are fitting rows. |
| `class_weight` | `balanced_subsample` | Calculate class weights from each bootstrap sample's class frequencies. |
| `random_state` | `12` | Fix model randomness for reproduction. |
| `n_jobs` | `2` | Use two workers. |

**Bootstrap sampling is separate from the 8,000/2,000 validation split.** An 8,000-row fit makes 6,400 draws per tree; the final 10,000-row fit makes 8,000 draws per tree. Draws can repeat passages, so these are not counts of distinct articles. ExtraTrees supports training without bootstrap; this particular recipe explicitly enables it.

The wrapper clips final probabilities to `[1e-7, 1 - 1e-7]`, matching the retained submission implementation. See the official [ExtraTrees explanation](https://scikit-learn.org/1.7/modules/ensemble.html#extremely-randomized-trees) and [classifier API](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html).

## 6. Validation, selection, and the reported improvement

The original split uses `test_size=0.2`, `random_state=12`, and `stratify=label`. It contains 4,000 human and 4,000 AI training examples, plus 1,000 of each class in validation. The runner verifies both the source data and the exact split against the saved recipe.

Before selecting this model, development experiments compared 26 fitted configurations and 50 recipes, including fixed probability blends. Selection used diagnostic partitions **inside the original 8,000 examples**: random validation, topic holdout, layout holdout, and a whitespace-collapse stress check. The final selected recipe is one ET800 classifier, not a blend of those candidate models.

After selection, the fixed recipe was trained on the 8,000 examples and evaluated on the original 2,000-row validation partition:

| Measurement | Previous ET400 | Retained ET800 |
|---|---:|---:|
| Historical 2,000-row validation AUROC | 0.9958890 | 0.9956835 |
| Kaggle public AUROC change, reported by Dave | Reference submission | +0.004 |

The local score decreased slightly even though the reported public score increased. The original validation partition was used in earlier project work, and the development diagnostics were reused; neither should be described as a pristine independent test. The public score does not establish a private-leaderboard gain. Tree count, bootstrap sampling, and class weighting changed together, so this comparison does not isolate which change caused the public improvement.

## 7. Run the retained model

From the repository root, use Python 3.10; the original run used Python 3.10.8. Create an environment and install the pinned dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r models/dave_extratrees/requirements.txt
```

Place the original competition files at `data/train.csv`, `data/test.csv`, and `data/sample_submission.csv`. Then run:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
python -m models.dave_extratrees.train --data-dir data
```

The runner evaluates the frozen recipe on the historical 8,000/2,000 split, then refits the same recipe on all 10,000 labeled examples and predicts the 3,000 test passages. It checks data hashes, IDs, label balance, feature count, finite probabilities, serialized-model reload, and prediction consistency across batches. The validation result does not trigger parameter changes.

Use the original CSV files without rewriting or reordering them: the data checksums and ordered split digests in `recipe.json` deliberately enforce the same benchmark. The full ordered list of 145 columns is also available in [`feature_names.json`](feature_names.json).

Each run creates a new directory under `outputs/dave_extratrees/<UTC timestamp>/`:

| Output | Purpose |
|---|---|
| `submission.csv` | Kaggle predictions: exactly `id,label`, 3,000 rows, in sample-submission ID order. Here `label` is a probability. |
| `model.joblib` | The final model fitted on all 10,000 labeled passages. |
| `validation_predictions.csv` | Predictions from the separate 8,000-row fit on the 2,000 validation rows. |
| `metrics.json` | Measured results, configuration, versions, hashes, and verification details. |
| `feature_names.json` | The 145 feature names in matrix-column order. |
| `recipe.json` | A copy of the exact recipe used for this run. |

Upload `submission.csv` manually through Kaggle's submission interface. These commands do not upload files or contact an LLM API. Raw competition data, generated prediction files, model binaries, and rejected searches are not needed in this code package; supply the original data locally.

Run the focused package tests with:

```bash
python -m unittest models.dave_extratrees.test_model -v
```

The package's historical training entry point and model implementation are [`train.py`](train.py) and [`model.py`](model.py). They reuse the existing feature extractor without changing the team's shared preprocessing or other model implementations. The method was originally developed on `Dave-new-method-1` and was later imported into the team's course-submission branch.

## 8. Reproduce the generalization and error analysis (Q4.1–Q4.4)

First run the training command in Section 7. Then replace `RUN_DIRECTORY` below
with the timestamped output directory printed by that command:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
python -m models.dave_extratrees.analyze_errors \
  --run-dir RUN_DIRECTORY --data-dir data
```

The analysis reads the run's `metrics.json`, `recipe.json`, and
`validation_predictions.csv`, plus the original `data/train.csv`. It verifies
the retained configuration, source/data hashes, and exact validation IDs and
labels. These predictions come from the **8,000-row benchmark fit**. The
separate model fitted on all 10,000 rows is not used for error analysis.

The analysis keeps the original model, features, and split fixed:

- **Q4.1:** report training AUROC from the benchmark's saved metrics and
  independently recompute validation AUROC from its saved predictions.
- **Q4.2:** count tokens with the feature extractor's regular expression and
  evaluate fixed bins of fewer than 150, 150–299, and at least 300 tokens.
- **Q4.3:** predict AI at probability **>= 0.5**; identify false positives
  (human predicted AI) and false negatives (AI predicted human). Select the
  five highest-confidence mistakes of each type, breaking ties by ID.
- **Q4.4:** summarize observable paragraph, line, and formatting patterns.
  Explanations of why individual passages failed still require text inspection;
  these are hypotheses, not established causal attributions.

Each run writes a new `analysis.json` under
`outputs/dave_extratrees/analysis/<UTC timestamp>/`. It contains aggregate
statistics and the ten selected cases' IDs, probabilities, and numeric
features, without original passages. No model fitting, tuning, or Kaggle upload
occurs. `--output-dir` can select a different output parent directory.

The expected results are recorded in
[`generalization_reference.json`](generalization_reference.json): validation
AUROC **0.9956835**, **19 false positives**, and **35 false negatives**. The
length-bin counts are **395 / 1,404 / 201**, with AUROCs approximately
**0.989915 / 0.995971 / 0.998298**, respectively. All 35 false negatives have
one paragraph under the extractor's blank-line definition. All 640 validation
passages with multiple paragraphs are AI-labeled and correctly classified;
that subgroup therefore has no human examples and no defined within-group
AUROC. The observed formatting correlation is not proof of causation. The
historical validation set was reused, and the ten cases are selected examples,
so neither should be presented as an unbiased independent test.

## Publication verification

The package was run in a fresh local clone containing the committed repository, this selected package, and the three original input CSVs. No untracked experiment code, cached feature matrices, diagnostic prediction arrays, or previous model binaries were required. It reproduced the historical validation AUROC **0.9956835**; all 2,000 validation predictions and 3,000 submission predictions matched the retained version within an absolute tolerance of **1e-12**. The 5 package tests, 5 shared-split tests, and 58 existing NB tests passed. Exact comparison errors, hashes, versions, and scope are recorded in [`reproduction.json`](reproduction.json).

## AI assistance

Codex assisted with implementation, experiment execution, verification, and documentation. The team is responsible for reviewing the code, understanding the method and its limitations, and accurately disclosing this assistance in the course submission.
