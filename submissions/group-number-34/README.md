# Human or AI? - Group 34

Team members: Sydney Patel (sjp294), Teddy Taylor (tft32), Zhenghao Zhou (zz2222).

Kaggle team: `group-number-34`.

## Classical models

### Setup

Use Python 3.12. Install the package versions listed in `requirements.txt`:

```sh
cd submissions/group-number-34
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place the competition files `train.csv` and `test.csv` in `data/`.
The input columns are `id`, `text`, and, for training, `label`.
Label 0 denotes human-written text; label 1 denotes AI-generated text.

Set the numerical-library thread counts used for the recorded experiments:

```sh
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
```

### Models and experiment settings

All models use TF-IDF with lowercasing and sublinear term frequency.
Word features use 1-2 grams, a minimum document frequency of 2, and at most 200,000 features.
Character features use `char_wb` 3-5 grams, a minimum document frequency of 3, and at most 300,000 features.
The combined model concatenates word and character features.
The linear SVM uses three-fold probability calibration within the training partition.

The saved IDs in `splits/val_ids.json` define a fixed stratified validation set of 2,000 examples.
The other 8,000 examples form the training partition. The random seed is 42.
TF-IDF is fitted on the training examples for each experiment.

Each of the four models searches `C = 0.25, 1, 4, 16, 64`.
Select the smallest C whose validation AUROC is less than 0.001 below the best candidate for that model, using full-precision scores.
The learning curve keeps the combined model at C=16 and uses stratified training subsets of 500, 1,000, 2,000, 5,000, and 8,000 examples.
The validation set and model settings stay fixed across these five training sizes.

### Reproduce the experiments

Run model comparison, parameter search, and the learning curve:

```sh
python run_experiments.py --train data/train.csv --output-dir runs/reproduction --seed 42
python plot_results.py --results-dir runs/reproduction --output-dir figures
```

The runner performs 24 distinct fits, reusing the 8,000-example combined C=16 fit for the last learning-curve point.
It writes `comparison_selected.csv`, `tuning.csv`, `learning_curve.csv`, full-precision predictions, and the experiment settings under `runs/reproduction/`.
The plot command writes `figures/learning_curve.pdf`.

### Recorded results

The following results were measured on an Apple M1 with 16 GB RAM using Python 3.12.14 and two numerical-library threads.
Fit time measures the model fitting step.

| Model | Selected C | Training AUROC | Validation AUROC | Fit time (s) |
| --- | ---: | ---: | ---: | ---: |
| Word TF-IDF + logistic regression | 16 | 1.000000 | 0.942561 | 2.80 |
| Character TF-IDF + logistic regression | 4 | 0.998259 | 0.938619 | 8.42 |
| Word TF-IDF + calibrated linear SVM | 1 | 1.000000 | 0.941849 | 2.77 |
| Word + character TF-IDF + logistic regression | 4 | 1.000000 | 0.950644 | 10.94 |

The combined model has the highest validation AUROC among these four approaches.
It uses C=4 for the probability-prediction workflow below; the learning-curve experiment keeps C=16.

| Training examples | Training AUROC | Validation AUROC |
| ---: | ---: | ---: |
| 500 | 1.000000 | 0.867260 |
| 1,000 | 1.000000 | 0.904932 |
| 2,000 | 1.000000 | 0.925271 |
| 5,000 | 1.000000 | 0.943838 |
| 8,000 | 1.000000 | 0.950553 |

### Generate probabilities

Train the selected classical model on all 10,000 labeled examples and generate its Kaggle prediction file:

```sh
python train.py --train data/train.csv --model word_char_lr --C 4 --seed 42 --full --model-out artifacts/model.joblib
python predict.py --test data/test.csv --model artifacts/model.joblib --out artifacts/submission.csv
```

`artifacts/submission.csv` contains one row per test example, in the original test-file order.
Its columns are `id` and `label`; `label` is the predicted probability of AI authorship.

### Development history

- The team's original commits introduce word features, character features, the calibrated SVM, the combined model, and the learning curve.
- `classical_fixed2000_v001` records the experiments on the fixed validation IDs with seed 42.
- `classical_fixed2000_v002` adds the combined model's C search and selects C=4. Its validation AUROC is 0.950644, compared with 0.950553 at C=16.
- The submission package uses one experiment command and one dependency file, and includes the required learning-curve plotting command.

## GPT-2 models

The GPT-2 implementation is in `gpt2/`. It includes model selection, final-model training, learning curves, the unlabeled-data comparison, generalization diagnostics, and the consistency experiment.

### Setup

Use Python 3.12 or 3.13 on Linux or Google Colab with a CUDA GPU. The recorded small-model runs used an A100 40GB; the large-model runs used an A100 80GB.

From the repository root:

```sh
cd submissions/group-number-34/gpt2
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place `train.csv`, `unlabeled.csv`, `test.csv`, and `sample_submission.csv` in the team's `data/` directory, one level above `gpt2/`. The text column is `text`, the identifier is `id`, and the supervised target is `label`. Label 0 denotes human-written text and label 1 denotes AI-generated text.

### Experiment versions

Version numbers identify the original experiments. Names describe the model and purpose.

| Version | Name | Model and purpose | Entry |
| --- | --- | --- | --- |
| v006 | `large_adaptation_selection` | GPT-2 large with continued pretraining; five-fold schedule selection | `experiments/v006_large_adaptation_selection/` |
| v007 | `large_format_selection` | Original-text, format-augmentation, and consistency comparisons; five-fold schedule selection for the two new arms | `experiments/v007_large_format_selection/` |
| v008 | `large_format_final` | Final GPT-2 large model with continued pretraining and format augmentation, fitted on all 10,000 labeled passages | `configs/v008_large_format_final.json` |
| v009 | `small_learning_curve` | Supervised GPT-2 small at five training sizes | `configs/v009_small_learning_curve.json` |
| v010 | `small_unlabeled_comparison` | GPT-2 small with and without continued pretraining | `configs/v010_small_unlabeled_comparison.json` |
| v011 | `large_diagnostics` | The v008 method fitted on 8,000 passages for fixed-validation diagnostics | `configs/v011_large_diagnostics.json` |
| v012 | `large_consistency` | Matched nine-epoch comparison of format augmentation with and without consistency training | `configs/v012_large_consistency.json` |

### Models and settings

Pretrained weights come from Hugging Face: `openai-community/gpt2` and `openai-community/gpt2-large`. The final model uses GPT-2 large, continued causal language-model pretraining, and training-time format augmentation.

All controlled comparisons use the same 2,000 validation IDs in `configs/fixed_validation_ids.csv`, leaving 8,000 labeled training passages. The seed is 42. The learning curve uses nested class-balanced subsets of 500, 1,000, 2,000, 5,000, and 8,000 passages.

For v008-v012, classification runs for nine epochs with a maximum length of 512 tokens, batch size 8, gradient accumulation 2, weight decay 0.01, BF16 precision, and 0.3 epoch of warmup. Learning rates are `1e-5` for epochs 1-4, `5e-6` for epochs 5-6, and `2.5e-6` for epochs 7-9. The configuration files record each experiment's settings. This nine-epoch schedule comes from the format-augmentation arm's five-fold selection in v007.

Continued pretraining uses 2,700 unlabeled passages for training and 300 for validation, runs for three epochs, and selects the checkpoint with the lowest validation next-token loss. The small-model and large-model experiments preserve their recorded unlabeled splits. Format augmentation selects each training passage with probability 0.25 and replaces sampled newline clusters with spaces. The consistency experiment adds a Jensen-Shannon loss with weight 0.5 between original and augmented views of changed labeled training passages. Prediction uses the original text.

### Reproduce model selection

The v006 and v007 commands use the recorded GPU configurations. Each experiment partitions the 8,000 development passages into five inner folds while keeping the fixed 2,000-passage validation set separate. A fold trains for at most ten epochs, with learning-rate reductions and early stopping. The refit schedule uses the median of the five selected epochs and the median learning rate at each retained epoch.

Run from `gpt2/`:

```sh
python experiments/v006_large_adaptation_selection/prepare_data.py --data_dir ../data --validation_ids configs/fixed_validation_ids.csv --output_dir runs/v006/prepared --seed 42
python experiments/v006_large_adaptation_selection/run_experiment.py --prepared_dir runs/v006/prepared --output_dir runs/v006/results --export_dir runs/v006/model --runtime_config experiments/v006_large_adaptation_selection/runtime_config.json --device cuda --seed 42 --max_length 512 --lm_max_epochs 3 --max_epochs 10 --min_epochs 3

python experiments/v007_large_format_selection/prepare_data.py --data_dir ../data --validation_ids configs/fixed_validation_ids.csv --output_dir runs/v007/prepared --seed 42
python experiments/v007_large_format_selection/run_experiment.py --prepared_dir runs/v007/prepared --output_dir runs/v007/results --export_dir runs/v007/model --runtime_config experiments/v007_large_format_selection/runtime_config.json --device cuda --seed 42 --max_length 512 --lm_max_epochs 3 --max_epochs 10 --min_epochs 3
```

The v006 run selects a schedule, evaluates an 8,000-passage refit, then trains one classifier on all 10,000 labeled passages. The v007 run shares adapted language-model weights across three arms. Its original-text control uses the archived two-epoch v006 schedule. Its format-augmentation and consistency arms each run five-fold schedule selection, followed by an 8,000-passage refit and evaluation in three formats. In the recorded v007 run, the final-candidate rule selected the consistency arm; v008 separately fits the format-augmentation arm for nine epochs on all 10,000 labeled passages.

### Reproduce the fixed-split experiments

Run these commands from `gpt2/`:

```sh
python train.py --experiment learning-curve --data-dir ../data --output-dir runs/v009_small_learning_curve
python plot_results.py --results-dir runs/v009_small_learning_curve --output-dir runs/v009_small_learning_curve/figures

python train.py --experiment unlabeled --data-dir ../data --output-dir runs/v010_small_unlabeled_comparison --supervised-run runs/v009_small_learning_curve
python plot_results.py --results-dir runs/v010_small_unlabeled_comparison --output-dir runs/v010_small_unlabeled_comparison/figures

python train.py --experiment diagnostics --data-dir ../data --output-dir runs/v011_large_diagnostics
python plot_results.py --results-dir runs/v011_large_diagnostics/evaluation --output-dir runs/v011_large_diagnostics/figures

python train.py --experiment consistency --data-dir ../data --output-dir runs/v012_large_consistency --baseline-run runs/v011_large_diagnostics
python plot_results.py --results-dir runs/v012_large_consistency/candidate/evaluation --output-dir runs/v012_large_consistency/figures
```

`--supervised-run` reuses the completed 8,000-passage learning-curve model. `--baseline-run` reuses the completed format-augmentation model. Omitting either option trains that comparison model again. Append `--prepare-only` to a `train.py` command to prepare the data partitions before training.

Each run writes its configuration, data checksums, training history, prediction probabilities, and metrics to its output directory. The diagnostics include AUROC, binary cross-entropy, length groups, and error cases at thresholds 0.5 and 0.2. The consistency comparison evaluates original text, newline replacement, and whitespace normalization with paired bootstrap intervals. Plot commands write PNG and PDF figures.

The consistency run also writes `runtime_comparison.csv`. Its stage timing adds the recorded continued-pretraining and classifier-stage durations, including internal validation and checkpoint writing. `comparison.csv` records process timing, which additionally includes process startup and model initialization. Each arm uses its own recorded adaptation duration.

### Generate the final submission

Train the final configuration on all 10,000 labeled passages and generate the test probabilities:

```sh
python train.py --experiment final --data-dir ../data --output-dir runs/v008_large_format_final
```

The trained classifier is saved in `runs/v008_large_format_final/classifier/best_model/`, and the prediction file is `runs/v008_large_format_final/submission.csv`. To regenerate predictions from the saved classifier:

```sh
python predict.py --checkpoint runs/v008_large_format_final/classifier/best_model --test-csv ../data/test.csv --sample-submission ../data/sample_submission.csv --output runs/v008_large_format_final/submission.csv
```

The output contains `id,label` columns in the sample submission's order. The `label` column contains the probability of AI authorship.

### Development history

The earlier GPT-2 training, formatting-sensitivity, and plotting commits remain in the repository history. The v006 and v007 additions preserve the recorded model-selection implementations. The shared commands cover v008-v012, and the versioned configuration names connect each command to its original experiment. Later commits add the completed consistency experiment's numerical timing comparison and documentation. The original experiment archives remain preserved separately.

## AI Disclosure

We used ChatGPT to help modify the code, plot the experimental results, and organize the documentation.
