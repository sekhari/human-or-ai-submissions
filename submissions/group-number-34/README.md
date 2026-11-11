# Classical Text Classification

Team members: Sydney Patel (sjp294), Teddy Taylor (tft32), Zhenghao Zhou (zz2222).

Kaggle team: `group-number-34`.

## Setup

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

## Models and experiment settings

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

## Reproduce the experiments

Run model comparison, parameter search, and the learning curve:

```sh
python run_experiments.py --train data/train.csv --output-dir runs/reproduction --seed 42
python plot_results.py --results-dir runs/reproduction --output-dir figures
```

The runner performs 24 distinct fits, reusing the 8,000-example combined C=16 fit for the last learning-curve point.
It writes `comparison_selected.csv`, `tuning.csv`, `learning_curve.csv`, full-precision predictions, and the experiment settings under `runs/reproduction/`.
The plot command writes `figures/learning_curve.pdf`.

## Recorded results

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

## Generate probabilities

Train the selected classical model on all 10,000 labeled examples and generate its Kaggle prediction file:

```sh
python train.py --train data/train.csv --model word_char_lr --C 4 --seed 42 --full --model-out artifacts/model.joblib
python predict.py --test data/test.csv --model artifacts/model.joblib --out artifacts/submission.csv
```

`artifacts/submission.csv` contains one row per test example, in the original test-file order.
Its columns are `id` and `label`; `label` is the predicted probability of AI authorship.

## Development history

- The team's original commits introduce word features, character features, the calibrated SVM, the combined model, and the learning curve.
- `classical_fixed2000_v001` records the experiments on the fixed validation IDs with seed 42.
- `classical_fixed2000_v002` adds the combined model's C search and selects C=4. Its validation AUROC is 0.950644, compared with 0.950553 at C=16.
- The submission package uses one experiment command and one dependency file, and includes the required learning-curve plotting command.

## AI Disclosure

I used ChatGPT to help modify the code, plot the experimental results, and organize the documentation.
