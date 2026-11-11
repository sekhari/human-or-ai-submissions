# Group Number 28 — Human or AI?

Cornell Tech · Applied Machine Learning · Kaggle competition code submission.

## Team members

| Team member | NetID | Main work |
| --- | --- | --- |
| Yousen Lin | yl4358 | Q1 supervised comparisons; Q5 improvements; final RoBERTa pipeline |
| Alexander Yen | ay476 | Q2 learning curves; Q4 generalization and error analysis |
| Gabriel Draghici | gpd42 | Q3 semi-supervised learning; Q6 integration |

## Kaggle team name

**Group Number 28**. Course submission directory:
`submissions/group-number-28/`.

The final prediction pipeline fine-tunes RoBERTa-base on the original text,
preserving whitespace. The source run used two epochs and seed 42. The reported
Kaggle public AUROC was **0.96970**; this is a historical leaderboard result,
not a guarantee for a new fit. Classical supervised comparisons, ablations,
semi-supervised experiments, and the Q2/Q4 notebook are retained alongside the
final entrypoints. See [development history](DEVELOPMENT_HISTORY.md) and
[AI disclosure](AI_DISCLOSURE.md).

## Setup

All commands below start in `submissions/group-number-28/`, the directory
containing this README. Python **3.11** is the neural environment; the source
experiments used Python 3.11.15. Create an isolated environment:

```sh
python3.11 -m venv .venv-final
source .venv-final/bin/activate
python -m pip install -r requirements-final.txt
```

Download the original competition files from Kaggle and place them in
`data/raw/`. The final pipeline needs `train.csv` and `test.csv`; Q3 additionally
needs `unlabeled.csv`. Data and trained model weights are not included in Git.
Use the original CSVs without rewriting or normalizing them: the canonical
split loader checks the original training-file hash.

```sh
python -m human_ai.prepare_data
```

This verifies the committed split IDs in `data/splits/` and materializes local
`data/processed/train.csv` and `data/processed/validation.csv`. The split contains
8,000 training rows and 2,000 validation rows, balanced by class, using seed 42.
The committed IDs and manifest are authoritative; preparation does not replace
an existing split.

The first model run downloads the public
[`FacebookAI/roberta-base`](https://huggingface.co/FacebookAI/roberta-base)
checkpoint at revision `e2da8e2f811d1448a5b465c236feacd80ffbac7b`. Internet access
is needed for that download unless the revision is already cached. Subsequent
offline runs can pass `--local-files-only`. Fine-tuning requires enough memory
for RoBERTa-base and its optimizer. `--device auto` selects CUDA, then Apple MPS,
then CPU; `--device cpu`, `--device mps`, and `--device cuda` select explicitly.
CPU execution is supported but slower. The original final model used Apple MPS
in FP32. Device and library differences can change fitted weights and scores;
cross-device bitwise agreement is not promised.

## Train / validate

```sh
python train.py --output-dir artifacts/roberta-validation
```

This fits only the canonical 8,000 training rows and evaluates on the 2,000
held-out validation rows after the final epoch. It writes
`artifacts/roberta-validation/validation_predictions.csv` with `id,label,p_ai`,
and `artifacts/roberta-validation/metrics.json` with the validation results.
The fitted model is saved below `artifacts/roberta-validation/model/`; the
completed run has a `completion.json` marker. Validate the prediction contract:

```sh
python -m human_ai.check_predictions artifacts/roberta-validation/validation_predictions.csv --kind validation
```

## Predict

After selecting the recipe, fit a separate model on all 10,000 supplied labeled
rows and predict the 3,000 test rows:

```sh
python train.py --full-data --output-dir artifacts/roberta-full
python predict.py --model-dir artifacts/roberta-full --output submission.csv
python -m human_ai.check_predictions submission.csv --kind submission
```

Pass the run directory to `--model-dir`, not its nested `model/` directory.
`submission.csv` has columns `id,label`; `label` is the probability of AI
authorship, class 1, rather than a thresholded class. The full-data fit has no
independent validation score. These commands generate the Kaggle file locally;
upload it through Kaggle separately.

## Seeds and notes

The fixed default recipe is:

| Setting | Value |
| --- | --- |
| Input | Original text, including whitespace; right truncation at 512 tokens |
| Seed | 42 |
| Epochs | 2 |
| Microbatch / gradient accumulation | 4 / 4; effective batch size 16 |
| Optimizer | AdamW; learning rate 0.00002; weight decay 0.01 |
| Learning-rate schedule | Constant; no warmup |
| Gradient clipping | Maximum norm 1.0 |
| Precision | FP32 |

Use new output directories for separate runs so their artifacts remain distinct.
The final entrypoints are independent of historical study receipts and local
research checkpoints.

### Historical classical experiments: Q1 and Q5

The original classical models used a separate Python **3.12** environment.
Keep its pinned NumPy and scikit-learn versions separate from the neural
environment, especially when reading saved models.

```sh
python3.12 -m venv .venv-classical
.venv-classical/bin/python -m pip install -r experiments/supervised/requirements.txt
.venv-classical/bin/python -m human_ai.prepare_data
.venv-classical/bin/python -m experiments.supervised.expanded_q1 --prefix a_q1_course_reproduction_001
.venv-classical/bin/python -m experiments.supervised.q5_job --prefix a_q5_course_reproduction_001
```

Detailed grids, study-specific seeds/configurations, and other commands are in
[`experiments/supervised/README.md`](experiments/supervised/README.md) and the
adjacent JSON plans. Some frozen crop, stress, and RoBERTa research scripts
deliberately depend on ignored local reference models, cached snapshots, source
hashes, or old run receipts. Their archived evidence documents the original
experiments; those scripts are not the independent final training/prediction
entrypoints above. A fresh clone does not recreate those historical prerequisites
automatically.

### Semi-supervised experiments: Q3

Q3 uses its recorded Python **3.13** environment, seed **20261109**, the canonical
8,000/2,000 split, and `data/raw/unlabeled.csv`. From this team directory:

```sh
python3.13 -m venv .venv-q3
.venv-q3/bin/python -m pip install -r experiments/semisupervised/requirements.txt
.venv-q3/bin/python -m human_ai.prepare_data
.venv-q3/bin/python -m experiments.semisupervised.run_q3 --prefix c_q3_course_reproduction_001
.venv-q3/bin/python -m experiments.semisupervised.plot_q3 --prefix c_q3_course_reproduction_001
.venv-q3/bin/python -m experiments.semisupervised.robustness_check --prefix c_rob_course_reproduction_001
```

Use unused prefixes: these experiment scripts refuse to overwrite an existing
run. See [`experiments/semisupervised/README.md`](experiments/semisupervised/README.md)
for pseudo-label thresholds, rounds, and the negative-result interpretation.

### Learning curves and error analysis: Q2 and Q4

The existing version of
[`experiments/generalization/AlexYen_Train.ipynb`](experiments/generalization/AlexYen_Train.ipynb)
is retained as provided. This packaging pass has not changed its experiment
code or rerun the notebook. Saved outputs and execution counts were cleared
for public distribution because error-analysis output contained competition
passage excerpts. The original notebook and results remain in the team
working repository; the archived learning-curve figure is retained here.

From the team directory, open it in VS Code:

```sh
code experiments/generalization/AlexYen_Train.ipynb
```

Select a separate Python 3 notebook kernel, choose **Restart**, then **Run All**.
The first executable cell installs the notebook's analysis dependencies.
Keep the input files in `data/raw/`. The notebook uses seed 42 for its split,
models and learning-curve sampling, 123 for shuffled CV, and 17 for its
distribution-shift diagnostic. It constructs its own split; the canonical-split
statement for the final pipeline above does not describe that notebook.

Results are displayed in the notebook and retained when it is saved. Its final
cells write their own candidate CSVs under `data/raw/`; use the root
`train.py` / `predict.py` commands above for the selected final RoBERTa
submission. The original execution instructions are in
[`experiments/generalization/README.md`](experiments/generalization/README.md).

## AI assistance

[AI_DISCLOSURE.md](AI_DISCLOSURE.md) contains the member statements supplied for
report Q6.3. Yousen used Codex for code generation, debugging, experiment execution,
and report editing. Gabriel used Claude for code generation, debugging, and
report editing. Alexander used an AI assistant to generate and debug the Q2
learning-curve and Q4 error-analysis code and improve report clarity. He reviewed
the experimental setup, checked plots and metrics against saved results,
assessed model errors, and revised the explanations to distinguish observations
from possible explanations. Their statements describe the ideas, decisions,
corrections, and checks they personally contributed. Automated checks
recorded in this repository should not be represented as personal review or
independent reproduction by a team member.

[DEVELOPMENT_HISTORY.md](DEVELOPMENT_HISTORY.md) records the development
lineage and the relationship between the team working repository and this
course submission. Preserve the substantive commits when reviewing or merging
the course PR.

## Submission checklist

- [x] Code is only under `submissions/<kaggle-team-name>/`
- [x] Team `README.md` documents environment, commands, seeds, and AI use
- [x] No competition CSVs, secrets, or huge weight dumps committed
- [x] Meaningful commit history retained (not squashed away)
- [x] We did not retrieve hidden labels from RAID or elsewhere
