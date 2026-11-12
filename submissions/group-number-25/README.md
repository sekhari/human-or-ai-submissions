# Group Number 25 — Human or AI?

**Kaggle team name:** Group Number 25 (folder / branch: `group-number-25`)

| Member | NetID | Main contribution |
|---|---|---|
| Jenson Shao | zs496 | Model B: word + character TF-IDF text model, style-feature model, blend (final submission) |
| Sanskriti Malakar | sm3292 | Baseline (TF-IDF + logistic regression), shared validation split, learning-curve study |
| Nihar Degaonkar | nd488 | Neural networks from scratch (TF-IDF MLP, text CNN) and their blend |

**Final Kaggle submission:** Model B, validation AUROC **0.970** on the shared 2,000-passage holdout.

## Environment

- Python **3.13** (developed on 3.13.13, macOS)
- Dependencies are pinned in `requirements.txt`. PyTorch is only needed for `03_nn_from_scratch.ipynb`.

```bash
cd submissions/group-number-25
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the competition files from Kaggle and place them in `data/` (not committed):

```text
submissions/group-number-25/data/train.csv
submissions/group-number-25/data/test.csv
submissions/group-number-25/data/sample_submission.csv
submissions/group-number-25/data/unlabeled.csv   # only for 05_modelB_analysis
```

## Write `submission.csv` (final model: Model B)

Run from `submissions/group-number-25/` with the environment active:

```bash
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
    --output-dir executed_notebooks notebooks/02_modelB.ipynb
cp outputs/submission_modelB.csv submission.csv
```

The notebook tunes and evaluates on the 8,000 / 2,000 split, refits on all 10,000 labeled passages, and writes `outputs/submission_modelB.csv` (columns `id`, `label` = predicted probability of AI authorship). It takes a few minutes on a laptop CPU. All other output files are written to `outputs/`.

## Train / evaluate on the labeled validation protocol

Every model uses the same stratified holdout: 2,000 of the 10,000 labeled passages, `random_state=42`. Run the baseline first, because it writes the split IDs (`outputs/train_ids.csv`, `outputs/validation_ids.csv`) that notebooks 03 and 04 load. Model B re-creates the split itself and asserts it matches those IDs.

```bash
NB="jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir executed_notebooks"
$NB notebooks/01_baseline.ipynb            # baseline; writes the shared split IDs
$NB notebooks/02_modelB.ipynb              # Model B (final submission)
$NB notebooks/03_nn_from_scratch.ipynb     # TF-IDF MLP + text CNN (needs torch)
$NB notebooks/04_training_data_size.ipynb  # learning curve for the baseline
$NB notebooks/05_modelB_analysis.ipynb     # Model B: learning curve, self-training, error analysis
```

| Notebook | Model | Validation AUROC |
|---|---|---|
| `01_baseline` | Word 1–2-gram TF-IDF + logistic regression | 0.936 |
| `03_nn_from_scratch` | TF-IDF MLP (from scratch) | 0.940 |
| `03_nn_from_scratch` | Text CNN with embeddings learned from scratch | 0.949 |
| `03_nn_from_scratch` | MLP + CNN rank-average blend | 0.961 |
| `02_modelB` | Text model: word + character TF-IDF + logistic regression | 0.953 |
| `02_modelB` | Style model: 12 writing-style features + gradient boosting | 0.892 |
| `02_modelB` | **Model B: 0.6 × text + 0.4 × style** | **0.970** |

No hyperparameter or blend weight was chosen with validation or test labels:

- **Model B:** text-model `C ∈ {1, 4, 10}` and style-model learning rate `{0.05, 0.1}` × max depth `{3, None}` are chosen by 3-fold CV on the 8,000 training passages. The blend weight `w ∈ {0, 0.1, …, 1}` is chosen with out-of-fold predictions on the same passages.
- **Neural networks:** hidden size, dropout and weight decay (MLP grid) and the number of epochs (early stopping) are chosen on a 10% inner split of the 8,000 training passages.
- **Baseline:** fixed settings.

## Seeds and important configuration

- Validation split: `train_test_split(..., test_size=2000, stratify=label, random_state=42)`
- Logistic regression and `HistGradientBoostingClassifier`: `random_state=42`. `GridSearchCV` / `cross_val_predict` use unshuffled stratified 3-fold CV, which is deterministic.
- Neural networks: inner early-stopping split `random_state=42`; PyTorch seeds `0, 1, 2` (predictions averaged). GPU (MPS/CUDA) runs can differ very slightly between runs.
- Preprocessing (Model B): all whitespace is collapsed to single spaces in every passage, because 66% of AI training passages contain blank lines and no human passages do (a data-collection artifact).
- No pretrained models or external data are used. The unlabeled set is used only in the self-training experiment in `05_modelB_analysis.ipynb`; it did not help (0.9685 vs. 0.9700 validation AUROC) and is not part of the final submission. Competition data and generated CSVs are not committed.

## AI assistance

Claude (Anthropic) was used through the Claude app:

- **Code:** it wrote most of `03_nn_from_scratch.ipynb` (architectures, hyperparameter grid, training loop) and `02_modelB.ipynb` / `05_modelB_analysis.ipynb` (Model B and its analysis).
- **Setup:** it guided the Git/GitHub and VS Code setup.
- **Analysis:** it compared the team's models from saved validation predictions, including a bootstrap check of Model B's lead.
- **Submission:** it prepared this submission branch: filtering data files out of the history, merging Model B, and writing this README.

All training was run by team members on their own machines, and the reported numbers were checked against the saved output files. See Q6.3 in the report for the full disclosure.
