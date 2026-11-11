# Small transformer and 75/25 blend

The selected entry combines 75% of v2's standardized score with 25% of the fine-tuned DistilRoBERTa margin. V2 itself averages standardized scores from a boosted style/language-model classifier and word TF-IDF logistic regression. `predict.py` writes the selected predictions to `outputs/submission.csv`.

## Reproduce

From the parent team directory (`submissions/group-21/`) on a CUDA machine, use a clean Python 3.11 or 3.12 environment. Put the original competition files `train.csv`, `test.csv`, `unlabeled.csv`, and `sample_submission.csv` in a private directory. Model downloads require internet access on the first run.

```sh
python -m pip install -r requirements.txt
python train.py --data-dir /path/to/data --out-dir outputs
python predict.py --data-dir /path/to/data --out-dir outputs
```


Training saves the v2 estimators and vocabulary in `outputs/distilroberta_v1/v2.joblib`. The fitted transformer and tokenizer are in the checkpoint directory named in `final_checkpoint.json`. Prediction uses those artifacts, `test.csv`, and `sample_submission.csv`. It can run on CPU or CUDA.

Use an empty output folder for a new experiment. For an exact replay of the historical split and checkpoint selection, copy the original `val_ids.csv` to the output folder and `pretrained_revision.txt` to its `distilroberta_v1/` subdirectory. Otherwise the runner generates the original experiment's stratified split with seed 42 and records the currently resolved DistilRoBERTa model revision. Reuse cached arrays only with the same inputs, ordering, model versions, and settings. GPU arithmetic and package versions can affect the scores.

## Experiment order

1. `01_baselines.py` builds word and character TF-IDF, style and distilgpt2 features, and all-MiniLM-L6-v2 embeddings. It runs grouped cross-validation, the size and pseudo-labeling experiments, and the original stack. Following the experiment, both TF-IDF vectorizers fit vocabulary and IDF on all labeled and unlabeled passages, including validation text without labels.
2. `02_robustness.py` ablates style features, trains on edited copies, scores held-out edits and exports v2 scores. V2 uses 25 style/LM features and a separate word model. The diagnostic export contains private passage text and belongs in the output directory.
3. `03_distilroberta.py` fine-tunes `distilbert/distilroberta-base` for two epochs on 8,000 rows, tests four variants on the fixed 2,000-row validation set, then fits on all 10,000 labels. It exports `submission_distilroberta.csv`. It uses neither synthetic edits nor unlabeled passages for training.
4. `04_paired_blend.py` compares both components on the exact same v2 probe strings, evaluates an exploratory character model, measures the selected 75/25 validation blend. The standalone transformer probe implementation uses different edits, so its table is separate from the paired comparison. The blend aligns each model to the sample submission by ID, transforms probabilities to logits, and standardizes over test rows before applying the fixed weights and sigmoid.

The training entrypoint runs the Python sources in `distilroberta/experiments/` in order. Use `--stage v2` for the baseline and robustness study, `--stage transformer` for the standalone transformer, or `--stage blend` to reuse an existing v2 run. Transformer stages reload completed checkpoints if present. The code reads competition data and writes predictions locally without submitting to Kaggle.

AI assistants helped develop and package the experiments and draft the report. All passage scores come from the models described above.
Set `RUN_TRAIN_DIAGNOSTIC=1` when running the full study to also save `selected_train_validation_auc.csv`, containing the train/validation AUROC gap of the selected blend.

## Evaluate saved validation outputs

From the team directory, run the following on the exported scores. This needs only NumPy, SciPy and scikit-learn. Both plain CSV and gzip-compressed CSV are supported.

```sh
python distilroberta/diagnostics.py --train /path/to/data/train.csv --v2-probes /path/to/v2_validation_probes.csv.gz --transformer-probes /path/to/validation_probes.csv.gz
```
