# Group Number 8

**Team members:** Kexin (Cosine) Feng (kf442), Ethan Nguyen (eln39), Ketan Verma (kv293)  
**Kaggle team name:** `group-number-8`

The submitted scores are `ENS_top2`: an equal-weight rank average of `sgd_bpe` (TF-IDF on a from-scratch byte-level BPE) and `gbm_format` (21 layout features, histogram gradient boosting). `ENS_top3` adds `lr_bpe` and is written for comparison. It is not the file we upload.

## Setup

Tested with `/opt/miniconda3/bin/python3.12` and scikit-learn 1.5.2. From this directory:

```bash
/opt/miniconda3/bin/python3.12 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`--system-site-packages` keeps the conda scikit-learn 1.5.2. `tokenizers==0.20.3` supplies the BPE tokenizer. No pretrained language model is used.

Place the Kaggle files where you like and pass that directory as `--data-dir`. Do not commit them. The commands below use the course CSVs that live next to this repository.

## Train / validate

This is the command behind the report. It holds out the stratified 2,000-row validation set (`random_state=20261023`, 1,000/1,000), searches hyperparameters with 3-fold CV on the other 8,000 rows, and writes `results/metrics.json` plus `figures/learning_curve.pdf`. It does not read `test.csv`.

```bash
python train.py \
  --data-dir ../../cornell-tech-cs-5785-combined-2026-ai-vs-human-text \
  --mode report \
  --seed 20261023
```

## Predict (writes the Kaggle file)

Refit `ENS_top2` on all 10,000 labels. BPE and TF-IDF are fit on train, unlabeled, and test text. Test labels are not used. `submission.csv` is the two-model rank average. `submission_ens_top3.csv` is the three-model average.

```bash
python train.py \
  --data-dir ../../cornell-tech-cs-5785-combined-2026-ai-vs-human-text \
  --mode submit \
  --out submission.csv
```

`predict.py` reloads `model.joblib` and writes the same two-model file:

```bash
python predict.py \
  --test ../../cornell-tech-cs-5785-combined-2026-ai-vs-human-text/test.csv \
  --model model.joblib \
  --out submission.csv
```

`model.joblib` and the CSV files are local artifacts. Do not commit them.

## Seeds and locked settings

- Validation split seed: `20261023`
- Classifier `random_state`: `42` (`SGDClassifier`, `HistGradientBoostingClassifier`)
- `sgd_bpe`: modified Huber, `alpha=5e-5`
- `lr_bpe`: liblinear, `C=4`
- `gbm_format`: 300 iterations, learning rate 0.05, 15 leaves, minimum 40 samples per leaf, L2 penalty 1
- Decision threshold for the error tables only: 0.5 on the validation rank average

The inner-CV grid is in `results/metrics.json`. `alpha=1e-4` and `C=16` were a few thousandths higher than the locked values. The booster grid was flat (0.9886 to 0.9894). The holdout was not used to pick these values.

## AI assistance

An AI coding assistant helped turn the model definitions into `train.py` / `predict.py` and drafted report sentences from `results/metrics.json`. The five models and the choice to submit `ENS_top2` were ours. Every AUROC in the report comes from the report command above. No tool labeled the test passages.
