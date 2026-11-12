# Reference baseline experiments

Commands run from the repository root. See the [main README](../README.md) for the final 0.98921 pipeline.

## Reference baseline and artifacts

The reference baseline achieves **0.997975 validation AUROC** on the fixed, stratified 2,000-row holdout. The team supplied a Kaggle screenshot reporting **0.96962 public AUROC** for the new submission on November 11, 2026, up from the previous user-reported **0.92433**. The validation-to-public gap is 0.028355; its cause is not established, and private AUROC remains unknown. The user uploaded the CSV; this code did not make an automated Kaggle submission.

- `output/submission_baseline.csv`: reference baseline probabilities for all 3,000 test IDs, in sample order (local; excluded from Git).
- `artifacts/final.joblib`: locally trained model, refit on all 10,000 labeled examples (excluded from Git).
- `report/main.tex`, `report/main.pdf`: editable Overleaf source and compiled report.
- `output/group11_overleaf.zip`: upload-ready LaTeX project (local).
- `results/`: actual experiments, fixed split, validation predictions, robustness diagnostics and data checksums. CSV evaluation logs contain IDs/results, not passages, and are generated locally (ignored by Git).

## Baseline setup

Run commands from the repository root. After setup, `bash submissions/group-number-11/scripts/reproduce.sh DATA_DIR` runs the baseline pipeline. Tested on macOS arm64, Python 3.14.2, 24 GiB RAM; CPU training for the reference baseline, without pretrained weights. The final pipeline in the main README adds local frozen language weights. Python 3.14.2 reproduces the tested environment; other Python versions require compatible package wheels. `environment.txt` records the exact baseline versions used. On macOS, install the LightGBM OpenMP dependency with `brew install libomp`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r submissions/group-number-11/requirements.txt
```

Keep the original Kaggle CSVs in `cornell-tech-cs-5785-combined-2026-ai-vs-human-text/` or substitute your own data-directory path. Do not commit the data, checkpoints or credentials.

## Baseline experiments

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python submissions/group-number-11/src/train.py \
  --data cornell-tech-cs-5785-combined-2026-ai-vs-human-text --final
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python submissions/group-number-11/src/robustness.py \
  --data cornell-tech-cs-5785-combined-2026-ai-vs-human-text
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python submissions/group-number-11/src/improve.py \
  --data cornell-tech-cs-5785-combined-2026-ai-vs-human-text --final
python submissions/group-number-11/src/analyze.py \
  --data cornell-tech-cs-5785-combined-2026-ai-vs-human-text
```

`train.py` reserves a fixed stratified 2,000-row validation set using seed **20261023** and fits all supervised representations and classifiers only on the remaining 8,000 rows. The starter file is not added as extra data. Models use text only; IDs are solely identifiers. We compare word/character logistic regression, linear SVM, NB-weighted logistic regression, and gradient boosting on writing-style features. The fixed grid searches LR/SVM C in {0.1, 1, 4, 16}, NB-LR C in {0.1, 1, 4}, and style boosting leaves in {7, 15, 31} with {300, 700} trees. For each representation, validation AUC chooses a component; a 15-entry blend grid (including repeated pure-style endpoints) chooses the final weights.

The reference blend is 0.125 word SVM (C=1), 0.125 character LR (C=16), and 0.75 style LightGBM (31 leaves, 700 trees). The SVM margin is mapped through a sigmoid; this is **not** calibrated probability estimation. The ensemble probabilities are optimized for ranking/AUROC, not individual authorship certainty.

The learning curve uses word LR C=4 with a freshly fitted vocabulary at each of 500/1,000/2,000/5,000/8,000 labeled examples. The unlabeled experiment learns word vocabulary and IDF on 8,000 training + 3,000 unlabeled texts and trains LR C=4 only on true training labels. It excludes validation text/labels and test text from fitting and does not improve validation AUC, so it is not used in the final model.

`improve.py` tests label-preserving whitespace augmentation. Its acceptance rule was fixed before execution: native AUC within 0.0005 of the selected blend and whitespace-collapsed AUC >= 0.98. No augmented candidate passes, so the original blend is retained. The strongest native-text augmented candidate scores 0.997503 native / 0.978602 collapsed; the original blend scores 0.997975 / 0.877512. Formatting dependence is a material limitation despite high random-split AUC.

`robustness.py` freezes settings before three secondary split checks (seeds 20261024/25/26), an ablation, and validation-text transformations. These secondary splits overlap model-selection data and are **stability checks**, not independent holdouts. Bootstrap intervals are conditional on the selected fitted model and exclude selection uncertainty. Nearest-neighbor filtering does not establish source/domain independence.

## Regenerate the baseline CSV

```bash
python submissions/group-number-11/src/predict.py \
  --test cornell-tech-cs-5785-combined-2026-ai-vs-human-text/test.csv \
  --sample cornell-tech-cs-5785-combined-2026-ai-vs-human-text/sample_submission.csv \
  --out submissions/group-number-11/output/submission_baseline.csv
python submissions/group-number-11/src/verify.py \
  --data cornell-tech-cs-5785-combined-2026-ai-vs-human-text
```

The final submission is documented in the main README; these commands regenerate the historical baseline. The columns are exactly `id,label`; label is P(AI). Do not threshold predictions. No external hidden labels were used or retrieved.
