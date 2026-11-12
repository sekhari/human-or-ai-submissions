# Linear SVM and MiniLM

From `submissions/group-number-36/jianing_svm/`, use a separate Python 3.13 environment (recorded experiments used 3.13.15 on Apple Silicon):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --prefix "$VIRTUAL_ENV" --name kaggle-jianing --display-name "Python (Kaggle Jianing)"
jupyter lab notebooks/01_Jianing_TF_MiniLM_SVM.ipynb
```

On Windows, activate `.venv\Scripts\Activate.ps1` and use `--prefix "$env:VIRTUAL_ENV"` for kernel installation. Put the supplied competition CSVs in `jianing_svm/data/raw/`. Run all notebook cells once to produce the feature cache and all experiments. First execution downloads the pinned public MiniLM weights; text is processed locally.

Alternatively run without opening the browser:

```bash
# macOS/Linux: permit initial weight download
HF_HUB_OFFLINE=0 python src/run_jianing_notebook.py
# After weights are cached:
python src/run_jianing_notebook.py
```

On Windows, set `$env:HF_HUB_OFFLINE="0"` before the first run. Notebook sections 1–6 cover TF, full-text chunked MiniLM and fusion. Sections 7–8 retain all 30 optimization candidates and the training-only CV diagnostic. Each run saves its configuration, timings, predictions and figures. Large processed matrices remain local.

After the notebook completes:

```bash
# Latest reviewed optimized candidate, full 10,000-row refit:
python src/jianing_optimized_submission.py
# Earlier TF + MiniLM candidate, retained for comparison:
python src/jianing_submission.py
```

The optimized entrypoint reads the recorded selected configuration, reproduces calibrated validation AUROC, and fits on all labels. Sigmoid calibration uses five stratified training-only folds (seed 42), with a fresh vocabulary/IDF within each fold and `ensemble=False`. Output is a new timestamped CSV under `jianing_svm/submissions/`, containing `id,label` where label is P(AI=1). It is not uploaded automatically. The old and optimized prediction scripts share the data/embedding helper in `src/jianing_submission.py`.

### Jianing's recorded progression

| Representation | C | Validation AUROC |
| --- | ---: | ---: |
| Normalized TF | 1 | 0.909056 |
| Full-text chunked MiniLM | 10 | 0.803924 |
| Equal-weight TF + MiniLM | 1 | 0.922313 |
| Character TF-IDF | 1 | 0.972045 |
| Word + character TF-IDF | 1 | 0.972561 |
| Word + character TF-IDF + MiniLM(0.25) | 1 | **0.974103** |

Training-only three-fold means: previous TF+MiniLM 0.916529; optimized candidate 0.967231. These remain in-distribution diagnostics, not a guarantee of Kaggle performance. Complete scores, unsuccessful settings, length-bin comparisons, ROC figures and calibration receipts are under `jianing_svm/results/`.

Section 9 reproduces the [learning curve](results/learning_curve/learning_curve.png) on five nested, balanced training subsets.
