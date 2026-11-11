# Learning curves and generalization — Q2, Q4

Owner: B. Build your own baseline directly from original data and canonical IDs;
do not wait on A's model. Add runnable scripts, configs and tested dependencies.
Write outputs to `results/generalization/<run_id>/`.

Q2: fixed model/settings, five required sizes, same held-out partition.
Q4: create a reusable evaluator taking `id,label,p_ai`, producing train/validation
gap, >=3 observable subgroups, and >=5 FP / >=5 FN. State error threshold and
do not claim inferred domains/generators as ground truth. Initially analyze your
own model; update the final report with selected-model predictions at integration.

## AlexYen notebook reproducibility

The experiment notebook is `experiments/generalization/AlexYen_Train.ipynb`.
Run it from the repository root so that paths and outputs are consistent.

### Dependencies

The notebook's first executable cell installs the analysis dependencies into the
selected VS Code notebook kernel:

- NumPy
- pandas
- Matplotlib
- SciPy
- scikit-learn
- IPython/Jupyter (provided by the selected VS Code notebook kernel)

The pinned versions used by the generalization experiments are recorded in
`experiments/generalization/requirements.txt`. The notebook requires Python 3
and does not require pretrained models, external datasets, or an LLM API.

### Input data

Place the original competition files in `data/raw/`. At minimum, the notebook
requires:

- `data/raw/train.csv`
- `data/raw/test.csv`

`data/raw/unlabeled.csv` is used for the pseudo-labeling section when present.
The notebook locates `data/raw` automatically when VS Code starts it from either
the repository root or the notebook directory.

### Random seeds

- Global NumPy seed, train/validation split, SVM models, learning-curve models,
  and stylometric model: `42`
- Three-fold shuffled cross-validation: `123`
- Train-versus-test distribution-shift diagnostic split: `17`

The fixed learning-curve sample sizes are `500`, `1000`, `2000`, `5000`, and
`8000`, all taken from the same seeded training split.

### Exact run commands

From the repository root, open the notebook in VS Code:

```bash
code experiments/generalization/AlexYen_Train.ipynb
```

Select the intended local Python kernel, choose **Restart**, and then choose
**Run All**. The first executable cell installs the required Python packages in
that kernel. Run all cells in order without changing the seed or model-selection
settings when reproducing the reported results.

### Output paths

Validation tables, diagnostic metrics, and plots are displayed in and saved
with the executed notebook when the notebook is saved:

```text
experiments/generalization/AlexYen_Train.ipynb
```

The final cells write submission predictions to:

```text
data/raw/submission.csv
data/raw/submission_<model>_pseudo_labeled_cv<score>[_style_blend_<weight>].csv
```

With the recorded default selection, `<model>` is `combined_svm_C1`. The
descriptive filename's cross-validation score and optional style-blend suffix
are generated from that run, so the exact descriptive filename is printed by
the final notebook cell.
