# Sections 4 and 5

These scripts use Dave's selected model: 145 style features followed by an
ExtraTrees classifier with 800 trees. Both scripts use the same stratified
8,000-row training partition and 2,000-row validation partition.

```text
common.py          shared data split, feature extraction, and model
learning_curve.py  section 4: effect of training-set size
unlabeled.py       section 5: confidence-based pseudo-labeling
```

Run section 4 from the repository root:

```sh
python -m models.dave_extratrees.q4_q5.learning_curve
```

It trains the same model with 500, 1,000, 2,000, 5,000, and 8,000 examples.
It writes `learning_curve.csv` and `learning_curve.png` to `outputs/q4_q5/`.

Run section 5:

```sh
python -m models.dave_extratrees.q4_q5.unlabeled
```

It first trains the supervised baseline on 8,000 labeled examples. It then
keeps unlabeled examples where the baseline predicts at least 0.95 probability
for either class, adds those pseudo-labeled examples to the training set, and
trains a second model. Both models are scored on the same 2,000 validation rows.

It writes `unlabeled_results.csv` and `pseudo_labels.csv` to `outputs/q4_q5/`.
The validation labels are used only for final scoring, never for pseudo-label
selection or model fitting.
