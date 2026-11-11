"""Which feature blocks help? One logreg per combo of blocks, C picked by 5-fold CV on train.

Run from submissions/group-number-10/:
    python -m logreg.ablation

Writes:
    results/logreg/metrics/logreg_<blocks>.json        one record per combo (train/val AUROC, time, C)
    results/logreg/predictions/logreg_<blocks>_val.csv  id,label,prob_ai for the 2,000 validation rows
    results/logreg/ablation.json                        the table printed at the end
"""

import json

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

from logreg import RESULTS_DIR, SEED
from logreg.features import BLOCKS, available_blocks
from logreg.model import folds, make_lr
from src.data import DATA_DIR, load_split, load_unlabeled
from src.evaluate import Timer, record_experiment

C_GRID = [1.0, 10.0, 30.0, 100.0, 300.0]

COMBOS = [
    ['words'],
    ['chars'],
    ['style'],
    ['lm'],
    ['words', 'style'],
    ['words', 'chars'],
    ['style', 'lm'],
    ['words', 'chars', 'style'],
    ['words', 'chars', 'style', 'lm'],
]

NOTES = ('tf-idf vocabulary fit on train + unlabeled.csv + test.csv text (no labels); '
         'C picked by 5-fold CV on the 8,000 train rows, val untouched until scoring')


def extra_vocab_texts():
    """unlabeled.csv + test.csv passages: only used to build the tf-idf vocabulary."""
    return [*load_unlabeled(DATA_DIR / 'unlabeled.csv')[1], *load_unlabeled(DATA_DIR / 'test.csv')[1]]


def pick_C(X, y):
    """5-fold CV on the training rows. Returns (best C, mean AUC, std AUC)."""
    best = None
    for C in C_GRID:
        scores = cross_val_score(make_lr(C), X, y, cv=folds(), scoring='roc_auc')
        if best is None or scores.mean() > best[1]:
            best = (C, scores.mean(), scores.std())
    return best


def bootstrap_ci(y, p, n=1000):
    """95% interval for AUC, by resampling rows with replacement."""
    rng = np.random.default_rng(SEED)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(aucs, [2.5, 97.5])


def main():
    split = load_split()
    vocab = extra_vocab_texts()
    y_train, y_val = split.train_labels, split.val_labels

    names = available_blocks()
    if 'lm' not in names:
        print('torch/transformers not installed -> skipping the lm block\n')

    # build every block once, reuse across combos
    print('building feature blocks...')
    built = {}
    for name in names:
        with Timer() as t:
            block = BLOCKS[name]()
            X_train = block.fit_transform(split.train_texts, vocab)
            X_val = block.transform(split.val_texts)
        built[name] = (X_train, X_val, t.seconds)

    print('running combos...')
    rows = []
    for combo in COMBOS:
        if not all(name in built for name in combo):
            continue
        tag = '+'.join(combo)
        X_train = sp.hstack([built[n][0] for n in combo]).tocsr()
        X_val = sp.hstack([built[n][1] for n in combo]).tocsr()
        C, cv_mean, cv_std = pick_C(X_train, y_train)

        with Timer() as t:
            lr = make_lr(C).fit(X_train, y_train)
        p_train, p_val = lr.predict_proba(X_train)[:, 1], lr.predict_proba(X_val)[:, 1]
        lo, hi = bootstrap_ci(y_val, p_val)

        # train time = building this combo's blocks + the final fit (not the C search)
        result = record_experiment(
            f'logreg_{tag}', split, p_train, p_val,
            t.seconds + sum(built[n][2] for n in combo), seed=SEED,
            hyperparameters={'blocks': combo, 'C': C, 'C_grid': C_GRID, 'cv_auc': cv_mean, 'cv_std': cv_std,
                             'n_cols': X_train.shape[1]},
            notes=NOTES, run_name=f'logreg_{tag}',
            metrics_dir=RESULTS_DIR / 'metrics', predictions_dir=RESULTS_DIR / 'predictions',
        )
        rows.append({'features': tag, 'n_cols': X_train.shape[1], 'best_C': C, 'cv_auc': cv_mean,
                     'cv_std': cv_std, 'train_auc': result.train_auc, 'val_auc': result.val_auc,
                     'val_ci': [float(lo), float(hi)], 'train_time_s': result.train_time_seconds})
        print(f'  done: {tag}')

    table = pd.DataFrame(rows)
    print()
    print(table.to_string(index=False, float_format='%.4f'))
    (RESULTS_DIR / 'ablation.json').write_text(json.dumps(rows, indent=1) + '\n')


if __name__ == '__main__':
    main()
