"""Stacked logreg: one logreg per block + a meta logreg, words C swept, then a test submission.

Run from submissions/group-number-10/:
    python -m logreg.stack

Test-like validation: val comes from the same pool as train, so it can't see the train -> test
shift (C=300 scored 0.997 on val but 0.948 on Kaggle public). A classifier trained to tell train
from test text scores how test-like each val passage is; the TESTLIKE_N most test-like val
passages stand in for the leaderboard when picking the words C.
Caveat: that pick uses val, so the chosen config's val AUROC is optimistic.

Writes:
    results/logreg/metrics/logreg_stacked_wordsC<C>.json        one record per words C
    results/logreg/predictions/logreg_stacked_wordsC<C>_val.csv
    results/logreg/stack_sweep.json                             the table printed below
    results/logreg/submissions/logreg_stacked_wordsC<C>.csv     id,label for test.csv (refit on all 10,000)
"""

import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from logreg import RESULTS_DIR, SEED
from logreg.ablation import bootstrap_ci
from logreg.features import available_blocks, clean_text
from logreg.model import StackedLogReg
from src.data import DATA_DIR, load_split, load_unlabeled
from src.evaluate import Timer, record_experiment

BLOCK_C = {'words': 300.0, 'chars': 10.0, 'style': 300.0, 'lm': 1.0}   # best C per block from the ablation
WORDS_C_GRID = [0.3, 1.0, 3.0, 10.0, 30.0, 300.0]
TESTLIKE_N = 500

NOTES = ('stacked: per-block logreg (out-of-fold logits) -> meta logreg; tf-idf vocabulary fit on '
         'train + unlabeled.csv + test.csv text (no labels); words C chosen by AUROC on the 500 most '
         'test-like val passages, so the chosen run\'s val AUROC is optimistic')


def testlike_mask(train_texts, val_texts, test_texts):
    texts = [clean_text(t) for t in [*train_texts, *test_texts]]
    is_test = np.r_[np.zeros(len(train_texts)), np.ones(len(test_texts))]

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    adv = LogisticRegression(C=10, solver='liblinear').fit(vec.fit_transform(texts), is_test)
    test_likeness = adv.predict_proba(vec.transform([clean_text(t) for t in val_texts]))[:, 1]

    mask = np.zeros(len(val_texts), dtype=bool)
    mask[np.argsort(test_likeness)[-TESTLIKE_N:]] = True
    return mask


def main():
    split = load_split()
    unlabeled = load_unlabeled(DATA_DIR / 'unlabeled.csv')[1]
    test_ids, test_texts = load_unlabeled(DATA_DIR / 'test.csv')
    vocab = [*unlabeled, *test_texts]
    y_val = split.val_labels

    names = available_blocks()
    mask = testlike_mask(split.train_texts, split.val_texts, test_texts)

    print('stacking: fit on train, score on val...')
    rows = []
    for C in WORDS_C_GRID:
        block_C = {n: (C if n == 'words' else BLOCK_C[n]) for n in names}
        with Timer() as t:
            model = StackedLogReg(block_C, vocab).fit(split.train_texts, split.train_labels)
        p_val = model.predict_proba(split.val_texts)
        p_train = model.predict_proba(split.train_texts)

        testlike_auc = roc_auc_score(y_val[mask], p_val[mask])
        t_lo, t_hi = bootstrap_ci(y_val[mask], p_val[mask])
        weights = model.meta_weights()
        result = record_experiment(
            'logreg_stacked', split, p_train, p_val, t.seconds, seed=SEED,
            hyperparameters={**model.get_config(), 'meta_weights': weights, 'testlike_auc': testlike_auc,
                             'testlike_n': TESTLIKE_N},
            notes=NOTES, run_name=f'logreg_stacked_wordsC{C:g}',
            metrics_dir=RESULTS_DIR / 'metrics', predictions_dir=RESULTS_DIR / 'predictions',
        )
        rows.append({'words_C': C, 'train_auc': result.train_auc, 'val_auc': result.val_auc,
                     'testlike_auc': testlike_auc, 'testlike_ci': [float(t_lo), float(t_hi)],
                     'train_time_s': result.train_time_seconds, **{f'w_{n}': w for n, w in weights.items()}})
        print(f'  done: words C={C:g}')

    table = pd.DataFrame(rows)
    print(f'\ntest-like = the {TESTLIKE_N} most test-like val docs; w_* = meta weights')
    print(table.drop(columns='testlike_ci').to_string(index=False, float_format='%.4f'))
    (RESULTS_DIR / 'stack_sweep.json').write_text(json.dumps(rows, indent=1) + '\n')

    best_C = table.loc[table['testlike_auc'].idxmax(), 'words_C']
    print(f'\nbest words C by test-like AUC: {best_C:g}')

    print('\nrefit on train + val with that C, predict test...')
    block_C = {n: (best_C if n == 'words' else BLOCK_C[n]) for n in names}
    model = StackedLogReg(block_C, vocab).fit(
        split.train_texts + split.val_texts, np.r_[split.train_labels, split.val_labels])
    out = RESULTS_DIR / 'submissions' / f'logreg_stacked_wordsC{best_C:g}.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'id': test_ids, 'label': model.predict_proba(test_texts)}).to_csv(out, index=False)
    print(f'submission written: {out}')


if __name__ == '__main__':
    main()
