"""Portable replay of PR #1's notebook, using the original saved split.

Preserves original Day 2 measurements under results/day2. New measurements go
to a separate directory; the 11-feature model is not Day 1's 28-feature model.
"""
import argparse
import json
import re
import string
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from data_utils import get_or_create_split, load_data
from models import SEED, build_model


def style_features(texts):
    rows = []
    for text in texts:
        words = re.findall(r'\b\w+\b', text.lower())
        sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
        nc, nw = max(len(text), 1), max(len(words), 1)
        rows.append([len(text), len(words), len(sentences), len(text) / nw,
                     float(np.mean([len(w) for w in words])) if words else 0.,
                     len(set(words)) / nw, sum(ch in string.punctuation for ch in text) / nc,
                     sum(ch in ',;:' for ch in text) / nc, sum(ch.isdigit() for ch in text) / nc,
                     sum(ch.isupper() for ch in text) / nc, text.count('\n') / nc])
    return np.asarray(rows, dtype=float)


def run(data_dir, split, out):
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'replay.json').exists():
        raise FileExistsError('Use a fresh output directory to preserve measured evidence')
    data = load_data(data_dir / 'train.csv', expected_rows=10000)
    train, val = get_or_create_split(data, split)
    unlabeled = load_data(data_dir / 'unlabeled.csv', labeled=False, expected_rows=3000)
    rows, probabilities = [], {}

    def record(name, model, seconds):
        tr, va = model.predict_proba(train.text.tolist())[:, 1], model.predict_proba(val.text.tolist())[:, 1]
        rows.append({'method': name, 'train_auroc': float(roc_auc_score(train.label, tr)),
                     'validation_auroc': float(roc_auc_score(val.label, va)),
                     'validation_log_loss': float(log_loss(val.label, va)), 'fit_seconds': seconds})
        probabilities[name] = va

    with threadpool_limits(limits=2):
        base = build_model({'kind': 'char_lr', 'C': 3.})
        start = time.perf_counter()
        base.fit(train.text.tolist(), train.label)
        base_seconds = time.perf_counter() - start
        record('supervised_character', base, base_seconds)
        pu = base.predict_proba(unlabeled.text.tolist())[:, 1]
        human, ai = np.flatnonzero(pu <= .02), np.flatnonzero(pu >= .98)
        human, ai = human[np.argsort(pu[human])][:750], ai[np.argsort(-pu[ai])][:750]
        selected = np.r_[human, ai]
        pseudo_labels = np.r_[np.zeros(len(human), dtype=int), np.ones(len(ai), dtype=int)]
        pseudo = build_model({'kind': 'char_lr', 'C': 3.})
        start = time.perf_counter()
        pseudo.fit(train.text.tolist() + unlabeled.iloc[selected].text.tolist(), np.r_[train.label, pseudo_labels])
        record('pseudo_character', pseudo, time.perf_counter() - start)
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True)
        # Time the entire second feature/model pipeline, unlike notebook's classifier-only timer.
        start = time.perf_counter()
        xw = vec.fit_transform(train.text)
        scale = StandardScaler()
        xs = scale.fit_transform(style_features(train.text))
        word = LogisticRegression(C=1., solver='liblinear', max_iter=2000, random_state=SEED)
        word.fit(hstack([xw, csr_matrix(.1 * xs)], format='csr'), train.label)
        word_seconds = time.perf_counter() - start
        pw = word.predict_proba(hstack([vec.transform(val.text), csr_matrix(.1 * scale.transform(style_features(val.text)))], format='csr'))[:, 1]
        pt = word.predict_proba(hstack([xw, csr_matrix(.1 * xs)], format='csr'))[:, 1]
        blend = .5 * probabilities['supervised_character'] + .5 * pw
        pbase_train = base.predict_proba(train.text.tolist())[:, 1]
        rows.append({'method': 'compact_blend', 'train_auroc': float(roc_auc_score(train.label, .5 * pbase_train + .5 * pt)),
                     'validation_auroc': float(roc_auc_score(val.label, blend)),
                     'validation_log_loss': float(log_loss(val.label, blend)),
                     'fit_seconds': base_seconds + word_seconds})
        probabilities['compact_blend'] = blend
    pd.DataFrame(rows).to_csv(out / 'comparison.csv', index=False)
    pred = pd.DataFrame({'id': val.id, 'label': val.label, **probabilities})
    pred.to_csv(out / 'validation_predictions.csv', index=False)
    result = {'seed': SEED, 'split': str(split), 'pseudo_human': len(human), 'pseudo_ai': len(ai),
              'results': rows, 'timing_note': 'Replay times feature extraction and fitting; original Day 2 second-component timer excluded preprocessing.'}
    (out / 'replay.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--split', type=Path, default=Path('results/split.json'))
    parser.add_argument('--out', type=Path, default=Path('results/day3/day2_replay'))
    args = parser.parse_args()
    run(args.data, args.split, args.out)
