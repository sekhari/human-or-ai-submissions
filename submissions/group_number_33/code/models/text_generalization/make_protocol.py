"""Create training-only random and held-out-topic evaluation partitions."""
from itertools import combinations
import json
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from threadpoolctl import threadpool_limits

from .common import ROOT, OUT, PROTOCOL_FILE, sha256


def main():
    if PROTOCOL_FILE.exists():
        raise FileExistsError('Frozen protocol already exists; do not silently redefine it')
    started = perf_counter()
    raw = pd.read_csv(ROOT / 'data/train.csv')
    indices = np.arange(len(raw))
    development_indices, outer_indices = train_test_split(indices, test_size=.2, random_state=12, stratify=raw.label)
    development = raw.iloc[development_indices].reset_index(drop=True)
    tr, va = train_test_split(np.arange(len(development)), test_size=.25, random_state=2026, stratify=development.label)
    # Unsupervised topic groups define a stress partition; the classifier's
    # vectorizer remains independently fitted on each protocol's fitting rows.
    vec = TfidfVectorizer(max_features=12000, min_df=5, max_df=.8, stop_words='english', sublinear_tf=True)
    features = vec.fit_transform(development.text)
    reduced = Normalizer().fit_transform(TruncatedSVD(n_components=48, random_state=2026).fit_transform(features))
    groups = MiniBatchKMeans(n_clusters=8, random_state=2026, batch_size=1024, n_init=5).fit_predict(reduced)
    # Choose two clusters closest to 2000 examples and balanced labels, without
    # fitting or inspecting any AI detector's predictions.
    options = []
    for pair in combinations(range(8), 2):
        mask = np.isin(groups, pair)
        n = int(mask.sum())
        if n < 1000 or n > 3200:
            continue
        fraction = float(development.label.to_numpy()[mask].mean())
        if not .25 <= fraction <= .75:
            continue
        objective = abs(n-2000)/2000 + abs(fraction-.5)
        options.append((objective, pair, mask))
    _, pair, mask = min(options, key=lambda item:(item[0],item[1]))
    protocols = dict(random=dict(train=tr.tolist(), validation=va.tolist()),
                     topic=dict(train=np.flatnonzero(~mask).tolist(), validation=np.flatnonzero(mask).tolist()))
    report = dict(seed=2026, train_sha256=sha256(ROOT/'data/train.csv'),
                  development_indices=development_indices.tolist(), outer_indices=outer_indices.tolist(),
                  protocols=protocols, topic_groups=groups.tolist(), heldout_topic_groups=list(pair),
                  grouping='Word TF-IDF +48D normalized SVD +8 minibatch k-means; unsupervised on 8000 only',
                  selection='Mean AUROC across random and held-out-topic protocols; worst score breaks ties',
                  outer_scope='Original 2000 rows excluded from candidate search; reused development benchmark, not an untouched test',
                  no_test_or_unlabeled_data_used_for_grouping=True,
                  elapsed_seconds=perf_counter()-started)
    OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL_FILE.write_text(json.dumps(report, indent=2))
    summary={}
    for name, split in protocols.items():
        labels=development.label.iloc[split['validation']]
        summary[name]=dict(training_rows=len(split['train']), validation_rows=len(labels),
                           validation_counts={str(k):int(v) for k,v in labels.value_counts().items()})
    (OUT/'protocol_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
