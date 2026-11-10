"""Integrity checks and reproducible partitions for the supplied competition CSVs.

The actual files contain id/text/label (train and starter), id/text (unlabeled
and test), and id/label (sample submission). IDs are never model features.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

SEED = 20261023
SIZES = (500, 1000, 2000, 5000, 8000)


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def load_data(path, *, labeled=True, id_col='id', text_col='text', label_col='label', expected_rows=None):
    """Read all cells as strings, then validate and convert the binary labels."""
    path = Path(path)
    columns = [id_col, text_col] + ([label_col] if labeled else [])
    if len(set(columns)) != len(columns):
        raise ValueError('ID, text, and label columns must be distinct')
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f'{path.name}: missing columns {sorted(missing)}')
    if not labeled and label_col in frame:
        raise ValueError(f'{path.name}: expected an unlabeled text file')
    frame = frame[columns].copy()
    frame.columns = ['id', 'text'] + (['label'] if labeled else [])
    if frame.empty or (expected_rows is not None and len(frame) != expected_rows):
        raise ValueError(f'{path.name}: expected {expected_rows or "nonzero"} rows, got {len(frame)}')
    for column in ('id', 'text'):
        if frame[column].isna().any() or frame[column].str.strip().eq('').any():
            raise ValueError(f'{path.name}: blank/null {column}')
    if frame.id.duplicated().any():
        raise ValueError(f'{path.name}: duplicate IDs')
    if labeled:
        labels = frame.label.str.strip()
        if not labels.isin(['0', '1']).all() or set(labels) != {'0', '1'}:
            raise ValueError(f'{path.name}: require both binary labels 0=human, 1=AI')
        frame['label'] = labels.astype(int)
        if expected_rows in (10000, 2000) and (frame.label.value_counts() != expected_rows // 2).any():
            raise ValueError(f'{path.name}: supplied labeled file must be class balanced')
    frame.attrs.update(source_sha256=file_sha256(path), source_name=path.name,
                       schema={'id': id_col, 'text': text_col, 'label': label_col if labeled else None})
    return frame


def _counts(frame):
    return {str(k): int((frame.label == k).sum()) for k in (0, 1)}


def normalized_text(text):
    return ' '.join(text.split())


def prefix_groups(frame):
    """Observable shared openings, not inferred source/domain/generator labels."""
    keys = frame.text.map(normalized_text).str[:80]
    return [group.tolist() for _, group in frame.groupby(keys, sort=True).groups.items() if len(group) > 1]


def get_or_create_split(frame, path, *, seed=SEED, validation_size=2000):
    """Preserve exact ordered split IDs; refuse stale data or modified membership."""
    path = Path(path)
    if 'label' not in frame or 'source_sha256' not in frame.attrs:
        raise ValueError('Load a labeled CSV with load_data first')
    fingerprint = _hash(frame.sort_values('id')[['id', 'text', 'label']].values.tolist())
    expected = {'version': 1, 'seed': seed, 'validation_size': validation_size,
                'rows': len(frame), 'source_sha256': frame.attrs['source_sha256'], 'data_sha256': fingerprint}
    if path.exists():
        manifest = json.loads(path.read_text())
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f'Saved split {key} changed; never silently reset the holdout')
        training_ids, validation_ids = manifest['train_ids'], manifest['validation_ids']
    else:
        training_ids, validation_ids = train_test_split(frame.id.tolist(), test_size=validation_size,
                                                       stratify=frame.label, random_state=seed)
        manifest = dict(expected, source_name=frame.attrs.get('source_name'),
                        strategy='fixed stratified random split; lexical overlap evaluated separately without retuning',
                        train_ids=training_ids, validation_ids=validation_ids)
    if len(training_ids) != len(frame) - validation_size or len(validation_ids) != validation_size:
        raise ValueError('Saved split has incorrect sizes')
    combined = training_ids + validation_ids
    if len(set(combined)) != len(frame) or set(combined) != set(frame.id):
        raise ValueError('Split IDs overlap, are missing, duplicated, or unknown')
    indexed = frame.set_index('id', drop=False)
    training = indexed.loc[training_ids].reset_index(drop=True).copy()
    validation = indexed.loc[validation_ids].reset_index(drop=True).copy()
    counts = {'train': _counts(training), 'validation': _counts(validation)}
    membership_hash = _hash([training_ids, validation_ids])
    if path.exists():
        if manifest['split_sha256'] != membership_hash or manifest['class_counts'] != counts:
            raise ValueError('Saved split membership/order or labels were changed')
    else:
        manifest.update(class_counts=counts, split_sha256=membership_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2) + '\n')
    return training, validation


def nested_stratified_subsets(train, sizes=SIZES, *, seed=SEED):
    """Seeded class-wise permutations interleaved into nested balanced prefixes."""
    sizes = tuple(sizes)
    if not sizes or len(set(sizes)) != len(sizes) or min(sizes) < 2 or max(sizes) > len(train):
        raise ValueError('Subset sizes must be unique integers between 2 and training size')
    if set(train.label) != {0, 1}:
        raise ValueError('Training subsets require both classes')
    rng = np.random.default_rng(seed)
    permutations = {k: rng.permutation(np.flatnonzero(train.label.to_numpy() == k)) for k in (0, 1)}
    used = {0: 0, 1: 0}
    ordering = []
    for n in range(1, max(sizes) + 1):
        target_zero = (2 * n * len(permutations[0]) + len(train)) // (2 * len(train))
        label = 0 if target_zero > used[0] else 1
        ordering.append(int(permutations[label][used[label]]))
        used[label] += 1
    return {n: train.iloc[ordering[:n]].reset_index(drop=True).copy() for n in sizes}


def _duplicates(frame, mode):
    text = frame.text if mode == 'raw' else frame.text.map(normalized_text)
    if mode == 'casefold':
        text = text.str.casefold()
    counts = text.value_counts()
    result = {'groups': int((counts > 1).sum()), 'extra_rows': int(text.duplicated().sum())}
    if 'label' in frame:
        result['conflicting_label_groups'] = int((frame.groupby(text).label.nunique() > 1).sum())
    return result


def _summary(frame):
    words = frame.text.str.split().str.len()
    result = {'rows': len(frame), 'source_sha256': frame.attrs.get('source_sha256'),
              'words': {'min': int(words.min()), 'mean': float(words.mean()), 'median': float(words.median()), 'max': int(words.max())},
              'duplicates': {kind: _duplicates(frame, kind) for kind in ('raw', 'whitespace', 'casefold')}}
    if 'label' in frame:
        result['class_counts'] = _counts(frame)
        result['mean_words_by_label'] = {str(k): float(words[frame.label == k].mean()) for k in (0, 1)}
    return result


def _overlap(left, right):
    shared = set(left.text.map(normalized_text)) & set(right.text.map(normalized_text))
    return {'shared_ids': len(set(left.id) & set(right.id)), 'normalized_texts': len(shared),
            'left_rows': int(left.text.map(normalized_text).isin(shared).sum()),
            'right_rows': int(right.text.map(normalized_text).isin(shared).sum())}


def lexical_audit(frame, validation_ids=(), thresholds=(0.5, 0.7, 0.9, 0.95)):
    """Exact all-pairs word-trigram Jaccard for auditing, never model features.

    Sparse binary intersections have no hashing collisions. Both documents' full
    shingle counts enter the union. This cannot rule out paraphrased shared sources.
    """
    vectorizer = CountVectorizer(binary=True, ngram_range=(3, 3), dtype=np.float32, token_pattern=r'(?u)\b\w+\b')
    matrix = vectorizer.fit_transform(frame.text)
    lengths = np.asarray(matrix.sum(axis=1)).ravel()
    transposed = matrix.T.tocsr()
    in_validation = frame.id.isin(set(validation_ids)).to_numpy()
    counts = {str(t): 0 for t in thresholds}
    crossing = {str(t): 0 for t in thresholds}
    sensitivity_rows = set()
    top = []
    for start in range(0, len(frame), 100):
        intersections = (matrix[start:start + 100] @ transposed).tocoo()
        row, col = intersections.row + start, intersections.col
        keep = row < col
        row, col, common = row[keep], col[keep], intersections.data[keep]
        similarity = common / (lengths[row] + lengths[col] - common)
        for threshold in thresholds:
            selected = similarity >= threshold
            counts[str(threshold)] += int(selected.sum())
            crossing[str(threshold)] += int((in_validation[row[selected]] != in_validation[col[selected]]).sum())
        selected = similarity >= min(thresholds)
        top.extend((float(score), int(i), int(j)) for score, i, j in zip(similarity[selected], row[selected], col[selected]))
        for i, j in zip(row[selected], col[selected]):
            if in_validation[i] != in_validation[j]:
                sensitivity_rows.add(int(i if in_validation[i] else j))
    groups = prefix_groups(frame)
    for group in groups:
        if any(in_validation[i] for i in group) and not all(in_validation[i] for i in group):
            sensitivity_rows.update(i for i in group if in_validation[i])
    return {'method': 'Exact lowercase word-trigram set Jaccard across all supplied labeled rows; audit only',
            'vocabulary_size': len(vectorizer.vocabulary_), 'pair_counts': counts,
            'cross_split_pair_counts': crossing,
            'top_pairs': [{'id_a': str(frame.id.iloc[i]), 'id_b': str(frame.id.iloc[j]), 'jaccard': score,
                           'cross_split': bool(in_validation[i] != in_validation[j])} for score, i, j in sorted(top, reverse=True)[:25]],
            'shared_80_character_prefix': {'groups': len(groups), 'rows': sum(map(len, groups)),
                'cross_split_groups': sum(any(in_validation[i] for i in group) and not all(in_validation[i] for i in group) for group in groups),
                'id_groups': [frame.loc[group, 'id'].tolist() for group in groups]},
            'sensitivity_excluded_validation_ids': sorted(str(frame.id.iloc[i]) for i in sensitivity_rows),
            'limitations': 'Shared openings/lexical overlap are observable proxies, not known source or domain identities. Paraphrases and shared prompts can remain undetected.'}


def audit_data(frame, train, validation, *, unlabeled=None, test=None, starter=None):
    """Return metadata/aggregate evidence; never persist raw passages."""
    result = {'labeled': _summary(frame), 'train': _summary(train), 'validation': _summary(validation),
              'train_validation_overlap': _overlap(train, validation),
              'lexical': lexical_audit(frame.reset_index(drop=True), validation.id)}
    excluded_ids = result['lexical']['sensitivity_excluded_validation_ids']
    retained = validation[~validation.id.isin(excluded_ids)]
    result['source_overlap_sensitivity'] = {
        'excluded_validation_ids': excluded_ids, 'excluded_rows': len(excluded_ids),
        'retained_rows': len(retained), 'retained_class_counts': _counts(retained),
        'rule': 'Exclude validation rows sharing a whitespace-normalized first-80-character prefix OR word-trigram Jaccard >=0.50 with a training row.',
        'interpretation': 'Related-text sensitivity only. Recipe templates may trigger overlap without common source; source identity is unavailable.',
        'selection_policy': 'Use the complete fixed 2000-row validation AUROC for selection; never tune on the sensitivity score.'}
    for name, other in (('unlabeled', unlabeled), ('test', test)):
        if other is not None:
            result[name] = _summary(other)
            result[f'labeled_{name}_overlap'] = _overlap(frame, other)
            result[f'validation_{name}_overlap'] = _overlap(validation, other)
    if unlabeled is not None and test is not None:
        result['unlabeled_test_overlap'] = _overlap(unlabeled, test)
    if starter is not None:
        indexed = frame.set_index('id')
        ids_valid = set(starter.id).issubset(indexed.index)
        matches = ids_valid and starter.set_index('id')[['text', 'label']].equals(indexed.loc[starter.id, ['text', 'label']])
        result['starter'] = {'rows': len(starter), 'source_sha256': starter.attrs.get('source_sha256'),
                             'class_counts': _counts(starter), 'exact_labeled_subset': bool(matches)}
        if not matches:
            raise ValueError('Starter rows do not exactly match the supplied training file')
    return result
