"""Prepare the fixed validation split and the recorded training subsets."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

TRAINING_SIZES = (500, 1000, 2000, 5000, 8000)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    content = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(content).hexdigest()


def read_csv(path):
    csv.field_size_limit(16 * 1024 * 1024)
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        columns, rows = reader.fieldnames, list(reader)
    if not columns or not rows or 'id' not in columns:
        raise ValueError(f'Expected a nonempty CSV with an id column: {path}')
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError(f'Duplicate IDs: {path}')
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f'Incomplete CSV row: {path}')
    return columns, rows


def write_csv(path, columns, rows, newline='\r\n'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, quoting=csv.QUOTE_ALL,
                                lineterminator=newline)
        writer.writeheader()
        writer.writerows({name: row[name] for name in columns} for row in rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n',
                          encoding='utf-8')


def nested_subsets(development, seed):
    ranks = {
        label: sorted((row for row in development if row['label'] == label),
                      key=lambda row: fingerprint(['q2_nested_v1', seed, label, row['id']]))
        for label in ('0', '1')
    }
    return {
        size: [row for pair in zip(ranks['0'][:size // 2], ranks['1'][:size // 2])
               for row in pair]
        for size in TRAINING_SIZES
    }


def adaptation_split(rows, method, seed):
    if method == 'ranked':
        validation = sorted(rows, key=lambda row: fingerprint(
            ['q3_lm_split_v1', seed, row['id']]))[:300]
    elif method == 'random':
        from sklearn.model_selection import train_test_split
        _, validation_indices = train_test_split(
            list(range(len(rows))), test_size=300, shuffle=True, random_state=seed)
        validation = [rows[index] for index in sorted(validation_indices)]
    else:
        raise ValueError(f'Unknown adaptation split: {method}')
    held_ids = {row['id'] for row in validation}
    training = [row for row in rows if row['id'] not in held_ids]
    return training, validation


def prepare_data(data_dir, validation_ids, output_dir, experiment, seed=42):
    data_dir, validation_ids, output_dir = map(Path, (data_dir, validation_ids, output_dir))
    if seed != 42:
        raise ValueError('The recorded experiments use seed 42.')
    columns, full = read_csv(data_dir / 'train.csv')
    if not {'id', 'text', 'label'} <= set(columns):
        raise ValueError('train.csv requires id, text, and label columns.')
    if len(full) != 10000 or Counter(row['label'] for row in full) != {'0': 5000, '1': 5000}:
        raise ValueError('train.csv must contain 5,000 human and 5,000 AI passages.')
    if any(not row['text'] for row in full):
        raise ValueError('Training passages must be nonempty.')
    _, held_rows = read_csv(validation_ids)
    held_ids = [row['id'] for row in held_rows]
    by_id = {row['id']: row for row in full}
    if len(held_ids) != 2000 or not set(held_ids) < set(by_id):
        raise ValueError('The fixed validation file must select 2,000 training IDs.')
    held = [by_id[identifier] for identifier in held_ids]
    if Counter(row['label'] for row in held) != {'0': 1000, '1': 1000}:
        raise ValueError('The fixed validation split must contain 1,000 passages per class.')
    held_set = set(held_ids)
    development = [row for row in full if row['id'] not in held_set]
    small_model = experiment in {'learning-curve', 'unlabeled'}
    newline = '\r\n' if small_model else '\n'
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f'Choose an empty preparation directory: {output_dir}')
    write_csv(output_dir / 'fixed_validation.csv', ['id', 'text', 'label'], held, newline)
    if small_model:
        write_csv(output_dir / 'fixed_validation_ids.csv', ['id'], held_rows, newline)
    else:
        shutil.copyfile(validation_ids, output_dir / 'fixed_validation_ids.csv')
    source_files = {'train.csv': sha256(data_dir / 'train.csv'),
                    'fixed_validation_ids.csv': sha256(validation_ids)}
    if small_model:
        subsets = nested_subsets(development, seed)
        sizes = TRAINING_SIZES if experiment == 'learning-curve' else (8000,)
        for size in sizes:
            write_csv(output_dir / f'train_{size}.csv', ['id', 'text', 'label'],
                      subsets[size], newline)
            write_csv(output_dir / f'train_{size}_ids.csv', ['id'], subsets[size], newline)
        training_order = 'Class-balanced SHA256 ranks with interleaved classes.'
    else:
        write_csv(output_dir / 'train_8000.csv', ['id', 'text', 'label'], development, newline)
        write_csv(output_dir / 'train_8000_ids.csv', ['id'], development, newline)
        training_order = 'Original row order after removing fixed validation IDs.'
    adaptation = None
    if experiment != 'learning-curve':
        unlabeled_columns, unlabeled = read_csv(data_dir / 'unlabeled.csv')
        if set(unlabeled_columns) != {'id', 'text'} or len(unlabeled) != 3000:
            raise ValueError('unlabeled.csv requires 3,000 id,text rows.')
        labeled_texts = {row['text'] for row in full}
        seen, retained, exclusions = set(), [], []
        for row in unlabeled:
            if not row['text']:
                raise ValueError('Unlabeled passages must be nonempty.')
            if row['text'] in labeled_texts or row['text'] in seen:
                exclusions.append(row['id'])
            else:
                retained.append(row)
                seen.add(row['text'])
        if exclusions:
            raise ValueError('The recorded adaptation split requires 3,000 unique unlabeled passages without labeled-text overlaps.')
        method = 'ranked' if small_model else 'random'
        lm_train, lm_validation = adaptation_split(retained, method, seed)
        write_csv(output_dir / 'unlabeled.csv', ['id', 'text'], retained, newline)
        write_csv(output_dir / 'lm_train_ids.csv', ['id'], lm_train, newline)
        write_csv(output_dir / 'lm_validation_ids.csv', ['id'], lm_validation, newline)
        source_files['unlabeled.csv'] = sha256(data_dir / 'unlabeled.csv')
        adaptation = {'training_rows': len(lm_train), 'validation_rows': len(lm_validation),
                      'split_method': method, 'exact_text_exclusions': len(exclusions)}
    if experiment == 'final':
        shutil.copyfile(data_dir / 'train.csv', output_dir / 'train_all.csv')
        for name in ('test.csv', 'sample_submission.csv'):
            source_files[name] = sha256(data_dir / name)
            shutil.copyfile(data_dir / name, output_dir / name)
        test_columns, test = read_csv(output_dir / 'test.csv')
        sample_columns, sample = read_csv(output_dir / 'sample_submission.csv')
        if set(test_columns) != {'id', 'text'} or sample_columns != ['id', 'label']:
            raise ValueError('Expected test id,text and sample-submission id,label columns.')
        if len(test) != 3000 or {row['id'] for row in test} != {row['id'] for row in sample}:
            raise ValueError('Test and sample-submission files must contain the same 3,000 IDs.')
    manifest = {
        'experiment': experiment, 'seed': seed, 'source_files_sha256': source_files,
        'training_partition_rows': 8000, 'validation_rows': 2000,
        'validation_class_counts': {'0': 1000, '1': 1000},
        'training_order': training_order, 'adaptation': adaptation,
        'raw_text_transform': 'identity',
        'files_sha256': {path.name: sha256(path) for path in sorted(output_dir.glob('*.csv'))},
    }
    write_json(output_dir / 'split_manifest.json', manifest)
    return manifest
