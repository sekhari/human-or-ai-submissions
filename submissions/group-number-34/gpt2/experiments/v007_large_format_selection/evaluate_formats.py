"""Prepare fixed format stress tests and summarize predictions on matched rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
from pathlib import Path


FORMAT_RULES = {
    'original': 'Preserve the original parsed CSV text exactly.',
    'newlines_only': 'Replace each contiguous run of CR or LF characters with one ASCII space; preserve all other characters.',
    'all_whitespace': 'Replace each Unicode whitespace run with one ASCII space and strip leading/trailing whitespace.',
}


def transform_text(text, variant):
    if variant == 'original':
        return text
    if variant == 'newlines_only':
        return re.sub(r'[\r\n]+', ' ', text)
    if variant == 'all_whitespace':
        return re.sub(r'\s+', ' ', text).strip()
    raise ValueError(f'Unknown format variant: {variant}')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def load_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        columns, rows = reader.fieldnames, list(reader)
    if not rows or not columns or 'id' not in columns:
        raise ValueError(f'Missing rows or IDs: {path}')
    if len({row['id'] for row in rows}) != len(rows) or any(not row['id'] for row in rows):
        raise ValueError(f'Duplicate or empty IDs: {path}')
    return columns, rows


def prepare_format_inputs(source_csv, output_dir):
    """Generate deterministic diagnostics without changing the source file."""
    source_csv, output_dir = Path(source_csv).resolve(), Path(output_dir)
    columns, rows = load_csv(source_csv)
    if not {'text', 'label'} <= set(columns):
        raise ValueError('Format diagnostics require id, text, and label columns.')
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {'original': source_csv}
    changed = {'original': 0}
    for variant in FORMAT_RULES:
        if variant == 'original':
            continue
        transformed = [{**row, 'text': transform_text(row['text'], variant)} for row in rows]
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=columns, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(transformed)
        content = buffer.getvalue().encode('utf-8')
        path = output_dir / f'{variant}.csv'
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f'An existing format diagnostic input changed: {path}')
        if not path.exists():
            temporary = path.with_name(path.name + '.tmp')
            temporary.write_bytes(content)
            os.replace(temporary, path)
        inputs[variant] = path
        changed[variant] = sum(a['text'] != b['text'] for a, b in zip(rows, transformed))
    manifest = {'source_sha256': digest(source_csv), 'rows': len(rows), 'rules': FORMAT_RULES,
                'changed_rows': changed, 'input_sha256': {variant: digest(path) for variant, path in inputs.items()},
                'purpose': 'Paired format diagnostics evaluated after checkpoint and schedule selection.'}
    path = output_dir / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('The format-input manifest differs from its frozen specification.')
    atomic_json(path, manifest)
    return inputs


def prediction_rows(path):
    columns, rows = load_csv(path)
    if columns != ['id', 'label', 'probability']:
        raise ValueError(f'Expected id,label,probability columns: {path}')
    for row in rows:
        row['label'], row['probability'] = int(row['label']), float(row['probability'])
        if row['label'] not in (0, 1) or not math.isfinite(row['probability']) or not 0 <= row['probability'] <= 1:
            raise ValueError(f'Invalid label or probability: {path}')
    if {row['label'] for row in rows} != {0, 1}:
        raise ValueError('Both label classes are required.')
    return rows


def score(rows):
    import numpy as np
    from sklearn.metrics import log_loss, roc_auc_score
    labels = np.asarray([row['label'] for row in rows])
    probabilities = np.asarray([row['probability'] for row in rows])
    predicted = probabilities >= 0.5
    return {'auc': float(roc_auc_score(labels, probabilities)),
            'bce': float(log_loss(labels, probabilities, labels=[0, 1])),
            'human_fpr_at_0_5': float(predicted[labels == 0].mean()),
            'ai_fnr_at_0_5': float((~predicted[labels == 1]).mean()),
            'false_positives': int(((labels == 0) & predicted).sum()),
            'false_negatives': int(((labels == 1) & ~predicted).sum())}


def require_matched(reference, candidate):
    if [(r['id'], r['label']) for r in reference] != [(r['id'], r['label']) for r in candidate]:
        raise ValueError('Paired format comparisons require identical IDs, labels, and order.')


def summarize_formats(evaluation_dir):
    directory = Path(evaluation_dir)
    predictions = {variant: prediction_rows(directory / variant / 'validation_predictions.csv')
                   for variant in FORMAT_RULES}
    original = predictions['original']
    variants = {}
    for variant, rows in predictions.items():
        require_matched(original, rows)
        delta = [abs(a['probability'] - b['probability']) for a, b in zip(original, rows)]
        flips = sum((a['probability'] >= 0.5) != (b['probability'] >= 0.5) for a, b in zip(original, rows))
        variants[variant] = {'metrics': score(rows),
            'mean_absolute_probability_change_from_original': sum(delta) / len(delta),
            'threshold_flips_from_original': flips,
            'prediction_sha256': digest(directory / variant / 'validation_predictions.csv')}
    return {'status': 'complete', 'rows': len(original), 'formats': variants, 'format_rules': FORMAT_RULES,
            'primary_metric': 'original.auc', 'diagnostic_variants': ['newlines_only', 'all_whitespace'],
            'used_for_checkpoint_or_schedule_selection': False,
            'interpretation': 'Paired evaluation of the same validation documents in three formats. Training uses partial newline replacement; the diagnostic variants use complete newline or whitespace collapsing.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation_dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.output, summarize_formats(args.evaluation_dir))
