"""Inspect the actual competition corpus before fitting a classifier."""
import argparse
import json
from pathlib import Path
import pandas as pd
from data_utils import load_data, get_or_create_split, audit_data, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument('--id-col', default='id')
    parser.add_argument('--text-col', default='text')
    parser.add_argument('--label-col', default='label')
    parser.add_argument('--preview-train', type=int, default=0)
    args = parser.parse_args()
    if args.preview_train < 0:
        parser.error('--preview-train must be nonnegative')
    schema = {'id_col': args.id_col, 'text_col': args.text_col, 'label_col': args.label_col}
    frame = load_data(args.data / 'train.csv', expected_rows=10000, **schema)
    train, validation = get_or_create_split(frame, args.results / 'split.json')
    additional = {}
    for name, filename, labeled, size in [('unlabeled', 'unlabeled.csv', False, 3000), ('test', 'test.csv', False, 3000), ('starter', 'train_starter_2000.csv', True, 2000)]:
        path = args.data / filename
        additional[name] = load_data(path, labeled=labeled, expected_rows=size, **schema) if path.exists() else None
    audit = audit_data(frame, train, validation, **additional)
    sample_path = args.data / 'sample_submission.csv'
    if sample_path.exists():
        sample = pd.read_csv(sample_path, dtype=str, keep_default_na=False)
        expected = [args.id_col, args.label_col]
        matches = sample.columns.tolist() == expected and additional['test'] is not None and sample[args.id_col].tolist() == additional['test'].id.tolist()
        audit['sample_submission'] = {'rows': len(sample), 'columns': sample.columns.tolist(), 'source_sha256': file_sha256(sample_path), 'columns_and_test_id_order_match': matches}
        if not matches:
            raise ValueError('Sample submission columns/ID order do not match the supplied test file')
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / 'audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
    output = {'audit_path': str(args.results / 'audit.json'), 'labeled': audit['labeled'],
              'train_validation_overlap': audit['train_validation_overlap'],
              'lexical_pair_counts': audit['lexical']['pair_counts'],
              'lexical_cross_split_pair_counts': audit['lexical']['cross_split_pair_counts'],
              'shared_prefix_groups': audit['lexical']['shared_80_character_prefix']['groups'],
              'shared_prefix_cross_split_groups': audit['lexical']['shared_80_character_prefix']['cross_split_groups'],
              'note': 'Inspection only; no classifier fit. Prefix groups are observable proxies, not source IDs.'}
    if args.preview_train:
        output['training_previews'] = [{'id': row.id, 'label': int(row.label), 'text_first_500_chars': row.text[:500]} for row in train.head(args.preview_train).itertuples(index=False)]
    print(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
