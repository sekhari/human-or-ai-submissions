"""Evaluate a saved classifier and export numerical diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from analysis import (FORMATS, classification_metrics, compare_formats, error_cases,
                      length_subgroups, read_predictions, transform_text, write_json)


def compare_saved(baseline_dir, candidate_dir, replicates=2000):
    baseline_dir, candidate_dir = Path(baseline_dir), Path(candidate_dir)
    def load(directory):
        return {name: read_predictions(directory / ('validation_predictions.csv' if name == 'original'
                       else f'formats/{name}.csv')) for name in FORMATS}
    comparison = compare_formats(load(baseline_dir), load(candidate_dir), replicates)
    comparison.to_csv(candidate_dir / 'comparison.csv', index=False)
    write_json(candidate_dir / 'paired_comparison.json',
               {'baseline_directory': str(baseline_dir.resolve()), 'seed': 42,
                'threshold': 0.5, 'bootstrap_replicates': replicates,
                'formats': comparison.to_dict(orient='records')})
    return comparison


def evaluate(args):
    from training_utils import (choose_device, choose_precision, encode_frame, load_model,
                                make_loader, make_tokenizer, read_frame, set_seed)
    from train_stage import evaluate_classifier
    set_seed(42)
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training = read_frame(args.train_csv, labeled=True)
    validation = read_frame(args.validation_csv, labeled=True)
    if set(training.id) & set(validation.id):
        raise ValueError('Training and validation IDs must be disjoint.')
    device = choose_device(args.device)
    precision = choose_precision(args.precision, device)
    tokenizer = make_tokenizer(args.checkpoint)
    model, loading = load_model(args.checkpoint, 'classifier', tokenizer, device)
    if loading['missing_keys'] or loading['unexpected_keys']:
        raise ValueError('The checkpoint must contain every classifier parameter.')
    metadata_path = args.checkpoint / 'training_metadata.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    max_length = metadata.get('max_length', 512)

    def predict(frame):
        encoded = encode_frame(frame, tokenizer, max_length, 'classifier')
        loader = make_loader(encoded, tokenizer, args.batch_size, shuffle=False)
        probabilities, measured = evaluate_classifier(model, loader, device, precision,
                                                        args.attention_backend, frame.label.to_numpy())
        result = frame[['id', 'label']].copy()
        result['probability'] = probabilities
        metrics = classification_metrics(frame.label, probabilities)
        metrics['bce_from_probabilities'] = metrics['bce']
        metrics['bce'] = measured['bce']
        return result, metrics

    result = {'status': 'complete', 'checkpoint': str(args.checkpoint.resolve()), 'seed': 42,
              'max_length': max_length, 'precision': precision, 'threshold': 0.5,
              'evaluation_text': 'original'}
    for name, frame in [('training', training), ('validation', validation)]:
        predictions, metrics = predict(frame)
        predictions.to_csv(args.output_dir / f'{name}_predictions.csv', index=False)
        result[name] = metrics
        if name == 'validation':
            validation_predictions = predictions
    joined = validation.merge(validation_predictions[['id', 'probability']], on='id',
                              how='left', validate='one_to_one')
    length_subgroups(joined).to_csv(args.output_dir / 'length_subgroups.csv', index=False)
    thresholds = []
    for threshold, suffix in [(0.5, '0_5'), (0.2, '0_2')]:
        error_cases(joined, threshold).to_csv(args.output_dir / f'errors_at_{suffix}.csv', index=False)
        thresholds.append(classification_metrics(joined.label, joined.probability, threshold))
    pd.DataFrame(thresholds).to_csv(args.output_dir / 'threshold_metrics.csv', index=False)
    if args.format_variants or args.baseline_dir:
        (args.output_dir / 'formats').mkdir(exist_ok=True)
        result['formats'] = {}
        original = validation_predictions.probability.to_numpy()
        for name in FORMATS:
            frame = validation.copy()
            frame['text'] = frame.text.map(lambda text: transform_text(text, name))
            predictions, metrics = (validation_predictions, dict(result['validation'])) if name == 'original' else predict(frame)
            predictions.to_csv(args.output_dir / 'formats' / f'{name}.csv', index=False)
            values = predictions.probability.to_numpy()
            metrics['changed_text_rows'] = int((frame.text != validation.text).sum())
            metrics['mean_probability_change'] = float(np.abs(values - original).mean())
            metrics['threshold_flips'] = int(((values >= 0.5) != (original >= 0.5)).sum())
            result['formats'][name] = metrics
    result['elapsed_seconds'] = time.perf_counter() - started
    write_json(args.output_dir / 'metrics.json', result)
    if args.baseline_dir:
        compare_saved(args.baseline_dir, args.output_dir)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--train-csv', type=Path)
    parser.add_argument('--validation-csv', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'auto'])
    parser.add_argument('--precision', default='bf16', choices=['auto', 'fp32', 'fp16', 'bf16'])
    parser.add_argument('--attention-backend', default='flash_or_math', choices=['flash', 'math', 'flash_or_math'])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--format-variants', action='store_true')
    parser.add_argument('--baseline-dir', type=Path)
    parser.add_argument('--compare-only', action='store_true')
    args = parser.parse_args()
    if args.compare_only:
        if args.baseline_dir is None:
            parser.error('--compare-only requires --baseline-dir.')
        comparison = compare_saved(args.baseline_dir, args.output_dir)
        print(comparison.to_json(orient='records', indent=2))
        return
    if any(value is None for value in (args.checkpoint, args.train_csv, args.validation_csv)):
        parser.error('Evaluation requires --checkpoint, --train-csv, and --validation-csv.')
    result = evaluate(args)
    print(json.dumps({'training_auc': result['training']['auc'],
                      'validation_auc': result['validation']['auc'],
                      'output_dir': str(args.output_dir)}, indent=2))


if __name__ == '__main__':
    main()
