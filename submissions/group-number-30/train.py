"""Train a saved configuration for validation or refit on all 10,000 labeled rows."""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
from threadpoolctl import threadpool_limits

from data_utils import get_or_create_split, load_data
from experiment import THREADS, artifact, evaluate, write_json
from models import SEED, build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--full-data', action='store_true', help='Refit selected configuration on all labeled rows')
    parser.add_argument('--model-out', type=Path, default=Path('artifacts/model.joblib'))
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument('--resume', action='store_true', help='Verify/reuse identical validation run')
    args = parser.parse_args()
    data = load_data(args.train, expected_rows=10000)
    config = json.loads(args.config.read_text())
    if args.full_data:
        if args.model_out.exists():
            parser.error(f'Refusing to overwrite {args.model_out}; choose a new --model-out path')
        model = build_model(config, seed=SEED)
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=THREADS):
            warnings.simplefilter('always')
            start = time.perf_counter()
            model.fit(data.text.tolist(), data.label.to_numpy())
            seconds = time.perf_counter() - start
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        # Verify serialization before exposing a completed model path. A successful
        # fit is insufficient if an interrupted/short write leaves an unreadable file.
        temporary = args.model_out.with_name(args.model_out.name + '.tmp')
        joblib.dump(artifact(model, config, data, fit_scope='all_labeled', seed=SEED), temporary, compress=3)
        restored = joblib.load(temporary)
        if restored['fit_scope'] != 'all_labeled' or restored['train_rows'] != len(data):
            raise RuntimeError('Serialized model failed training-scope verification')
        temporary.replace(args.model_out)
        write_json(args.model_out.with_suffix('.metadata.json'), {
            'config': config, 'seed': SEED, 'train_rows': len(data), 'source_sha256': data.attrs['source_sha256'],
            'fit_scope': 'all_labeled', 'fit_seconds': seconds,
            'warnings': [f'{item.category.__name__}: {item.message}' for item in caught]})
        print(f'Trained {len(data)} labeled rows; saved {args.model_out}', flush=True)
    else:
        train, validation = get_or_create_split(data, args.results / 'split.json', seed=SEED)
        record = evaluate(config, train, validation, args.results / 'verify', 'selected', save_model=True, resume=args.resume)
        print(f'Validation AUROC: {record["validation_auroc"]:.6f}; model at {args.results / "verify/selected.joblib"}')


if __name__ == '__main__':
    main()
