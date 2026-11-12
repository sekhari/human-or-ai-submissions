"""Evaluate the fixed 8000/2000 split, or explicitly refit the selected model on all 10000 rows."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

TEAM = Path(__file__).resolve().parent
sys.path.insert(0, str(TEAM / 'code'))
from models.dave_extratrees.model import DaveExtraTreesClassifier
from models.dave_extratrees.train import (
    FEATURE_SOURCE, PACKAGE, RECIPE_FILE, ROOT, _ids_valid,
    indices_sha256, load_recipe, sha256,
)
from models.text_generalization.style_search import FEATURE_NAMES


def run(args):
    destination = (args.output_dir or TEAM / 'artifacts' / args.mode).resolve()
    if destination.exists():
        raise ValueError(f'Output directory already exists: {destination}; choose a new --output-dir')
    recipe = load_recipe()
    path = args.data_dir.resolve() / 'train.csv'
    data_hash = sha256(path)
    if data_hash != recipe['data_sha256']['train.csv']:
        raise ValueError('train.csv checksum differs from the original competition file; do not rewrite or reorder it')
    train = pd.read_csv(path)
    if (list(train.columns) != ['id', 'text', 'label'] or len(train) != 10000
            or not _ids_valid(train) or train.text.isna().any()
            or not train.text.map(lambda value: isinstance(value, str)).all()
            or train.label.value_counts().to_dict() != {0: 5000, 1: 5000}):
        raise ValueError('Expected 10000 original labeled passages, unique integer IDs, and 5000 examples per class')

    source_files = [PACKAGE / '__init__.py', PACKAGE / 'model.py',
                    PACKAGE / 'train.py', FEATURE_SOURCE]
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in source_files}
    entrypoint_hash = sha256(__file__)
    recipe_hash = sha256(RECIPE_FILE)
    try:
        git_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=TEAM,
                                            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = None
    versions = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    scipy=scipy.__version__, sklearn=sklearn.__version__, joblib=joblib.__version__)
    metrics = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), mode=args.mode, git_head=git_head,
        recipe_sha256=recipe_hash, source_sha256=source_hashes, entrypoint_sha256=entrypoint_hash,
        data_sha256={'train.csv': data_hash}, model_params=recipe['model_params'],
        feature_count=145, versions=versions, external_or_test_labels_used=False,
        unlabeled_data_used=False, test_data_read=False,
    )
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'recipe.json').write_bytes(RECIPE_FILE.read_bytes())
    (destination / 'feature_names.json').write_text(json.dumps(list(FEATURE_NAMES), indent=2) + '\n')
    model = DaveExtraTreesClassifier(params=recipe['model_params'])
    with threadpool_limits(limits=2):
        if args.mode == 'validate':
            tr, va = train_test_split(np.arange(len(train)), test_size=.2,
                                      random_state=12, stratify=train.label)
            split_hashes = dict(development_indices_sha256=indices_sha256(tr),
                                validation_indices_sha256=indices_sha256(va))
            if any(value != recipe['split'][key] for key, value in split_hashes.items()):
                raise ValueError('Split row order differs from the frozen stratified seed-12 benchmark')
            started = perf_counter()
            model.fit(train.text.iloc[tr], train.label.iloc[tr])
            fit_seconds = perf_counter() - started
            validation_probability = model.predict_proba(train.text.iloc[va])[:, 1]
            validation_auroc = float(roc_auc_score(train.label.iloc[va], validation_probability))
            metrics.update(
                benchmark_training_rows=len(tr), benchmark_validation_rows=len(va),
                benchmark_fit_seconds=fit_seconds, split=recipe['split'],
                reproduced_split_sha256=split_hashes,
                train_auroc=float(roc_auc_score(train.label.iloc[tr], model.predict_proba(train.text.iloc[tr])[:, 1])),
                validation_auroc=validation_auroc,
                validation_auroc_delta_vs_reference=validation_auroc - recipe['reference']['validation_auroc'],
                validation_scope='Fixed 2000-row development holdout, reused during earlier experiments; not an untouched test.',
            )
            pd.DataFrame(dict(id=train.id.iloc[va].to_numpy(), label=train.label.iloc[va].to_numpy(),
                              probability=validation_probability)).to_csv(destination / 'validation_predictions.csv', index=False)
            # Saving IDs makes it possible to audit exactly which rows were used.
            train[['id']].iloc[tr].to_csv(destination / 'training_ids.csv', index=False)
            train[['id']].iloc[va].to_csv(destination / 'validation_ids.csv', index=False)
            if model.n_training_rows_ != 8000:
                raise ValueError('Validation model must be trained on exactly 8000 rows')
            print(f'Train AUROC: {metrics["train_auroc"]:.9f}\nValidation AUROC: {validation_auroc:.9f}', flush=True)
        else:
            started = perf_counter()
            model.fit(train.text, train.label)
            metrics.update(final_training_rows=len(train), final_fit_seconds=perf_counter() - started)
            if model.n_training_rows_ != 10000:
                raise ValueError('Final model must be trained on exactly 10000 rows')
            bundle = dict(model=model, mode='final', training_rows=10000,
                          recipe_sha256=recipe_hash, feature_source_sha256=recipe['feature_source_sha256'],
                          training_data_sha256=data_hash, versions=versions)
            model_path = destination / 'model.joblib'
            joblib.dump(bundle, model_path, compress=3)
            reloaded = joblib.load(model_path)['model']
            # Verify persistence on a small training probe; this is not a validation score.
            before = model.predict_proba(train.text.iloc[:64])
            after = reloaded.predict_proba(train.text.iloc[:64])
            np.testing.assert_allclose(before, after, rtol=0, atol=1e-12)
            metrics.update(model_sha256=sha256(model_path), reload_max_abs_error=float(np.max(np.abs(before - after))))
            print(f'Final model fitted on 10000 rows: {model_path}', flush=True)

    if (sha256(path) != data_hash or sha256(RECIPE_FILE) != recipe_hash
            or sha256(__file__) != entrypoint_hash
            or any(sha256(ROOT / name) != value for name, value in source_hashes.items())):
        raise ValueError('Input data, source code, or recipe changed during training')
    metrics['verification'] = 'PASS: canonical train data, frozen recipe, fit row count, and source checks'
    (destination / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n')
    print(f'Run records: {destination}', flush=True)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['validate', 'final'], default='validate',
                        help='validate: fit 8000/evaluate 2000 (default); final: fit all 10000 for Kaggle')
    parser.add_argument('--data-dir', type=Path, default=TEAM / 'data', help='Directory containing original train.csv')
    parser.add_argument('--output-dir', type=Path, help='New run directory; default: artifacts/<mode> beside this script')
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError) as error:
        parser.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    main()
