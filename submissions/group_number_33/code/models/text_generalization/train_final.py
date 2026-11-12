"""Evaluate the frozen recipe once, then refit all labeled rows for submission."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from .common import ROOT, OUT, PROTOCOL_FILE, sha256
from .final_model import WeightedTextEnsemble
from .style_search import FEATURE_NAMES


def validate_submission(submission, test, sample):
    if list(submission.columns) != ['id', 'label'] or len(submission) != 3000:
        raise ValueError('Submission must have 3000 rows and exactly id,label columns')
    if submission.id.isna().any() or not submission.id.is_unique:
        raise ValueError('Missing or duplicate submission IDs')
    if set(submission.id) != set(test.id):
        raise ValueError('Submission/test ID mismatch')
    np.testing.assert_array_equal(submission.id.to_numpy(), sample.id.to_numpy())
    values = submission.label.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError('Invalid predicted probabilities')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, default=OUT/'selection/selected.json')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    # Freeze/check the recipe before loading the original validation labels.
    selection_bytes = args.selection.read_bytes()
    selection = json.loads(selection_bytes)
    selection_sha = sha256(args.selection)
    if sha256(PROTOCOL_FILE) != selection['protocol_sha256']:
        raise ValueError('Protocol changed after model selection')
    layout_plan = OUT/'layout_stress/frozen_plan.json'
    if sha256(layout_plan) != selection['layout_plan_sha256']:
        raise ValueError('Layout protocol changed after model selection')
    protocol = json.loads(PROTOCOL_FILE.read_text())
    if sha256(ROOT/'data/train.csv') != protocol['train_sha256']:
        raise ValueError('Training source changed')
    model_template = WeightedTextEnsemble(**selection['model_config'])
    train = pd.read_csv(ROOT/'data/train.csv')
    test = pd.read_csv(ROOT/'data/test.csv')
    sample = pd.read_csv(ROOT/'data/sample_submission.csv')
    if len(train) != 10000 or len(test) != 3000 or len(sample) != 3000:
        raise ValueError('Unexpected dataset sizes')
    for frame in (train, test):
        if not frame.id.is_unique or frame.id.isna().any() or frame.text.isna().any():
            raise ValueError('Invalid IDs or missing texts')
    if set(train.label.unique()) != {0, 1}:
        raise ValueError('Expected human=0, AI=1 labels')
    development = np.asarray(protocol['development_indices'])
    outer = np.asarray(protocol['outer_indices'])
    np.testing.assert_array_equal(np.sort(np.r_[development, outer]), np.arange(len(train)))
    if len(development) != 8000 or len(outer) != 2000:
        raise ValueError('Original 8000/2000 split changed')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination = args.output_dir or OUT/'final'/stamp
    destination.mkdir(parents=True, exist_ok=False)
    (destination/'selected.json').write_bytes(selection_bytes)
    (destination/'protocol.json').write_bytes(PROTOCOL_FILE.read_bytes())
    versions = dict(python=platform.python_version(), numpy=np.__version__,
                    pandas=pd.__version__, scipy=scipy.__version__,
                    scikit_learn=sklearn.__version__, joblib=joblib.__version__)
    source_hashes = {str(p.relative_to(ROOT)): sha256(p)
                     for p in sorted((ROOT/'models/text_generalization').glob('*.py'))}
    metrics = dict(selection_sha256=selection_sha, versions=versions,
                   source_sha256=source_hashes,
                   git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   data_sha256={name: sha256(ROOT/'data'/name)
                                for name in ('train.csv', 'test.csv', 'sample_submission.csv')},
                   selection_protocol_scores=selection['selected'],
                   original_validation_scope='Reused development benchmark; not an untouched test. Frozen recipe evaluated after selection.',
                   final_model_training_rows=10000, unlabeled_data_used=False,
                   kaggle_test_labels_used=False, feature_count=len(FEATURE_NAMES))
    with threadpool_limits(limits=2):
        start = perf_counter()
        benchmark = clone(model_template).fit(train.text.iloc[development], train.label.iloc[development])
        metrics['benchmark_fit_seconds'] = perf_counter()-start
        benchmark_probability = benchmark.predict_proba(train.text.iloc[outer])[:, 1]
        metrics['original_validation_auroc'] = float(roc_auc_score(train.label.iloc[outer], benchmark_probability))
        metrics['benchmark_training_rows'] = benchmark.n_training_rows_
        metrics['benchmark_validation_rows'] = len(outer)
        pd.DataFrame(dict(id=train.id.iloc[outer].to_numpy(),
                          label=train.label.iloc[outer].to_numpy(),
                          probability=benchmark_probability)).to_csv(destination/'validation_predictions.csv', index=False)
        print(f"Frozen-recipe original 2000 validation AUROC: {metrics['original_validation_auroc']:.9f}", flush=True)
        # No branching on the benchmark score: final recipe is unchanged.
        start = perf_counter()
        final_model = clone(model_template).fit(train.text, train.label)
        metrics['final_fit_seconds'] = perf_counter()-start
        probabilities = final_model.predict_proba(test.text)[:, 1]
        by_id = pd.Series(probabilities, index=test.id)
        submission = pd.DataFrame(dict(id=sample.id, label=sample.id.map(by_id)))
        validate_submission(submission, test, sample)
        csv_path = destination/'submission_style_extratrees_full10000.csv'
        submission.to_csv(csv_path, index=False)
        readback = pd.read_csv(csv_path)
        validate_submission(readback, test, sample)
        np.testing.assert_allclose(readback.label, submission.label, rtol=0, atol=1e-15)
        model_path = destination/'model.joblib'
        joblib.dump(final_model, model_path, compress=3)
        reloaded = joblib.load(model_path)
        reloaded_probabilities = reloaded.predict_proba(test.text)[:, 1]
        np.testing.assert_allclose(reloaded_probabilities, probabilities, rtol=0, atol=1e-12)
        batched = np.concatenate([reloaded.predict_proba(test.text.iloc[i:i+257])[:, 1]
                                  for i in range(0, len(test), 257)])
        np.testing.assert_allclose(batched, probabilities, rtol=0, atol=1e-12)
        assert final_model.n_training_rows_ == reloaded.n_training_rows_ == 10000
        if sha256(args.selection) != selection_sha:
            raise ValueError('Frozen selection changed during final training')
        metrics.update(submission_rows=len(submission), submission_sha256=sha256(csv_path),
                       model_sha256=sha256(model_path),
                       probability_min=float(probabilities.min()), probability_max=float(probabilities.max()),
                       probability_mean=float(probabilities.mean()),
                       reload_max_abs_error=float(np.max(np.abs(reloaded_probabilities-probabilities))),
                       batch_max_abs_error=float(np.max(np.abs(batched-probabilities))),
                       verification='PASS: schema, IDs, probabilities, reload, batch invariance, fit row counts, frozen selection')
    (destination/'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output_dir=str(destination), submission=str(csv_path),
                          validation_auroc=metrics['original_validation_auroc'],
                          final_training_rows=10000, verification=metrics['verification']), indent=2), flush=True)


if __name__ == '__main__':
    main()
