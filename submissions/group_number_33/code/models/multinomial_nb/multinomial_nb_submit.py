"""Generate a verified submission without relying on local experiment outputs.

From the repository root:
    python3 -m models.multinomial_nb.multinomial_nb_submit
    python3 -m models.multinomial_nb.multinomial_nb_submit --model single
Both recipes preserve the original 8,000/2,000 split and shared preprocessing.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
from time import perf_counter

import numpy as np
import pandas as pd
import scipy
import sklearn
import threadpoolctl
from threadpoolctl import threadpool_limits

from .multinomial_nb import ROOT, MODEL_DIR, OUTPUT_DIR, REFERENCE_DIR, SEED
from .multinomial_nb import calculate_metrics, fit_weighted_nb, sha256
from .multinomial_nb_blend import ALT_CONFIG, NBScoreBlend
from preprocess import preprocess_data


REFERENCE = REFERENCE_DIR / "baseline.json"
SINGLE_CONFIG = dict(alpha=3., sample_power=-1., prior_power=-16., ai_weight=1., weight_cap=2.)


def probability_fingerprint(probabilities):
    """SHA256 of ordered probabilities rounded to 10 decimals, little-endian f8."""
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Probability fingerprint requires a finite one-dimensional array")
    return hashlib.sha256(np.round(values, 10).astype('<f8').tobytes()).hexdigest()


def csr_fingerprint(matrix):
    """Canonical CSR shape<i8, values<f8, indices<i4, indptr<i4 SHA256."""
    digest = hashlib.sha256()
    for values, dtype in [(matrix.shape, '<i8'), (matrix.data, '<f8'),
                          (matrix.indices, '<i4'), (matrix.indptr, '<i4')]:
        digest.update(np.asarray(values, dtype=dtype).tobytes())
    return digest.hexdigest()


def make_submission(test, template, probabilities):
    """Match predictions to template IDs rather than relying on row position."""
    if "id" not in test or list(template.columns) != ["id", "label"]:
        raise ValueError("Require test IDs and template columns id,label")
    for frame in (test, template):
        if frame.id.isna().any() or not frame.id.is_unique:
            raise ValueError("Submission and test IDs must be unique and nonmissing")
    if len(test) != len(template) or set(test.id) != set(template.id):
        raise ValueError("Template must contain every test ID exactly once")
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(test),) or not np.isfinite(probabilities).all():
        raise ValueError("Require one finite prediction per test row")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Predictions must be probabilities between 0 and 1")
    by_id = pd.Series(probabilities, index=test.id)
    return pd.DataFrame({"id": template.id.to_numpy(),
                         "label": by_id.loc[template.id].to_numpy()})


def verify_probability_reference(probabilities, expected, name):
    if probability_fingerprint(probabilities) != expected:
        raise ValueError(
            f"{name} prediction fingerprint differs from the frozen reference. "
            "Check the pinned dependency versions and model implementation. "
            "This strict rounded fingerprint can also differ for tiny floating-point "
            "changes at rounding boundaries; inspect numerical parity before accepting a new baseline."
        )


def run_submission(model_name="ensemble"):
    if model_name not in {"ensemble", "single"}:
        raise ValueError("Choose ensemble or single")
    started = datetime.now(timezone.utc)
    reference = json.loads(REFERENCE.read_text())
    recipe = reference["models"][model_name]
    for name, digest in reference["input_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Shared input differs from the frozen reference: {name}")
    if model_name == "ensemble":
        if NBScoreBlend().get_params() != recipe["model_parameters"] or ALT_CONFIG != recipe["alternate_parameters"]:
            raise ValueError("Ensemble settings differ from the frozen reference")
    elif SINGLE_CONFIG != recipe["model_parameters"]:
        raise ValueError("Single NB settings differ from the frozen reference")

    raw = pd.read_csv(ROOT / "data/train.csv")
    test = pd.read_csv(ROOT / "data/test.csv")
    template = pd.read_csv(ROOT / "data/sample_submission.csv")
    if len(raw) != 10000 or len(test) != 3000 or len(template) != 3000:
        raise ValueError("Expected 10,000 labeled examples and 3,000 test examples")
    make_submission(test, template, np.full(len(test), .5))
    start = perf_counter()
    X, y, V, vy, T = preprocess_data(n=8000, random_state=SEED)
    preprocessing_seconds = perf_counter() - start
    split = pd.concat([
        pd.DataFrame({"id": raw.loc[y.index, "id"].to_numpy(), "split": "train"}),
        pd.DataFrame({"id": raw.loc[vy.index, "id"].to_numpy(), "split": "validation"}),
    ], ignore_index=True)
    split_csv = split.to_csv(index=False, lineterminator='\n')
    if hashlib.sha256(split_csv.encode()).hexdigest() != reference["split_csv_sha256"]:
        raise ValueError("Training/validation IDs or order differ from the reference")
    for name, matrix in [('train', X), ('validation', V)]:
        if list(matrix.shape) != reference[f'{name}_shape'] or csr_fingerprint(matrix) != reference[f'{name}_tfidf_sha256']:
            raise ValueError(f"{name} TF-IDF differs from the reference")
    if list(T.shape) != reference['test_shape']:
        raise ValueError("Unexpected test feature shape")
    for name, labels in [('train', y), ('validation', vy)]:
        digest = hashlib.sha256(np.asarray(labels, dtype='<i8').tobytes()).hexdigest()
        if digest != reference[f'{name}_labels_sha256']:
            raise ValueError(f"{name} labels/order differ from the reference")

    print(f"Fitting frozen {model_name} NB on the original 8,000 training rows...", flush=True)
    start = perf_counter()
    model = (NBScoreBlend().fit(X, y) if model_name == "ensemble"
             else fit_weighted_nb(X, y, **SINGLE_CONFIG))
    fit_seconds = perf_counter() - start
    np.testing.assert_array_equal(model.classes_, [0, 1])
    validation_probabilities = model.predict_proba(V)[:, 1]
    verify_probability_reference(validation_probabilities, recipe['validation_probability_sha256'], 'Validation')
    validation_metrics = calculate_metrics(vy, validation_probabilities)
    for metric, expected in recipe['validation'].items():
        np.testing.assert_allclose(validation_metrics[metric], expected, rtol=0, atol=1e-12)
    print(f"Validation reproduced: AUROC={validation_metrics['auroc']:.6f}", flush=True)
    start = perf_counter()
    probabilities = model.predict_proba(T)[:, 1]
    predict_seconds = perf_counter() - start
    verify_probability_reference(probabilities, recipe['test_probability_sha256'], 'Test')
    submission = make_submission(test, template, probabilities)
    destination = OUTPUT_DIR / 'kaggle_submissions' / (started.strftime('%Y%m%dT%H%M%S_%fZ') + '_' + model_name)
    destination.mkdir(parents=True, exist_ok=False)
    csv_path = destination / 'submission.csv'
    submission.to_csv(csv_path, index=False)
    readback = pd.read_csv(csv_path)
    pd.testing.assert_frame_equal(readback, submission, check_exact=False, rtol=0, atol=1e-15)
    make_submission(readback, template, readback.label.to_numpy())
    (destination / 'split_ids.csv').write_text(split_csv)
    pd.DataFrame(dict(id=raw.loc[vy.index, 'id'].to_numpy(), true_label=vy.to_numpy(),
                      probability_ai=validation_probabilities)).to_csv(destination / 'validation_predictions.csv', index=False)
    source_names = list(reference['input_sha256']) + [str(p.relative_to(ROOT)) for p in sorted(MODEL_DIR.glob('*.py'))]
    manifest = dict(
        created_at_utc=started.isoformat(), model_name=model_name,
        reference=str(REFERENCE.relative_to(ROOT)), reference_sha256=sha256(REFERENCE),
        reference_run=recipe['historical_run'],
        mode='Original 8000/2000 split; no validation-label fitting, no full-data refit',
        model_parameters=recipe['model_parameters'], alternate_parameters=recipe.get('alternate_parameters'),
        training_shape=list(X.shape), validation_shape=list(V.shape), test_shape=list(T.shape),
        validation=validation_metrics, exact_reference_split_ids_and_order=True,
        probability_fingerprint_algorithm=reference['probability_fingerprint_algorithm'],
        validation_probability_fingerprint_matched=True, test_probability_fingerprint_matched=True,
        test_labels_used=False, unlabeled_data_used=False, uploaded_to_kaggle=False,
        submission_rows=len(readback), submission_columns=list(readback.columns), ids_in_sample_submission_order=True,
        probability_min=float(probabilities.min()), probability_max=float(probabilities.max()),
        timing_seconds=dict(preprocessing=preprocessing_seconds, model_fit=fit_seconds, test_prediction=predict_seconds),
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                      scipy=scipy.__version__, scikit_learn=sklearn.__version__, threadpoolctl=threadpoolctl.__version__),
        source_sha256={name: sha256(ROOT / name) for name in source_names}, submission_sha256=sha256(csv_path),
        notes=['Development validation was reused for model selection; it is not a Kaggle score.',
               'Only upload submission.csv. All generated artifacts stay local.',
               'Rounded probability fingerprints are strict environment checks, not an atol comparison.'])
    (destination / 'metadata.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print(f"Verified 3000 rows; submission file: {csv_path}")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['ensemble', 'single'], default='ensemble')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run_submission(args.model)


if __name__ == '__main__':
    main()
