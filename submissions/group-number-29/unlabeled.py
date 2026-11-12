"""Q3: confidence-based pseudo-labeling of unlabeled.csv on the shared validation split."""

import argparse
import hashlib
import json
import platform
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from improvement import build_candidate
from style_experiment import build_model as build_style_model
from train import SEED, load_data, split_data

# Pseudo-label the most confident fraction of unlabeled rows, split evenly by class.
FRACTIONS = (0.1, 0.25, 0.5, 1.0)


# Final model (word 1-2 TF-IDF + 20 style features) and the earlier Q5 character model.
MODELS = {
    'word_and_style': ('word 1-2 TF-IDF + 20 standardized style features, logistic regression C=10',
                       lambda: build_style_model('word_and_style')),
    'character': ('lowercase char TF-IDF 2-5, logistic regression C=30',
                  lambda: build_candidate('character').set_params(classifier__C=30.0,
                                                                  features__ngram_range=(2, 5))),
}


def build_selected():
    # The earlier Q5 character model, kept so existing callers still work.
    return MODELS['character'][1]()


def normalize(texts):
    return texts.str.normalize('NFKC').str.casefold().str.split().str.join(' ')


def fit_and_score(build, training, validation):
    model = build()
    start = time.perf_counter()
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(training['text'], training['label'])
    seconds = time.perf_counter() - start
    ai_column = list(model.classes_).index(1)
    val_probs = model.predict_proba(validation['text'])[:, ai_column]
    return model, val_probs, seconds


def pseudo_label(unlabeled, probs, fraction):
    # Rank by teacher probability; take the k most human-like and k most AI-like rows.
    per_class = int(len(unlabeled) * fraction) // 2
    order = np.argsort(probs, kind='stable')
    human, ai = order[:per_class], order[::-1][:per_class]
    rows = pd.concat([unlabeled.iloc[human].assign(label=0),
                      unlabeled.iloc[ai].assign(label=1)])
    cutoffs = {'human_max_prob': float(probs[human].max()), 'ai_min_prob': float(probs[ai].min())}
    return rows, cutoffs


def noise_estimate(validation, val_probs, cutoffs):
    # Teacher error on validation rows at least as confident as the chosen pseudo-labels.
    labels = validation['label'].to_numpy()
    human = val_probs <= cutoffs['human_max_prob']
    ai = val_probs >= cutoffs['ai_min_prob']
    confident = human | ai
    errors = (human & (labels == 1)) | (ai & (labels == 0))
    return {'val_rows_at_confidence': int(confident.sum()),
            'val_error_rate_at_confidence': float(errors.sum() / max(confident.sum(), 1))}


def bootstrap_difference(labels, baseline, other, draws=2000):
    # Paired bootstrap over validation rows: 95% interval for AUROC(other) - AUROC(baseline).
    rng = np.random.default_rng(SEED)
    differences = []
    for _ in range(draws):
        rows = rng.integers(0, len(labels), len(labels))
        if labels[rows].min() != labels[rows].max():
            differences.append(roc_auc_score(labels[rows], other[rows])
                               - roc_auc_score(labels[rows], baseline[rows]))
    return [float(value) for value in np.percentile(differences, [2.5, 97.5])]


def format_table(results):
    lines = ['| Method | Pseudo-labeled rows | Teacher cutoffs (human <= / AI >=) '
             '| Est. pseudo-label error | Validation AUROC | Change vs supervised (95% CI) '
             '| Fit seconds |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in results:
        cutoffs = (f"{row['human_max_prob']:.3f} / {row['ai_min_prob']:.3f}"
                   if row['pseudo_rows'] else '-')
        noise = (f"{row['val_error_rate_at_confidence']:.2%} "
                 f"({row['val_rows_at_confidence']:,} val rows)" if row['pseudo_rows'] else '-')
        change = (f"{row['auroc_change']:+.6f} [{row['change_ci_low']:+.5f}, "
                  f"{row['change_ci_high']:+.5f}]" if row['pseudo_rows'] else '-')
        lines.append(f"| {row['method']} | {row['pseudo_rows']:,} | {cutoffs} | {noise} "
                     f"| {row['validation_auroc']:.6f} | {change} | {row['fit_seconds']:.1f} |")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--unlabeled', type=Path, required=True)
    parser.add_argument('--split', type=Path, default=Path(__file__).with_name('validation_ids.json'))
    parser.add_argument('--model', choices=MODELS, default='word_and_style')
    parser.add_argument('--out-dir', type=Path, default=Path('outputs/q3'))
    args = parser.parse_args()
    description, build = MODELS[args.model]
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error('Output directory is not empty; choose a new --out-dir')
    data = load_data(args.train, labeled=True)
    fingerprint = hashlib.sha256(args.train.read_bytes()).hexdigest()
    training, validation = split_data(data, args.split, fingerprint)
    unlabeled = load_data(args.unlabeled).sort_values('id').reset_index(drop=True)
    # Validation passages must never enter fitting, including through unlabeled.csv.
    if set(unlabeled['id']) & set(data['id']):
        raise ValueError('Unlabeled IDs overlap labeled IDs')
    if normalize(unlabeled['text']).isin(set(normalize(validation['text']))).any():
        raise ValueError('An unlabeled passage duplicates a validation passage')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    metadata = {
        'code_commit': commit, 'working_tree_dirty': dirty, 'seed': SEED,
        'train_sha256': fingerprint,
        'unlabeled_sha256': hashlib.sha256(args.unlabeled.read_bytes()).hexdigest(),
        'split_sha256': hashlib.sha256(args.split.read_bytes()).hexdigest(),
        'python': platform.python_version(), 'sklearn': sklearn.__version__,
        'train_rows': len(training), 'validation_rows': len(validation),
        'unlabeled_rows': len(unlabeled), 'native_threads': 1,
        'model': description,
        'fractions': list(FRACTIONS),
        'protocol': 'teacher fit on 8,000 training rows; one round; balanced top-k by teacher '
                    'probability; student refits features and classifier on train + pseudo rows',
    }
    (args.out_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')

    print('Fitting supervised teacher...', flush=True)
    teacher, val_probs, seconds = fit_and_score(build, training, validation)
    teacher_probs = teacher.predict_proba(unlabeled['text'])[:, list(teacher.classes_).index(1)]
    pd.DataFrame({'id': unlabeled['id'], 'probability_ai': teacher_probs}).to_csv(
        args.out_dir / 'unlabeled_teacher_probabilities.csv', index=False)
    results = [{'method': 'Supervised only', 'fraction': 0.0, 'pseudo_rows': 0,
                'validation_auroc': roc_auc_score(validation['label'], val_probs),
                'fit_seconds': seconds,
                'unlabeled_predicted_ai_share': float((teacher_probs >= 0.5).mean())}]
    print(json.dumps(results[-1]), flush=True)
    predictions = validation[['id', 'label']].assign(probability_ai=val_probs)
    predictions.to_csv(args.out_dir / 'supervised_validation_predictions.csv', index=False)

    for fraction in FRACTIONS:
        print(f'Fitting student with top {fraction:.0%} pseudo-labels...', flush=True)
        rows, cutoffs = pseudo_label(unlabeled, teacher_probs, fraction)
        _, student_probs, seconds = fit_and_score(build, pd.concat([training, rows]), validation)
        labels = validation['label'].to_numpy()
        ci_low, ci_high = bootstrap_difference(labels, val_probs, student_probs)
        results.append({'method': f'Pseudo-labels, top {fraction:.0%}', 'fraction': fraction,
                        'pseudo_rows': len(rows),
                        'validation_auroc': roc_auc_score(validation['label'], student_probs),
                        'auroc_change': roc_auc_score(labels, student_probs) - results[0]['validation_auroc'],
                        'change_ci_low': ci_low, 'change_ci_high': ci_high,
                        'predictions_flipped_at_0.5': int(((val_probs >= 0.5) != (student_probs >= 0.5)).sum()),
                        'fit_seconds': seconds, **cutoffs,
                        **noise_estimate(validation, val_probs, cutoffs)})
        print(json.dumps(results[-1]), flush=True)
        predictions = validation[['id', 'label']].assign(probability_ai=student_probs)
        predictions.to_csv(args.out_dir / f'pseudo_{int(fraction * 100)}_validation_predictions.csv',
                           index=False)
        pd.DataFrame(results).to_csv(args.out_dir / 'results.csv', index=False)

    report = f"""# Q3: learning from unlabeled data

## Protocol

- Code revision: `{commit}` (dirty: {dirty}); seed: `{SEED}`; Python: `{platform.python_version()}`; scikit-learn `{sklearn.__version__}`.
- Model: {description} (`--model {args.model}`), unchanged across rows.
- Teacher: fit on the 8,000 training rows only; it scores all {len(unlabeled):,} unlabeled passages.
- Pseudo-labels: one round. For each fraction, the k most human-like rows get label 0 and the k most AI-like get label 1 (k = fraction x {len(unlabeled):,} / 2), keeping the added rows balanced.
- Student: a fresh pipeline (vocabulary, IDF and classifier) fit on training + pseudo-labeled rows, scored on the same fixed 2,000-row validation set.
- Leakage guards: validation rows are never fit or pseudo-labeled; the script checks that unlabeled IDs and normalized texts do not overlap the labeled/validation data. `test.csv` is not used.
- Estimated pseudo-label error: the teacher's error rate on validation rows that are at least as confident as each fraction's cutoffs. The true unlabeled error is unknown because those labels are hidden.
- The teacher labels {results[0]['unlabeled_predicted_ai_share']:.1%} of unlabeled passages as AI at threshold 0.5.

## Results

{format_table(results)}

Fit seconds use one native thread and include vectorization. The 95% interval is a paired bootstrap (2,000 resamples of the validation rows, seed {SEED}); an interval containing 0 means the change is within validation noise. Choosing a fraction by these scores uses the validation set.

## Reproduction

```sh
.venv/bin/python unlabeled.py --train data/train.csv --unlabeled data/unlabeled.csv --model {args.model} --out-dir {args.out_dir.as_posix()}
```

Local outputs: `{args.out_dir.as_posix()}/` (results, teacher probabilities, validation predictions; not committed).
"""
    Path(__file__).with_name('q3_results.md').write_text(report)
    print(f'Results: q3_results.md; outputs: {args.out_dir}')


if __name__ == '__main__':
    main()
