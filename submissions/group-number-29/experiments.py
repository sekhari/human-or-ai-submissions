"""Compare text classifiers and measure the baseline learning curve."""

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from train import SEED, build_model, load_data, split_data


def build_count_nb(alpha):
    vectorizer = CountVectorizer(
        lowercase=True,
        ngram_range=(1, 1),
        min_df=2,
        max_df=1.0,
        max_features=100000,
        token_pattern=r'(?u)\b\w\w+\b',
    )
    return Pipeline([
        ('counts', vectorizer),
        ('classifier', MultinomialNB(alpha=alpha, fit_prior=True)),
    ])


def build_linear_svm(c):
    vectorizer = build_model().named_steps['tfidf']
    return Pipeline([
        ('tfidf', vectorizer),
        ('classifier', LinearSVC(C=c, tol=0.0001, max_iter=5000,
                                random_state=SEED, dual='auto')),
    ])


def positive_scores(model, texts):
    if hasattr(model, 'predict_proba'):
        positive_column = list(model.classes_).index(1)
        return model.predict_proba(texts)[:, positive_column]
    return model.decision_function(texts)


def fit_and_score(model, training, validation):
    start = time.perf_counter()
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(training['text'], training['label'])
    seconds = time.perf_counter() - start
    train_scores = positive_scores(model, training['text'])
    validation_scores = positive_scores(model, validation['text'])
    return {
        'train_auroc': roc_auc_score(training['label'], train_scores),
        'validation_auroc': roc_auc_score(validation['label'], validation_scores),
        'fit_seconds': seconds,
    }


def _logistic_model(c):
    model = build_model()
    model.set_params(classifier__C=c)
    return model


def candidate_specs():
    return [
        ('Logistic regression / word TF-IDF', 'logistic_regression', {'C': c},
         lambda c=c: _logistic_model(c))
        for c in (0.1, 1.0, 10.0)
    ] + [
        ('Multinomial Naive Bayes / counts', 'count_naive_bayes', {'alpha': alpha},
         lambda alpha=alpha: build_count_nb(alpha))
        for alpha in (0.01, 0.1, 1.0)
    ] + [
        ('Linear SVM / word TF-IDF', 'linear_svm', {'C': c},
         lambda c=c: build_linear_svm(c))
        for c in (0.1, 1.0, 10.0)
    ]


def compare_models(training, validation, repeats):
    results = []
    for model_name, family, params, build in candidate_specs():
        measurements = [fit_and_score(build(), training, validation)
                        for _ in range(repeats)]
        results.append({
            'model': model_name,
            'family': family,
            'parameters': json.dumps(params, sort_keys=True),
            'score_type': 'decision margin' if family == 'linear_svm' else 'probability',
            'train_auroc': sum(item['train_auroc'] for item in measurements) / repeats,
            'validation_auroc': sum(item['validation_auroc'] for item in measurements) / repeats,
            'fit_seconds_min': min(item['fit_seconds'] for item in measurements),
            'fit_seconds_max': max(item['fit_seconds'] for item in measurements),
            'repeats': repeats,
            'seed': SEED,
        })
    return results


def balanced_nested_subset(training, size):
    half_size = size // 2
    class_rows = [
        training[training['label'] == label].sample(
            frac=1, random_state=SEED + label).iloc[:half_size]
        for label in (0, 1)
    ]
    subset = pd.concat(class_rows).sort_values('id').reset_index(drop=True)
    if len(subset) != size or subset['label'].value_counts().to_dict() != {
        0: half_size, 1: half_size
    }:
        raise ValueError(f'Could not create a balanced {size}-row training subset')
    return subset


def learning_curve(training, validation):
    results = []
    for size in (500, 1000, 2000, 5000, 8000):
        subset = balanced_nested_subset(training, size)
        model = build_model()
        result = fit_and_score(model, subset, validation)
        results.append({
            'train_rows': size,
            'train_auroc': result['train_auroc'],
            'validation_auroc': result['validation_auroc'],
            'fit_seconds': result['fit_seconds'],
            'seed': SEED,
            'vocabulary_size': len(model.named_steps['tfidf'].vocabulary_),
        })
    return results


def write_svg(path, results):
    width, height = 760, 470
    left, right, top, bottom = 82, 28, 48, 68
    plot_width, plot_height = width - left - right, height - top - bottom
    x_min, x_max = math.log2(500), math.log2(8000)
    y_min, y_max = 0.5, 1.0

    def x_position(size):
        return left + (math.log2(size) - x_min) / (x_max - x_min) * plot_width

    def y_position(score):
        return top + (y_max - score) / (y_max - y_min) * plot_height

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="82" y="27" font-family="sans-serif" font-size="18" fill="#18212b">Baseline learning curve</text>',
    ]
    for score in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        y = y_position(score)
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#d8dee5"/>')
        elements.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#52606d">{score:.1f}</text>')
    for size in (500, 1000, 2000, 5000, 8000):
        x = x_position(size)
        elements.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" stroke="#eef1f4"/>')
        elements.append(f'<text x="{x:.1f}" y="{height-bottom+24}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#52606d">{size:,}</text>')
    elements.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#52606d"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#52606d"/>',
        f'<text x="{left + plot_width/2:.1f}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#18212b">Training rows (log scale)</text>',
        f'<text x="22" y="{top + plot_height/2:.1f}" transform="rotate(-90 22 {top + plot_height/2:.1f})" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#18212b">AUROC</text>',
    ])
    for key, color, label in (
        ('train_auroc', '#147d75', 'Training'),
        ('validation_auroc', '#c24f35', 'Validation'),
    ):
        points = ' '.join(
            f'{x_position(row["train_rows"]):.1f},{y_position(row[key]):.1f}'
            for row in results
        )
        elements.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for row in results:
            x, y = x_position(row['train_rows']), y_position(row[key])
            elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}"/>')
        legend_y = top + (13 if key == 'train_auroc' else 35)
        elements.append(f'<line x1="{width-185}" y1="{legend_y}" x2="{width-160}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        elements.append(f'<text x="{width-153}" y="{legend_y+4}" font-family="sans-serif" font-size="12" fill="#18212b">{label}</text>')
    elements.append('</svg>')
    path.write_text('\n'.join(elements) + '\n')


def git_revision():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def format_model_table(results, repeats):
    lines = [
        '| Pipeline | Parameters | Train AUROC | Validation AUROC | Fit seconds (min-max, '
        f'{repeats} runs) |',
        '|---|---:|---:|---:|---:|',
    ]
    for row in results:
        lines.append(
            f"| {row['model']} | `{row['parameters']}` | {row['train_auroc']:.6f} "
            f"| {row['validation_auroc']:.6f} | {row['fit_seconds_min']:.3f}-"
            f"{row['fit_seconds_max']:.3f} |"
        )
    return '\n'.join(lines)


def format_curve_table(results):
    lines = [
        '| Training rows | Train AUROC | Validation AUROC | Fit seconds | Vocabulary size |',
        '|---:|---:|---:|---:|---:|',
    ]
    for row in results:
        lines.append(
            f"| {row['train_rows']:,} | {row['train_auroc']:.6f} "
            f"| {row['validation_auroc']:.6f} | {row['fit_seconds']:.3f} "
            f"| {row['vocabulary_size']:,} |"
        )
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--split', type=Path, default=Path(__file__).with_name('validation_ids.json'))
    parser.add_argument('--out-dir', type=Path, default=Path('outputs/q1_q2'))
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be at least 1')

    data = load_data(args.train, labeled=True)
    train_hash = hashlib.sha256(args.train.read_bytes()).hexdigest()
    training, validation = split_data(data, args.split, train_hash)
    comparisons = compare_models(training, validation, args.repeats)
    curve = learning_curve(training, validation)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparisons).to_csv(args.out_dir / 'model_comparison.csv', index=False)
    pd.DataFrame(curve).to_csv(args.out_dir / 'learning_curve.csv', index=False)
    revision = git_revision()
    report_path = Path(__file__).with_name('q1_q2_results.md')
    chart_path = Path(__file__).with_name('q1_q2_learning_curve.svg')
    write_svg(chart_path, curve)
    report = f"""# Q1/Q2: model comparison and training size

## Protocol

- Code revision: `{revision}`; seed: `{SEED}`; Python: `{platform.python_version()}`.
- Labeled data: 10,000 rows, 5,000 per class; SHA-256: `{train_hash}`.
- Fixed validation: 2,000 rows, 1,000 per class; IDs from `validation_ids.json` (unchanged).
- Every pipeline fits its own vocabulary and IDF/count features on that run's training rows only. Model comparison uses the same fixed holdout; SVM AUROC uses decision margins, not probabilities.
- Candidate runtimes are fit-only seconds, repeated {args.repeats} times with one BLAS/OpenMP thread. Learning-curve runtimes are one fit per size.

## Q1: complete pipeline comparison

The feature representation varies by model family, so this is a comparison of complete pipelines. Hyperparameters were selected within each family by validation AUROC; no test rows were used.

{format_model_table(comparisons, args.repeats)}

Best candidate per family by validation AUROC:

""" + '\n'.join(
        f"- {family}: `{row['parameters']}`; train AUROC "
        f"{row['train_auroc']:.6f}, validation AUROC {row['validation_auroc']:.6f}."
        for family in dict.fromkeys(row['family'] for row in comparisons)
        for row in [max((item for item in comparisons if item['family'] == family),
                        key=lambda item: item['validation_auroc'])]
    ) + f"""

## Q2: baseline learning curve

The baseline settings are unchanged from `train.py`; the fixed validation set is reused. Each size uses a deterministic, class-balanced subset nested within larger sizes. The text pipeline is newly fitted for every size.

{format_curve_table(curve)}

![Training and validation AUROC by training-set size](q1_q2_learning_curve.svg)

The x-axis is logarithmic. The chart and detailed CSVs are generated by `experiments.py`.

## Reproduction

From this directory, activate the Python 3.11 environment and run:

```sh
.venv/bin/python experiments.py --train /path/to/train.csv
```

Candidate results: `{(args.out_dir / 'model_comparison.csv').as_posix()}`; learning-curve values: `{(args.out_dir / 'learning_curve.csv').as_posix()}`.

AI assisted with the experiment implementation and generated report. Automated checks and scores are not represented as independent human verification; team review remains necessary. No final Kaggle model is selected here.
"""
    report_path.write_text(report)
    print(f'Comparison candidates: {len(comparisons)}; repeats each: {args.repeats}')
    print(f'Best validation AUROC: {max(row["validation_auroc"] for row in comparisons):.6f}')
    print(f'Results: {report_path}; chart: {chart_path}; CSVs: {args.out_dir}')


if __name__ == '__main__':
    main()