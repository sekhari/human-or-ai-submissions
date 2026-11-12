"""Q4: generalization and error analysis of the selected model on the shared validation split."""

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
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from train import SEED, load_data, split_data
from unlabeled import MODELS

# Selected C for each model; the other C values only show how the train/validation gap responds.
SELECTED_C = {'word_and_style': 10.0, 'character': 30.0}
GAP_C_VALUES = (1.0, 3.0, 10.0, 30.0)


def fit_and_predict(build, c, training, validation):
    model = build().set_params(classifier__C=c)
    start = time.perf_counter()
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(training['text'], training['label'])
    seconds = time.perf_counter() - start
    ai_column = list(model.classes_).index(1)
    # Same passages with every whitespace run collapsed to one space, to test formatting reliance.
    flat_text = validation['text'].str.replace(r'\s+', ' ', regex=True)
    return (model['classifier'].coef_.shape[1], seconds,
            model.predict_proba(training['text'])[:, ai_column],
            model.predict_proba(validation['text'])[:, ai_column],
            model.predict_proba(flat_text)[:, ai_column])


def summarize(labels, probs):
    return {'auroc': roc_auc_score(labels, probs),
            'accuracy_at_0.5': accuracy_score(labels, probs >= 0.5),
            'log_loss': log_loss(labels, probs, labels=[0, 1]),
            'share_confident': float(((probs <= 0.01) | (probs >= 0.99)).mean())}


def gap_table(rows):
    lines = ['| C | Train AUROC | Validation AUROC | Gap | Train / val accuracy at 0.5 '
             '| Train / val log loss | Confident (<=0.01 or >=0.99) train / val | Fit seconds |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        t, v = row['train'], row['validation']
        lines.append(f"| {row['C']:g} | {t['auroc']:.6f} | {v['auroc']:.6f} "
                     f"| {t['auroc'] - v['auroc']:.6f} "
                     f"| {t['accuracy_at_0.5']:.4f} / {v['accuracy_at_0.5']:.4f} "
                     f"| {t['log_loss']:.4f} / {v['log_loss']:.4f} "
                     f"| {t['share_confident']:.1%} / {v['share_confident']:.1%} "
                     f"| {row['fit_seconds']:.1f} |")
    return '\n'.join(lines)


def length_bins(validation, probs):
    # Equal-count word-count terciles of the validation set.
    words = validation['text'].str.split().str.len()
    bins = pd.qcut(words, 3, labels=['Short', 'Medium', 'Long'])
    labels = validation['label'].to_numpy()
    rng = np.random.default_rng(SEED)
    rows = []
    for name in bins.cat.categories:
        mask = (bins == name).to_numpy()
        y, p = labels[mask], probs[mask]
        draws = []
        for _ in range(2000):
            sample = rng.integers(0, len(y), len(y))
            if y[sample].min() != y[sample].max():
                draws.append(roc_auc_score(y[sample], p[sample]))
        rows.append({'bin': name, 'min_words': int(words[mask].min()),
                     'max_words': int(words[mask].max()), 'count': int(mask.sum()),
                     'human': int((y == 0).sum()), 'ai': int((y == 1).sum()),
                     'auroc': roc_auc_score(y, p),
                     'ci_low': float(np.percentile(draws, 2.5)),
                     'ci_high': float(np.percentile(draws, 97.5)),
                     'accuracy_at_0.5': accuracy_score(y, p >= 0.5),
                     'median_words_human': float(words[mask][y == 0].median()),
                     'median_words_ai': float(words[mask][y == 1].median())})
    return rows


def length_table(rows):
    lines = ['| Bin (words) | Count | Human / AI | Validation AUROC (95% CI) | Accuracy at 0.5 |',
             '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['bin']} ({row['min_words']}-{row['max_words']}) | {row['count']:,} "
                     f"| {row['human']} / {row['ai']} "
                     f"| {row['auroc']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}] "
                     f"| {row['accuracy_at_0.5']:.4f} |")
    return '\n'.join(lines)


def confident_errors(validation, probs, threshold=0.5, count=5):
    # Most confident mistakes: humans with the highest P(AI), AI passages with the lowest.
    rows = validation[['id', 'label']].assign(probability_ai=probs,
                                              words=validation['text'].str.split().str.len())
    fp = rows[(rows['label'] == 0) & (rows['probability_ai'] >= threshold)]
    fn = rows[(rows['label'] == 1) & (rows['probability_ai'] < threshold)]
    fp = fp.sort_values(['probability_ai', 'id'], ascending=[False, True])
    fn = fn.sort_values(['probability_ai', 'id'])
    return len(fp), len(fn), pd.concat([fp.head(count).assign(error='FP'),
                                        fn.head(count).assign(error='FN')])


def error_table(errors):
    lines = ['| ID | Error type | P(AI) | Words |', '|---|---|---:|---:|']
    for row in errors.itertuples():
        lines.append(f'| {row.id} | {row.error} | {row.probability_ai:.4f} | {row.words} |')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--split', type=Path, default=Path(__file__).with_name('validation_ids.json'))
    parser.add_argument('--model', choices=MODELS, default='word_and_style')
    parser.add_argument('--out-dir', type=Path, default=Path('outputs/q4'))
    args = parser.parse_args()
    description, build = MODELS[args.model]
    selected_c = SELECTED_C[args.model]
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error('Output directory is not empty; choose a new --out-dir')
    data = load_data(args.train, labeled=True)
    fingerprint = hashlib.sha256(args.train.read_bytes()).hexdigest()
    training, validation = split_data(data, args.split, fingerprint)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    metadata = {
        'code_commit': commit, 'working_tree_dirty': dirty, 'seed': SEED,
        'train_sha256': fingerprint,
        'split_sha256': hashlib.sha256(args.split.read_bytes()).hexdigest(),
        'python': platform.python_version(), 'sklearn': sklearn.__version__,
        'train_rows': len(training), 'validation_rows': len(validation), 'native_threads': 1,
        'model': description, 'selected_C': selected_c,
        'gap_C_values': list(GAP_C_VALUES),
    }
    (args.out_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')

    rows = []
    for c in GAP_C_VALUES:
        print(f'Fitting C={c:g}...', flush=True)
        features, seconds, train_probs, val_probs, flat_probs = fit_and_predict(
            build, c, training, validation)
        rows.append({'C': c, 'features': features, 'fit_seconds': seconds,
                     'train': summarize(training['label'], train_probs),
                     'validation': summarize(validation['label'], val_probs),
                     'flat_validation_auroc': roc_auc_score(validation['label'], flat_probs)})
        print(json.dumps(rows[-1]), flush=True)
        if c == selected_c:
            selected_val_probs = val_probs
            # Saved for the subgroup and error analysis; joined to text by ID, never by position.
            validation[['id', 'label']].assign(probability_ai=val_probs,
                                               flat_probability_ai=flat_probs).to_csv(
                args.out_dir / 'validation_predictions.csv', index=False)
            training[['id', 'label']].assign(probability_ai=train_probs).to_csv(
                args.out_dir / 'train_predictions.csv', index=False)
    (args.out_dir / 'gap.json').write_text(json.dumps(rows, indent=2) + '\n')

    selected = next(row for row in rows if row['C'] == selected_c)
    lengths = length_bins(validation, selected_val_probs)
    pd.DataFrame(lengths).to_csv(args.out_dir / 'length_bins.csv', index=False)
    fp_count, fn_count, errors = confident_errors(validation, selected_val_probs)
    errors.to_csv(args.out_dir / 'confident_errors.csv', index=False)
    words = validation['text'].str.split().str.len()
    medians = '; '.join([f"overall {words[validation['label'] == 0].median():.0f} vs "
                         f"{words[validation['label'] == 1].median():.0f}"] +
                        [f"{row['bin'].lower()} {row['median_words_human']:.0f} vs "
                         f"{row['median_words_ai']:.0f}" for row in lengths])
    report = f"""# Q4: generalization and error analysis

## Protocol

- Code revision: `{commit}` (dirty: {dirty}); seed: `{SEED}`; Python: `{platform.python_version()}`; scikit-learn `{sklearn.__version__}`.
- Selected model: {description} (`--model {args.model}`), fit on the 8,000 training rows only ({selected['features']:,} features).
- All numbers use the fixed 2,000-row validation set; hidden test labels are not used.

## Q4.1: training vs validation

The selected model scores **{selected['train']['auroc']:.6f}** training AUROC and **{selected['validation']['auroc']:.6f}** validation AUROC (gap {selected['train']['auroc'] - selected['validation']['auroc']:.6f}).

To check whether the gap is regularization-driven, the same features were refit with other L2 strengths (smaller C = stronger regularization). Only C={selected_c:g} is the selected model; the other rows are diagnostics, not a new selection.

{gap_table(rows)}

Log loss and the confident share describe how extreme the probabilities are; AUROC depends only on ranking.

Formatting check: with every whitespace run in the validation passages collapsed to one space (no line breaks or paragraph gaps), the selected model's validation AUROC is **{selected['flat_validation_auroc']:.6f}** (vs {selected['validation']['auroc']:.6f} on the original text).

## Q4.2: validation AUROC by text length

Words are whitespace-separated tokens. Bins are equal-count terciles of the validation set's word counts, chosen so each AUROC uses about the same number of passages. Intervals are percentile bootstraps (2,000 resamples within each bin, seed {SEED}).

{length_table(lengths)}

Median words, human vs AI: {medians}.

## Q4.3: most confident validation errors

Threshold 0.5: a false positive (FP) is a human passage with P(AI) >= 0.5; a false negative (FN) is an AI passage with P(AI) < 0.5. At this threshold the model makes **{fp_count} FPs** (of 1,000 human) and **{fn_count} FNs** (of 1,000 AI). Listed are the five most confident of each: FPs with the highest P(AI), FNs with the lowest. Passage text is not reproduced here; failure hypotheses are written in the report after reading each passage.

{error_table(errors)}

## Reproduction

```sh
.venv/bin/python errors.py --train data/train.csv --model {args.model} --out-dir {args.out_dir.as_posix()}
```

Local outputs: `{args.out_dir.as_posix()}/` (gap.json, train/validation predictions; not committed).
"""
    Path(__file__).with_name('q4_results.md').write_text(report)
    print(f'Results: q4_results.md; outputs: {args.out_dir}')


if __name__ == '__main__':
    main()
