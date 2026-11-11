"""Q2: Model 2 learning curve on nested balanced training subsets."""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from common import (SIZES, SEED, SIMILARITY, config, require, normalized, near_reference,
                    load_inputs, subsets, fit_model, new_run, save_fitted, write_json, run_metadata)


def plot(table, out):
    os.environ.setdefault('MPLCONFIGDIR', str(out / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.ticker import StrMethodFormatter
    fig, ax = plt.subplots(figsize=(8.2, 5.0), layout='constrained')
    ax.plot(table.training_rows, table.training_auroc, 'o-', color='#1765AD', lw=2, label='Training AUROC')
    ax.plot(table.training_rows, table.validation_auroc, 's-', color='#D86720', lw=2, label='Validation AUROC')
    for x, y in zip(table.training_rows, table.validation_auroc):
        ax.annotate(f'{y:.4f}', (x, y), xytext=(0, -18), textcoords='offset points', ha='center', fontsize=9)
    ax.set(xlabel='Number of labeled training examples', ylabel='AUROC', xticks=SIZES,
           ylim=(0.5, 1.025), title='Model 2: character TF-IDF + calibrated linear SVM')
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.grid(alpha=.2); ax.legend(loc='lower right')
    fig.suptitle('Fixed 2,000-example validation set; fixed selected configuration; seed = 42', fontsize=10)
    fig.savefig(out / 'learning_curve.png', dpi=200)
    fig.savefig(out / 'learning_curve.pdf')
    plt.close(fig)


def run(data_dir, output_dir, reference_dir=None):
    out = new_run(output_dir)
    train, val = load_inputs(data_dir)
    require(not set(train.text.map(normalized)) & set(val.text.map(normalized)), 'Normalized duplicate across split.')
    for name, frame in [('train', train), ('validation', val)]:
        frame[['id', 'label']].to_csv(out / f'{name}_ids.csv', index=False)
    print('Checking existing train/validation near-duplicate sensitivity...', flush=True)
    overlap = near_reference(train, val)
    overlap.to_csv(out / 'validation_overlap.csv', index=False)
    excluded = overlap.loc[overlap.word_trigram_cosine >= SIMILARITY, 'id'].tolist()
    rows = []
    for size, subset in subsets(train):
        point = new_run(out / f'n_{size}')
        subset[['id', 'label']].to_csv(point / 'train_ids.csv', index=False)
        fold_ids = np.zeros(size, dtype=int)
        for i, (_, heldout) in enumerate(StratifiedKFold(3, shuffle=True, random_state=SEED).split(subset.text, subset.label)):
            fold_ids[heldout] = i
        pd.DataFrame({'id': subset.id.to_numpy(), 'calibration_heldout_fold': fold_ids}).to_csv(
            point / 'calibration_folds.csv', index=False)
        print(f'Q2: fitting {size:,} examples, including training-only calibration...', flush=True)
        model, seconds, caught = fit_model(subset)
        metrics = save_fitted(point, model, {'train': subset, 'validation': val}, seconds, caught,
                              sensitivity_ids=excluded)
        row = {'training_rows': size, 'training_auroc': metrics['train_auroc'],
               'validation_auroc': metrics['validation_auroc'],
               'gap': metrics['train_auroc'] - metrics['validation_auroc'],
               'fit_seconds_including_calibration': seconds, 'feature_count': metrics['feature_count'],
               'validation_sensitivity_auroc': metrics['validation_sensitivity_auroc']}
        rows.append(row)
        print(f"  train={row['training_auroc']:.6f}; validation={row['validation_auroc']:.6f}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / 'learning_curve.csv', index=False)
    plot(table, out)
    reference = {'checked': False}
    if reference_dir:
        differences = {}
        for name in ('train', 'validation'):
            old = pd.read_csv(Path(reference_dir) / f'{name}_predictions.csv', dtype={'id': str})
            new = pd.read_csv(out / 'n_8000' / f'{name}_predictions.csv', dtype={'id': str})
            require(old.id.tolist() == new.id.tolist(), 'Teammate ordered IDs differ.')
            require(old.label.tolist() == new.label.tolist(), 'Teammate labels differ.')
            for col in ('margin', 'probability'):
                differences[f'{name}_{col}_max_abs'] = float(np.max(np.abs(old[col] - new[col])))
                require(np.allclose(old[col], new[col], atol=1e-8, rtol=0), f'Teammate {name} {col} differs.')
        reference = {'checked': True, 'status': 'PASS', 'differences': differences}
    write_json(out / 'summary.json', {**run_metadata(), 'status': 'TRAINING_COMPLETE', 'question': 'Q2',
                                     'rows': rows, 'sampling': 'Nested balanced subsets; default_rng(42); Model 2 row order',
                                     'replicates_per_size': 1, 'sensitivity_excluded_validation_ids': excluded,
                                     'reference_comparison': reference})
    print('Q2 complete; ready for independent verification.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--reference-dir', type=Path, help='Optional teammate selected prediction folder.')
    a = p.parse_args()
    run(a.data_dir, a.output_dir, a.reference_dir)
