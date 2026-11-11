"""Export report-ready AUROC figures from completed classical-model experiments.

Usage:
    python plot_classical_fixed2000.py --results-dir RESULTS --output-dir FIGURES

The script reads full-precision CSV values, verifies them against summary.json,
and exports PNG, SVG, and PDF figures without fitting any model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator


BLUE = '#245E88'
GOLD = '#B08020'
INK = '#252B31'
GRAY = '#B7BEC5'
MODEL_ORDER = ('tfidf_lr', 'char_lr', 'tfidf_svm', 'word_char_lr')
MODEL_LABELS = {
    'tfidf_lr': 'Word TF–IDF + logistic regression',
    'char_lr': 'Character TF–IDF + logistic regression',
    'tfidf_svm': 'Word TF–IDF + linear SVM',
    'word_char_lr': 'Word + character TF–IDF + logistic regression',
}
PANEL_LABELS = {
    'tfidf_lr': 'Word TF–IDF + LR',
    'char_lr': 'Character TF–IDF + LR',
    'tfidf_svm': 'Word TF–IDF + SVM',
}


def read_table(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'model', 'C', 'seed', 'train_rows', 'validation_rows', 'train_auroc', 'val_auroc'}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f'Missing required columns in {path.name}.')
        rows = list(reader)
    if not rows:
        raise ValueError(f'No rows in {path.name}.')
    for row in rows:
        for key in ('C', 'train_auroc', 'val_auroc'):
            row[key] = float(row[key])
            if not math.isfinite(row[key]):
                raise ValueError(f'Nonfinite {key} in {path.name}.')
        for key in ('seed', 'train_rows', 'validation_rows'):
            value = float(row[key])
            if not math.isfinite(value) or value != int(value):
                raise ValueError(f'Invalid integer {key} in {path.name}.')
            row[key] = int(value)
        if row['C'] <= 0 or min(row['train_rows'], row['validation_rows']) <= 0:
            raise ValueError(f'Nonpositive parameter or row count in {path.name}.')
        if any(not 0 <= row[key] <= 1 for key in ('train_auroc', 'val_auroc')):
            raise ValueError(f'AUROC outside [0, 1] in {path.name}.')
    return rows


def load_results(directory):
    summary_path = directory / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    if summary.get('complete') is not True:
        raise ValueError('The experiment must finish before report figures are exported.')
    protocol = summary['protocol']
    files = {'curve': 'learning_curve.csv', 'tuning': 'tuning.csv', 'comparison': 'comparison_selected.csv'}
    tables = {key: read_table(directory / filename) for key, filename in files.items()}
    for rows in tables.values():
        if any(row['seed'] != protocol['seed'] or row['validation_rows'] != protocol['validation_rows'] for row in rows):
            raise ValueError('CSV seed or validation-row count differs from the recorded protocol.')
    curve = sorted(tables['curve'], key=lambda row: row['train_rows'])
    curve_protocol = protocol['learning_curve']
    if [row['train_rows'] for row in curve] != sorted(curve_protocol['sizes']):
        raise ValueError('The learning curve must contain each planned training size exactly once.')
    if any(row['model'] != curve_protocol['model'] or row['C'] != curve_protocol['C'] for row in curve):
        raise ValueError('Learning-curve model or C changed across training sizes.')
    if set(row['model'] for row in tables['tuning']) != set(MODEL_ORDER[:3]):
        raise ValueError('Expected tuning results for the three individual text models.')
    selected = {row['model']: row for row in tables['comparison']}
    if set(selected) != set(MODEL_ORDER) or len(selected) != len(tables['comparison']):
        raise ValueError('Expected one selected result for each of the four models.')
    for model in MODEL_ORDER[:3]:
        candidates = sorted((row for row in tables['tuning'] if row['model'] == model), key=lambda row: row['C'])
        if [row['C'] for row in candidates] != sorted(protocol['C_values']):
            raise ValueError(f'Tuning grid is missing or duplicated for {model}.')
        best = max(row['val_auroc'] for row in candidates)
        choice = min((row for row in candidates if best - row['val_auroc'] < protocol['tie_tolerance']),
                     key=lambda row: row['C'])
        if choice['C'] != summary['selection'][model]['selected_C'] or choice['C'] != selected[model]['C']:
            raise ValueError(f'Selected C disagrees with the recorded rule for {model}.')
        for key in ('train_auroc', 'val_auroc'):
            if abs(choice[key] - selected[model][key]) > 1e-12:
                raise ValueError(f'Selected metrics disagree with tuning results for {model}.')
    for key, summary_key in (('curve', 'learning_curve'), ('tuning', 'tuning'), ('comparison', 'comparison_selected')):
        recorded = {row['run_id']: row for row in summary[summary_key]}
        for row in tables[key]:
            if row['run_id'] not in recorded or any(abs(row[field] - recorded[row['run_id']][field]) > 1e-12
                                                   for field in ('C', 'train_auroc', 'val_auroc')):
                raise ValueError(f'CSV values disagree with summary.json for {row["run_id"]}.')
    tables['curve'] = curve
    return summary, tables, [summary_path, *[directory / name for name in files.values()]]


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 10.5,
        'axes.labelsize': 10.5, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
        'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': '#7E8790',
        'xtick.color': INK, 'ytick.color': INK, 'axes.linewidth': 0.7,
        'axes.spines.top': False, 'axes.spines.right': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'lines.linewidth': 1.6, 'lines.markersize': 5.5, 'legend.fontsize': 9.5,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    })


def focused_limits(values):
    low, high = min(values), max(values)
    span = max(high - low, 0.025)
    lower = max(0.0, low - span * 0.14)
    upper = min(1.005, high + span * 0.14)
    if upper - lower < 0.025:
        lower = max(0.0, upper - 0.025)
    return lower, upper


def finish_axis(ax, limits):
    ax.set_ylim(*limits)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.3f}'))
    ax.grid(axis='y', color='#E5E8EB', linewidth=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.65)


def save_figure(fig, directory, name):
    paths = []
    for suffix in ('png', 'svg', 'pdf'):
        path = directory / f'{name}.{suffix}'
        fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.08)
        paths.append(str(path.resolve()))
    size = list(fig.get_size_inches())
    plt.close(fig)
    return {'files': paths, 'nominal_size_inches': size}


def plot_learning_curve(summary, tables, output):
    rows = tables['curve']
    fig, ax = plt.subplots(figsize=(7.1, 4.5))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.76, bottom=0.23)
    fig.suptitle('Learning curve', x=0.12, y=0.98, ha='left', fontsize=13, fontweight='semibold')
    fig.text(0.12, 0.91, f'{MODEL_LABELS[rows[0]["model"]]} · C = {rows[0]["C"]:g}', fontsize=9.5)
    x = [row['train_rows'] for row in rows]
    ax.plot(x, [row['train_auroc'] for row in rows], color=BLUE, linestyle='--', marker='o',
            markerfacecolor='white', markeredgewidth=1.3, label='Training')
    ax.plot(x, [row['val_auroc'] for row in rows], color=GOLD, linestyle='-', marker='s', label='Validation')
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(FixedLocator(x))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{int(value):,}'))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(min(x) / 1.13, max(x) * 1.13)
    ax.set_xlabel('Training documents (log scale)')
    ax.set_ylabel('AUROC')
    limits = focused_limits([row[key] for row in rows for key in ('train_auroc', 'val_auroc')])
    finish_axis(ax, limits)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=2, frameon=False, borderaxespad=0)
    fig.text(0.12, 0.075, f'Fixed validation: {summary["protocol"]["validation_rows"]:,} documents; seed {summary["protocol"]["seed"]}.', fontsize=8.5)
    fig.text(0.12, 0.03, 'Stratified training subsets; the 8,000-document fit is reused for the final point.', fontsize=8.5, color='#59636D')
    result = save_figure(fig, output, 'learning_curve')
    result.update(y_limits=limits, x_scale='log', x_ticks=x)
    return result


def plot_hyperparameters(summary, tables, output):
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 4.0), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.72, bottom=0.24, wspace=0.16)
    fig.suptitle('Regularization search', x=0.075, y=0.98, ha='left', fontsize=13, fontweight='semibold')
    fig.text(0.075, 0.90, f'Validation AUROC · {summary["protocol"]["train_rows"]:,} training / {summary["protocol"]["validation_rows"]:,} validation documents', fontsize=9.5)
    limits = focused_limits([row['val_auroc'] for row in tables['tuning']])
    selected_C = {}
    for ax, model in zip(axes, MODEL_ORDER[:3]):
        rows = sorted((row for row in tables['tuning'] if row['model'] == model), key=lambda row: row['C'])
        chosen = next(row for row in rows if row['C'] == summary['selection'][model]['selected_C'])
        selected_C[model] = chosen['C']
        x = [row['C'] for row in rows]
        ax.plot(x, [row['val_auroc'] for row in rows], color=GOLD, marker='s', zorder=2)
        ax.scatter([chosen['C']], [chosen['val_auroc']], s=105, facecolors='none', edgecolors=BLUE, linewidths=1.7, zorder=3)
        ax.set_title(f'{PANEL_LABELS[model]}\nSelected C = {chosen["C"]:g}', pad=11)
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(FixedLocator(x))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(min(x) / 1.6, max(x) * 1.6)
        ax.set_xlabel('C (log scale)')
        finish_axis(ax, limits)
    axes[0].set_ylabel('Validation AUROC')
    tolerance = summary['protocol']['tie_tolerance']
    fig.text(0.075, 0.075, f'Blue ring: smallest C with AUROC loss strictly below {tolerance:g} relative to the best candidate.', fontsize=8.5)
    fig.text(0.075, 0.025, 'Common AUROC scale; the same fixed validation set is used to select C.', fontsize=8.5, color='#59636D')
    result = save_figure(fig, output, 'hyperparameter_search')
    result.update(y_limits=limits, x_scale='log', selected_C=selected_C)
    return result


def plot_model_comparison(summary, tables, output):
    by_model = {row['model']: row for row in tables['comparison']}
    rows = [by_model[model] for model in MODEL_ORDER]
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    fig.subplots_adjust(left=0.43, right=0.97, top=0.74, bottom=0.22)
    fig.suptitle('Selected model comparison', x=0.05, y=0.98, ha='left', fontsize=13, fontweight='semibold')
    fig.text(0.05, 0.91, f'{summary["protocol"]["train_rows"]:,} training / {summary["protocol"]["validation_rows"]:,} fixed validation documents', fontsize=9.5)
    for index, row in enumerate(rows):
        ax.plot([row['train_auroc'], row['val_auroc']], [index, index], color=GRAY, linewidth=1.0, zorder=1)
        ax.scatter(row['train_auroc'], index, color=BLUE, facecolors='white', marker='o', linewidths=1.4,
                   s=40, label='Training' if index == 0 else None, zorder=3)
        ax.scatter(row['val_auroc'], index, color=GOLD, marker='s', s=36,
                   label='Validation' if index == 0 else None, zorder=3)
    ax.set_yticks(range(len(rows)), [f'{MODEL_LABELS[row["model"]]}\nC = {row["C"]:g}' for row in rows])
    ax.set_ylim(len(rows) - 0.55, -0.6)
    limits = focused_limits([row[key] for row in rows for key in ('train_auroc', 'val_auroc')])
    ax.set_xlim(*limits)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.3f}'))
    ax.set_xlabel('AUROC (focused scale)')
    ax.grid(axis='x', color='#E5E8EB', linewidth=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', length=0, pad=8, labelsize=9)
    ax.spines['left'].set_visible(False)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=2, frameon=False, borderaxespad=0)
    fig.text(0.05, 0.075, 'Individual-model C values are validation-selected; combined word + character C is fixed.', fontsize=8.5)
    fig.text(0.05, 0.025, 'Seed 42; parameter selection and evaluation use the same fixed validation set.', fontsize=8.5, color='#59636D')
    result = save_figure(fig, output, 'model_comparison')
    result.update(x_limits=limits, model_order=list(MODEL_ORDER))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    summary, tables, source_paths = load_results(args.results_dir.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()
    figures = {
        'learning_curve': plot_learning_curve(summary, tables, args.output_dir),
        'hyperparameter_search': plot_hyperparameters(summary, tables, args.output_dir),
        'model_comparison': plot_model_comparison(summary, tables, args.output_dir),
    }
    manifest = {'status': 'complete', 'matplotlib_version': matplotlib.__version__,
                'source_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
                'protocol_fingerprint': summary.get('protocol_fingerprint'), 'figures': figures,
                'numeric_policy': 'Full-precision CSV values are used for all plotted coordinates; displayed ticks are formatted only.',
                'interpretation': 'Validation data also select regularization. No confidence intervals or independent-test claims are implied.'}
    (args.output_dir / 'plot_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'complete', 'output_dir': str(args.output_dir.resolve()), 'figures': list(figures)}))


if __name__ == '__main__':
    main()
