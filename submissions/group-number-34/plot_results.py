"""Plot the five training sizes in learning_curve.csv as a single PDF."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator


TRAIN_SIZES = [500, 1000, 2000, 5000, 8000]
MODEL_LABELS = {
    'tfidf_lr': 'Word TF-IDF + logistic regression',
    'char_lr': 'Character TF-IDF + logistic regression',
    'tfidf_svm': 'Word TF-IDF + linear SVM',
    'word_char_lr': 'Word + character TF-IDF + logistic regression',
}


def plot_learning_curve(results_dir, output_dir):
    with (results_dir / 'learning_curve.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    rows.sort(key=lambda row: int(row['train_rows']))
    sizes = [int(row['train_rows']) for row in rows]
    if sizes != TRAIN_SIZES:
        raise ValueError('Expected one row for each training size: 500, 1000, 2000, 5000, 8000.')
    train = [float(row['train_auroc']) for row in rows]
    validation = [float(row['val_auroc']) for row in rows]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in train + validation):
        raise ValueError('AUROC values must be finite and between 0 and 1.')
    for field in ('model', 'C', 'seed', 'validation_rows'):
        if len({row.get(field, '') for row in rows}) != 1:
            raise ValueError(f'All learning-curve rows must use the same {field}.')

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.labelsize': 10.5,
        'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
        'text.color': '#252B31', 'axes.labelcolor': '#252B31',
        'axes.edgecolor': '#7E8790', 'axes.linewidth': 0.7,
        'axes.spines.top': False, 'axes.spines.right': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'pdf.fonttype': 42,
    })
    fig, ax = plt.subplots(figsize=(7.1, 4.5))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.76, bottom=0.23)
    fig.suptitle('Learning curve', x=0.12, y=0.98, ha='left', fontsize=13, fontweight='semibold')
    model = rows[0].get('model', '')
    subtitle = MODEL_LABELS.get(model, model)
    if rows[0].get('C'):
        subtitle += f' (C = {float(rows[0]["C"]):g})'
    fig.text(0.12, 0.91, subtitle, fontsize=9.5)
    ax.plot(sizes, train, '--o', color='#245E88', markerfacecolor='white',
            markeredgewidth=1.3, linewidth=1.6, markersize=5.5, label='Training')
    ax.plot(sizes, validation, '-s', color='#B08020', linewidth=1.6,
            markersize=5.5, label='Validation')
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(FixedLocator(sizes))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{int(value):,}'))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(min(sizes) / 1.13, max(sizes) * 1.13)
    ax.set_xlabel('Training documents (log scale)')
    ax.set_ylabel('AUROC')
    low, high = min(train + validation), max(train + validation)
    span = max(high - low, 0.025)
    upper = min(1.005, high + span * 0.14)
    ax.set_ylim(max(0.0, min(low - span * 0.14, upper - 0.025)), upper)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.3f}'))
    ax.grid(axis='y', color='#E5E8EB', linewidth=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.65)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=2,
              frameon=False, borderaxespad=0, fontsize=9.5)
    details = []
    if rows[0].get('validation_rows'):
        details.append(f'Fixed validation: {int(rows[0]["validation_rows"]):,} documents')
    if rows[0].get('seed'):
        details.append(f'seed {rows[0]["seed"]}')
    fig.text(0.12, 0.075, '; '.join(details) + '.', fontsize=8.5)
    fig.text(0.12, 0.03, 'Stratified training subsets; the 8,000-document fit is reused for the final point.',
             fontsize=8.5, color='#59636D')
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / 'learning_curve.pdf'
    fig.savefig(output, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(plot_learning_curve(args.results_dir, args.output_dir))
