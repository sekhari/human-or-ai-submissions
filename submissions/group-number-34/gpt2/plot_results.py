"""Plot saved learning curves, model comparisons, and validation diagnostics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ['MPLBACKEND'] = 'Agg'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE = '#236B82', '#D78636'


def save_figure(figure, output_dir, name):
    figure.tight_layout()
    paths = []
    for extension in ['png', 'pdf']:
        path = output_dir / f'{name}.{extension}'
        figure.savefig(path, dpi=180, bbox_inches='tight')
        paths.append(str(path))
    plt.close(figure)
    return paths


def auc_limits(ax, values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite):
        margin = max(0.002, float(finite.max() - finite.min()) * 0.15)
        ax.set_ylim(max(0, float(finite.min()) - margin), min(1.005, float(finite.max()) + margin))
    ax.grid(axis='y', alpha=0.2)
    ax.set_ylabel('AUROC')


def plot_results(results_dir, output_dir):
    results_dir, output_dir = Path(results_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    curve_path = results_dir / 'learning_curve.csv'
    if curve_path.is_file():
        frame = pd.read_csv(curve_path).sort_values('training_rows')
        figure, ax = plt.subplots(figsize=(7.2, 4.3))
        ax.plot(frame.training_rows, frame.train_auc, 'o-', color=BLUE, label='Training')
        ax.plot(frame.training_rows, frame.val_auc, 'o--', color=ORANGE, label='Validation')
        ax.set_xticks(frame.training_rows)
        ax.set_xlabel('Labeled training passages')
        ax.set_title('GPT-2 learning curve')
        ax.legend(frameon=False)
        auc_limits(ax, np.r_[frame.train_auc, frame.val_auc])
        paths += save_figure(figure, output_dir, 'learning_curve')
    comparison_path = results_dir / 'comparison.csv'
    if comparison_path.is_file():
        frame = pd.read_csv(comparison_path)
        figure, ax = plt.subplots(figsize=(8, 4.3))
        positions = np.arange(len(frame))
        if {'format', 'baseline_auc', 'candidate_auc'}.issubset(frame.columns):
            ax.plot(positions, frame.baseline_auc, 'o-', color=BLUE, label='Format augmentation')
            ax.plot(positions, frame.candidate_auc, 'o--', color=ORANGE, label='Format augmentation + consistency')
            ax.set_xticks(positions, frame['format'].str.replace('_', ' '))
            ax.set_title('Validation AUROC across text formats')
            values = np.r_[frame.baseline_auc, frame.candidate_auc]
            name = 'format_comparison'
        elif {'train_auc', 'val_auc'}.issubset(frame.columns):
            label_column = next((column for column in ['arm', 'model', 'experiment', 'configuration']
                                 if column in frame.columns), None)
            labels = frame[label_column].astype(str).str.replace('_', ' ') if label_column else frame.index.astype(str)
            ax.plot(positions, frame.train_auc, 'o-', color=BLUE, label='Training')
            ax.plot(positions, frame.val_auc, 'o--', color=ORANGE, label='Validation')
            ax.set_xticks(positions, labels)
            ax.set_title('Model comparison')
            values = np.r_[frame.train_auc, frame.val_auc]
            name = 'model_comparison'
        else:
            raise ValueError('Comparison CSV requires AUROC columns and model or format labels.')
        ax.legend(frameon=False)
        auc_limits(ax, values)
        paths += save_figure(figure, output_dir, name)
        if {'format', 'baseline_mean_probability_change', 'candidate_mean_probability_change'}.issubset(frame.columns):
            transformed = frame.loc[frame['format'] != 'original']
            figure, ax = plt.subplots(figsize=(7.2, 4.3))
            positions = np.arange(len(transformed))
            ax.bar(positions - 0.18, transformed.baseline_mean_probability_change, 0.36, color=BLUE, label='Format augmentation')
            ax.bar(positions + 0.18, transformed.candidate_mean_probability_change, 0.36, color=ORANGE, label='Format augmentation + consistency')
            ax.set_xticks(positions, transformed['format'].str.replace('_', ' '))
            ax.set_ylabel('Mean absolute probability change')
            ax.set_title('Prediction changes after formatting transformations')
            maximum = max(transformed.baseline_mean_probability_change.max(),
                          transformed.candidate_mean_probability_change.max())
            ax.set_ylim(0, max(float(maximum), 1e-6) * 1.35)
            ax.legend(frameon=False)
            paths += save_figure(figure, output_dir, 'probability_drift')
    metric_path = results_dir / 'metrics.json'
    if metric_path.is_file():
        metrics = json.loads(metric_path.read_text())
        values = metrics.get('metrics', metrics)
        if 'training' in values and 'validation' in values:
            figure, ax = plt.subplots(figsize=(5.6, 4.1))
            scores = [values['training']['auc'], values['validation']['auc']]
            ax.plot([0, 1], scores, 'o-', color=BLUE)
            ax.set_xticks([0, 1], ['Training', 'Validation'])
            ax.set_title('Training and validation AUROC')
            auc_limits(ax, scores)
            paths += save_figure(figure, output_dir, 'training_validation')
    subgroup_path = results_dir / 'length_subgroups.csv'
    if subgroup_path.is_file():
        frame = pd.read_csv(subgroup_path)
        figure, ax = plt.subplots(figsize=(7.5, 4.3))
        x = np.arange(len(frame))
        y = frame.auc.to_numpy(dtype=float)
        ax.vlines(x, frame.auc_ci_lower, frame.auc_ci_upper, color=BLUE)
        ax.scatter(x, y, color=BLUE)
        ax.set_xticks(x, [f'{name}\n(n={count:,})' for name, count in zip(frame.subgroup, frame.rows)])
        ax.set_title('Validation AUROC by passage length')
        auc_limits(ax, np.r_[frame.auc_ci_lower, frame.auc_ci_upper])
        paths += save_figure(figure, output_dir, 'length_subgroups')
    if not paths:
        raise ValueError('No supported result files were found.')
    return paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(plot_results(args.results_dir, args.output_dir or args.results_dir / 'figures'), indent=2))
