"""Reproduce Day 1 experiments, optional structural/blend tests, or the saved learning curve."""
from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from data_utils import audit_data, get_or_create_split, load_data, nested_stratified_subsets
from experiment import THREADS, evaluate, paired_bootstrap, write_json, runtime_identity, verify_environment
from models import SEED


def candidate_name(config):
    key = 'alpha' if config['kind'] == 'word_nb' else 'C'
    value = f'{config[key]:g}'.replace('.', 'p')
    return f'{config["kind"]}_{key}{value}'


def plot_learning_curve(results, figure_out=None):
    """Plot recorded measurements without changing results or report prose."""
    results = Path(results)
    curves = sorted(json.loads((results / 'learning_curve.json').read_text()), key=lambda row: row['train_rows'])
    if [row['train_rows'] for row in curves] != [500, 1000, 2000, 5000, 8000]:
        raise ValueError('Expected exactly the five required learning-curve sizes')
    reference = curves[0]
    for row in curves:
        if row['validation_rows'] != 2000:
            raise ValueError('Learning curve must use the fixed 2,000-row holdout')
        for key in ('config', 'seed', 'source_sha256', 'validation_ids_sha256'):
            if not row.get(key) or row[key] != reference[key]:
                raise ValueError(f'Learning-curve provenance differs or is missing: {key}')
        for key in ('train_auroc', 'validation_auroc'):
            if not np.isfinite(row[key]) or not 0 <= row[key] <= 1:
                raise ValueError(f'Invalid recorded {key}')
    target = Path(figure_out) if figure_out is not None else results / 'learning_curve.pdf'
    if target.suffix.lower() not in {'.pdf', '.png'}:
        raise ValueError('Learning-curve output must be .pdf or .png')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.3, 4.3), layout='constrained')
    for key, label, color in [('train_auroc', 'Training', '#3556a4'),
                              ('validation_auroc', 'Validation', '#12806c')]:
        ax.plot([row['train_rows'] for row in curves], [row[key] for row in curves],
                marker='o', label=label, color=color)
    config = reference['config']
    setting = ', '.join(f'{key}={value}' for key, value in config.items() if key != 'kind')
    ax.set(xlabel='Labeled training rows', ylabel='AUROC',
           title=f'Fixed configuration: {config["kind"]}, {setting}', ylim=(.45, 1.015))
    ax.set_xticks([500, 1000, 2000, 5000, 8000], ['500', '1,000', '2,000', '5,000', '8,000'])
    ax.legend()
    ax.grid(alpha=.2)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180)
    plt.close(fig)
    print(f'Learning curve written to {target}', flush=True)
    return target


def blend_plan(results):
    plan = json.loads((Path(results) / 'extension_plan.json').read_text())
    if plan['config'] != {'kind': 'char_style_blend', 'char_C': 3.0,
                           'word_style_C': 1.0, 'weights': [.5, .5]}:
        raise ValueError('This experiment is the one predeclared equal-weight blend')
    if not plan.get('promotion_rule'):
        raise ValueError('The blend plan must record its promotion rule before evaluation')
    return plan


def run_blend(data_dir, results, *, resume=False):
    """Run the formerly separate extension using the same recorded selection rule."""
    out = Path(results)
    plan = blend_plan(out)
    config = plan['config']
    data = load_data(Path(data_dir) / 'train.csv', expected_rows=10000)
    train, validation = get_or_create_split(data, out / 'split.json')
    audit_path = out / 'source_audit.json'
    if not audit_path.exists():
        audit_path = out / 'audit.json'
    audit = json.loads(audit_path.read_text())
    base = pd.read_csv(out / 'runs' / 'char_lr_C3.predictions.tsv', sep='\t', dtype={'id': str})
    style = pd.read_csv(out / 'bonus' / 'word_style_lr.predictions.tsv', sep='\t', dtype={'id': str})
    expected_ids = validation.id.astype(str).tolist()
    expected_labels = validation.label.tolist()
    for component in (base, style):
        if component.id.tolist() != expected_ids or component.label.tolist() != expected_labels:
            raise ValueError('Blend component predictions do not match the fixed validation IDs and labels')
    record = evaluate(config, train, validation, out / 'extension', 'equal_blend',
                      save_model=True, resume=resume,
                      sensitivity_exclude_ids=audit['source_overlap_sensitivity']['excluded_validation_ids'])
    blend = pd.read_csv(out / 'extension' / 'equal_blend.predictions.tsv', sep='\t', dtype={'id': str})
    if blend.id.tolist() != expected_ids or blend.label.tolist() != expected_labels:
        raise ValueError('Blend predictions do not match the fixed validation IDs and labels')
    expected = (base.probability_ai.to_numpy() + style.probability_ai.to_numpy()) / 2
    if not np.allclose(expected, blend.probability_ai, rtol=0, atol=1e-8):
        raise ValueError('Refitted blend does not match fixed component average')
    effect = paired_bootstrap(base.label, base.probability_ai, blend.probability_ai)
    promote = effect['delta_auroc'] > 0 and effect['ci_95_percentile'][0] > 0
    recommended = config if promote else {'kind': 'char_lr', 'C': 3.0}
    write_json(out / 'extension_comparison.json', {
        'run': record, 'baseline_name': 'char_lr_C3', 'effect_vs_core': effect,
        'promoted_for_day1_handoff': promote, 'recommended_config': recommended,
        'selection_rule': plan['promotion_rule'],
        'component_prediction_max_abs_difference': float(np.max(np.abs(expected - blend.probability_ai))),
        'caution': 'Adaptive exploratory extension selected after earlier comparisons; bootstrap is not confirmatory. Day 2 should review generalization and complexity.'})
    write_json(out / 'recommended_config.json', recommended)
    print(json.dumps({'recommended': recommended, 'effect': effect, 'promote': promote}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, help='Competition data directory; required for fitting')
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument('--bonus', action='store_true', help='Four prespecified style/POS ablations after core experiments')
    parser.add_argument('--blend', action='store_true', help='Run the declared equal blend after --bonus; needs extension_plan.json')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--blend-only', action='store_true', help='Fit the blend using existing core/bonus predictions and extension_plan.json')
    modes.add_argument('--plot-only', action='store_true', help='Plot saved learning_curve.json without training or rewriting report prose')
    parser.add_argument('--figure-out', type=Path, help='PDF/PNG plot path; default: RESULTS/learning_curve.pdf')
    parser.add_argument('--resume', action='store_true', help='Verify and reuse completed runs with identical data/configuration')
    args = parser.parse_args()
    out = args.results
    if args.plot_only:
        if args.bonus or args.blend or args.resume:
            parser.error('--plot-only cannot be combined with --bonus, --blend or --resume')
        plot_learning_curve(out, args.figure_out)
        return
    if args.data is None:
        parser.error('--data is required for fitting')
    if args.blend_only:
        if args.bonus or args.blend or args.figure_out:
            parser.error('--blend-only cannot be combined with --bonus, --blend or --figure-out')
        run_blend(args.data, out, resume=args.resume)
        return
    if args.blend:
        if not args.bonus:
            parser.error('--blend requires --bonus; use --blend-only for existing component predictions')
        blend_plan(out)  # Reject a missing or changed plan before starting the core fits.
    if not args.resume and (out / 'comparison.json').exists():
        parser.error('Comparison results already exist; choose a new results directory or --resume')
    data = load_data(args.data / 'train.csv', expected_rows=10000)
    train, validation = get_or_create_split(data, out / 'split.json', seed=SEED)
    other = {}
    for key, name, labeled, rows in [('unlabeled', 'unlabeled.csv', False, 3000),
                                    ('test', 'test.csv', False, 3000),
                                    ('starter', 'train_starter_2000.csv', True, 2000)]:
        path = args.data / name
        other[key] = load_data(path, labeled=labeled, expected_rows=rows) if path.is_file() else None
    audit = audit_data(data, train, validation, **other)
    write_json(out / 'audit.json', audit)
    sensitivity = audit.get('source_overlap_sensitivity', {}).get('excluded_validation_ids')
    environment = {**runtime_identity(), 'platform': platform.platform(),
                   'seed': SEED, 'threads': THREADS, 'schema': {'id': 'id', 'text': 'text', 'label': 'label'},
                   'source_sha256': data.attrs['source_sha256'], 'NLTK_DATA': os.environ.get('NLTK_DATA', 'not set'),
                   'calibration': 'Three seeded stratified inner folds; sigmoid; training-only shared vocabulary/IDF'}
    if args.resume and (out / 'environment.json').is_file():
        verify_environment(json.loads((out / 'environment.json').read_text()), environment)
    else:
        if args.resume and ((out / 'comparison.json').is_file() or any((out / 'runs').glob('*.json'))):
            parser.error('Recorded runs lack environment.json; cannot verify resume provenance')
        write_json(out / 'environment.json', environment)
    candidates = [{'kind': kind, 'C': c} for kind in ('word_lr', 'char_lr') for c in (.3, 1., 3.)]
    candidates += [{'kind': 'word_nb', 'alpha': alpha} for alpha in (.1, 1., 3.)]
    candidates += [{'kind': 'word_svm', 'C': 1.}]
    plan = {'candidates': candidates, 'selection': 'maximum full-validation AUROC; exact ties first listed',
            'learning_sizes': [500, 1000, 2000, 5000, 8000], 'seed': SEED,
            'hypothesis_scope': 'Data-grounded lexical baselines; structural suggestions are separate exploratory ablations.'}
    plan_path = out / 'search_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        parser.error('Existing search plan differs; choose a new results directory')
    write_json(plan_path, plan)
    runs = []
    for config in candidates:
        runs.append(evaluate(config, train, validation, out / 'runs', candidate_name(config),
                             save_model=True, resume=args.resume, sensitivity_exclude_ids=sensitivity))
    best = max(runs, key=lambda row: row['validation_auroc'])
    write_json(out / 'comparison.json', runs)
    write_json(out / 'selected_config.json', best['config'])
    write_json(out / 'selection.json', {'selected': best,
               'scope': 'Day 1 core comparison; bonus results do not silently replace this selection'})
    curves = []
    for size, subset in nested_stratified_subsets(train, seed=SEED).items():
        write_json(out / 'learning_ids' / f'{size}.json', subset.id.astype(str).tolist())
        curves.append(evaluate(best['config'], subset, validation, out / 'learning', f'n{size}',
                               resume=args.resume, sensitivity_exclude_ids=sensitivity))
    write_json(out / 'learning_curve.json', curves)
    plot_learning_curve(out, args.figure_out)
    if args.bonus:
        configs = [{'kind': kind, 'C': 1.} for kind in ('style_lr', 'pos_lr', 'surprisal_lr', 'word_style_lr')]
        write_json(out / 'bonus_plan.json', {'candidates': configs, 'baseline': {'kind': 'word_lr', 'C': 1.},
                   'fixed': 'same 8000/2000 split, C=1; no bonus tuning',
                   'timing_note': 'POS cache is shared per process; later runs may reuse tagging work.'})
        bonus_runs, failures = [], []
        for config in configs:
            try:
                bonus_runs.append(evaluate(config, train, validation, out / 'bonus', config['kind'],
                                           save_model=True, resume=args.resume, sensitivity_exclude_ids=sensitivity))
            except Exception as exc:
                failure = {'config': config, 'type': type(exc).__name__, 'error': str(exc)}
                failures.append(failure)
                write_json(out / 'bonus' / f'{config["kind"]}.failure.json', failure)
                print(f'BONUS FAILED: {config["kind"]}: {exc}', flush=True)
        baseline_name = candidate_name({'kind': 'word_lr', 'C': 1.})
        baseline = pd.read_csv(out / 'runs' / f'{baseline_name}.predictions.tsv', sep='\t', dtype={'id': str})
        effects = {}
        for run in bonus_runs:
            prediction = pd.read_csv(out / 'bonus' / f'{run["name"]}.predictions.tsv', sep='\t', dtype={'id': str})
            if baseline.id.tolist() != prediction.id.tolist() or baseline.label.tolist() != prediction.label.tolist():
                raise ValueError('Bonus bootstrap inputs are not paired by identical IDs and labels')
            effects[run['name']] = paired_bootstrap(baseline.label, baseline.probability_ai, prediction.probability_ai)
        write_json(out / 'bonus_comparison.json', {'runs': bonus_runs, 'failures': failures, 'effects_vs_word_C1': effects})
    if args.blend:
        run_blend(args.data, out, resume=args.resume)
    print(f'Day 1 measured records written to {out}; selected {best["name"]} AUROC={best["validation_auroc"]:.6f}', flush=True)
    print('Review these measurements and edit the existing course report; this script does not rewrite prose.', flush=True)


if __name__ == '__main__':
    main()
