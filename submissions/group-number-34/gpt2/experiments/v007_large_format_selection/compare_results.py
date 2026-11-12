"""Select the new candidate on development folds and compare all held-out arms."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from evaluate_formats import FORMAT_RULES, atomic_json, digest, prediction_rows, require_matched, score


EXPERIMENT = 'v007_three_arm_format_consistency'
ARMS = {
    'raw_control': {'augmentation_probability': 0.0, 'consistency_weight': 0.0},
    'format_augmented': {'augmentation_probability': 0.25, 'consistency_weight': 0.0},
    'format_consistency': {'augmentation_probability': 0.25, 'consistency_weight': 0.5},
}
CANDIDATE_ARMS = ('format_augmented', 'format_consistency')
WINNER_RULE = {
    'eligible_arms': list(CANDIDATE_ARMS),
    'metric': 'equal_mean_of_five_selected_original_inner_validation_auc_values',
    'tie_tolerance': 0.0001,
    'tie_priority': list(CANDIDATE_ARMS),
    'timing': 'Lock before all fixed-validation inference.',
    'raw_control_role': 'Historical-schedule reference, excluded from the new candidate selection.',
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def select_from_scores(candidate_mean_auc):
    """Choose the simpler candidate if it is within the prespecified tolerance."""
    if set(candidate_mean_auc) != set(CANDIDATE_ARMS):
        raise ValueError('The winner rule requires exactly the two new candidate arms.')
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in candidate_mean_auc.values()):
        raise ValueError('Candidate mean AUC values must be finite and in [0, 1].')
    maximum = max(candidate_mean_auc.values())
    return next(arm for arm in CANDIDATE_ARMS
                if maximum - candidate_mean_auc[arm] <= WINNER_RULE['tie_tolerance'] + 1e-12)


def verify_candidate_pairing(root):
    """Require matched actual IDs, adapted initialization, and speed settings for B/C."""
    paired_fields = ('stage', 'model_files_sha256', 'train_csv_sha256', 'validation_ids_sha256', 'speed')
    for fold in range(5):
        directories = [root / 'arms' / arm / f'selection/fold_{fold}' for arm in CANDIDATE_ARMS]
        configs = [read_json(directory / 'stage_config.json') for directory in directories]
        if any(configs[0][field] != configs[1][field] for field in paired_fields):
            raise ValueError(f'Candidate fold inputs or runtime settings differ: {fold}')
        for arm, config in zip(CANDIDATE_ARMS, configs):
            if any(config[key] != value for key, value in ARMS[arm].items()):
                raise ValueError('Candidate training did not use the prespecified treatment settings.')
        for name in ('train_ids.csv', 'val_ids.csv'):
            if digest(directories[0] / 'training' / name) != digest(directories[1] / 'training' / name):
                raise ValueError('The two candidates did not train and select on matching IDs.')


def select_development_winner(results_dir):
    """Read only candidate development predictions and schedules; never holdout results."""
    root = Path(results_dir)
    configuration = read_json(root / 'run_config.json')
    if configuration.get('experiment') != EXPERIMENT or configuration.get('arms') != ARMS:
        raise ValueError('Expected the prespecified three-arm experiment.')
    if configuration.get('winner_rule') != WINNER_RULE:
        raise ValueError('The recorded candidate selection rule differs from the implementation.')
    verify_candidate_pairing(root)
    means, folds, inputs = {}, {}, {}
    reference = {}
    for arm in CANDIDATE_ARMS:
        folds[arm] = []
        for fold in range(5):
            path = root / 'arms' / arm / f'selection/fold_{fold}/training/validation_predictions.csv'
            rows = prediction_rows(path)
            if fold in reference:
                require_matched(reference[fold], rows)
            else:
                reference[fold] = rows
            metrics = score(rows)
            folds[arm].append({'fold': fold, **metrics})
            inputs[str(path.relative_to(root))] = digest(path)
        means[arm] = sum(item['auc'] for item in folds[arm]) / 5
        schedule = root / 'arms' / arm / 'selection/locked_schedule.json'
        inputs[str(schedule.relative_to(root))] = digest(schedule)
    return {'status': 'locked', 'run_fingerprint': configuration['run_fingerprint'],
            'selected_arm': select_from_scores(means), 'candidate_mean_auc': means,
            'fold_selection_metrics': folds, 'rule': WINNER_RULE, 'selection_inputs_sha256': inputs,
            'fixed_validation_used': False, 'historical_raw_fold_scores_used': False,
            'interpretation': 'Inner-validation predictions select fold checkpoints and compare candidate means. Fixed-validation evaluation follows candidate selection.'}


def compare(results_dir):
    root = Path(results_dir)
    configuration = read_json(root / 'run_config.json')
    winner = read_json(root / 'selection/winner.json')
    if winner != select_development_winner(root):
        raise ValueError('The locked candidate differs from the verified development-only rule.')
    summaries = {arm: read_json(root / 'arms' / arm / 'evaluation/metrics.json') for arm in ARMS}
    holdout_configs = {arm: read_json(root / 'arms' / arm / 'holdout_fit/stage_config.json') for arm in ARMS}
    paired_fields = ('stage', 'model_files_sha256', 'train_csv_sha256', 'validation_ids_sha256', 'speed')
    for arm, summary in summaries.items():
        if summary['run_fingerprint'] != configuration['run_fingerprint'] or summary['arm'] != arm:
            raise ValueError('An arm evaluation belongs to a different run.')
        if any(summary[key] != value or holdout_configs[arm][key] != value for key, value in ARMS[arm].items()):
            raise ValueError('An arm treatment setting differs from the run configuration.')
        if summary['used_for_selection'] or summary['final_10000_model_evaluated']:
            raise ValueError('Invalid fixed-validation evaluation scope.')
        if any(holdout_configs[arm][field] != holdout_configs['raw_control'][field] for field in paired_fields):
            raise ValueError('Holdout refits must share the adapted model, training data, and runtime settings.')
    variants = {}
    reference = None
    for variant in FORMAT_RULES:
        paths = {arm: root / 'arms' / arm / 'evaluation' / variant / 'validation_predictions.csv' for arm in ARMS}
        rows = {arm: prediction_rows(path) for arm, path in paths.items()}
        if reference is None:
            reference = rows['raw_control']
        metrics = {}
        for arm in ARMS:
            require_matched(reference, rows[arm])
            metrics[arm] = score(rows[arm])
            recorded = summaries[arm]['formats'][variant]
            if digest(paths[arm]) != recorded['prediction_sha256']:
                raise ValueError('A format prediction file differs from its recorded hash.')
            if any(abs(recorded['metrics'][key] - value) > 1e-12 for key, value in metrics[arm].items()):
                raise ValueError('Recorded format scores disagree with prediction files.')
        deltas = {}
        for left, right in (('format_augmented', 'raw_control'), ('format_consistency', 'raw_control'),
                            ('format_consistency', 'format_augmented')):
            deltas[f'{left}_minus_{right}'] = {key: metrics[left][key] - metrics[right][key] for key in metrics[left]}
        variants[variant] = {**metrics, 'differences': deltas}
    return {'status': 'complete', 'experiment': EXPERIMENT, 'run_fingerprint': configuration['run_fingerprint'],
            'rows_per_format': len(reference), 'primary_metric': 'original.auc', 'formats': variants,
            'locked_schedules': {arm: summaries[arm]['locked_schedule'] for arm in ARMS},
            'candidate_selection': winner,
            'shared_settings': {'protocol': configuration['protocol'], 'speed': configuration['speed'],
                                'pairing': configuration['pairing'], 'benchmark_sha256': configuration['benchmark_sha256']},
            'comparison_scope': 'Three fresh 8000-row refits from the shared adapted model, evaluated on matching fixed-validation IDs. Raw uses its archived two-epoch schedule without new cross-validation; B/C schedules and their final-candidate choice use only matching new development folds.',
            'validation_usage': 'Fixed validation set reused across development experiments.',
            'evaluation_design': 'One training seed; paired formats of the same validation documents; inner folds used for checkpoint and candidate selection.',
            'final_candidate_arm': winner['selected_arm'], 'winner_selected_from_holdout': False,
            'historical_final_model_replaced_automatically': False, 'final_10000_model_evaluated': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results_dir', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = compare(args.results_dir)
    atomic_json(args.output or args.results_dir / 'comparison.json', result)
    print(json.dumps(result, indent=2), flush=True)
