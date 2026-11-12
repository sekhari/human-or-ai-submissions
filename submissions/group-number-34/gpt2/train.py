"""Run GPT-2 training, controlled comparisons, and final test prediction."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from src.data import TRAINING_SIZES, prepare_data, read_csv, sha256, write_csv, write_json

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ('final', 'learning-curve', 'unlabeled', 'diagnostics', 'consistency')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def build_stage_command(config, stage, model, train_csv, output_dir, device,
                        validation_ids=None, augmentation=0.0, consistency=0.0):
    command = [sys.executable, '-u', str(ROOT / 'src/train_stage.py'),
               '--stage', stage, '--model_name_or_path', str(model),
               '--train_csv', str(train_csv), '--output_dir', str(output_dir),
               '--device', device]
    for name in ('seed', 'max_length', 'batch_size', 'gradient_accumulation',
                 'eval_batch_size', 'precision', 'attention_backend', 'weight_decay',
                 'warmup_epochs', 'learning_rate', 'min_learning_rate', 'min_delta',
                 'lr_patience', 'stop_patience'):
        command.extend(['--' + name, str(config[name])])
    command.extend(['--no-gradient_checkpointing', '--min_epochs',
                    str(config['lm_min_epochs'] if stage == 'lm' else 3),
                    '--augmentation_probability', str(augmentation),
                    '--consistency_weight', str(consistency)])
    if stage == 'lm':
        command.extend(['--val_ids', str(validation_ids), '--max_epochs',
                        str(config['adaptation_epochs'])])
    else:
        command.extend(['--fixed_schedule', str(ROOT / 'configs/schedule.json'),
                        '--max_epochs', str(config['classification_epochs'])])
    return command


def run_stage(config, stage, model, train_csv, directory, device,
              validation_ids=None, augmentation=0.0, consistency=0.0):
    directory = Path(directory)
    command = build_stage_command(config, stage, model, train_csv, directory, device,
                                  validation_ids, augmentation, consistency)
    receipt = directory / 'runtime.json'
    checkpoint = directory / 'best_model'
    if receipt.exists():
        result = read_json(receipt)
        if result['command'] != command or sha256(checkpoint / 'model.safetensors') != result['checkpoint_sha256']:
            raise ValueError(f'Training stage does not match the saved run: {directory}')
        return checkpoint, result['training_seconds']
    print(json.dumps({'event': 'stage_start', 'stage': stage, 'directory': str(directory)}), flush=True)
    started = time.perf_counter()
    subprocess.run(command, check=True)
    elapsed = time.perf_counter() - started
    metrics = read_json(directory / 'metrics.json')
    if metrics['status'] != 'complete':
        raise RuntimeError(f'Training stage did not complete: {directory}')
    write_json(receipt, {'command': command, 'training_seconds': elapsed,
                        'checkpoint_sha256': sha256(checkpoint / 'model.safetensors')})
    return checkpoint, elapsed


def evaluate(config, checkpoint, train_csv, validation_csv, output_dir, device,
             format_variants=False, baseline_dir=None):
    output_dir = Path(output_dir)
    command = [sys.executable, str(ROOT / 'evaluate.py'), '--checkpoint', str(checkpoint),
               '--train-csv', str(train_csv), '--validation-csv', str(validation_csv),
               '--output-dir', str(output_dir), '--device', device,
               '--precision', config['precision'], '--attention-backend', config['attention_backend'],
               '--batch-size', str(config['eval_batch_size'])]
    if format_variants:
        command.append('--format-variants')
    if baseline_dir is not None:
        command.extend(['--baseline-dir', str(baseline_dir)])
    started = time.perf_counter()
    subprocess.run(command, check=True)
    elapsed = time.perf_counter() - started
    metrics = read_json(output_dir / 'metrics.json')
    write_json(output_dir / 'runtime.json', {'evaluation_seconds': elapsed})
    return metrics, elapsed


def result_row(name, metrics, training_seconds, adaptation_seconds=0.0, evaluation_seconds=0.0,
               training_rows=8000):
    return {'arm': name, 'training_rows': training_rows,
            'train_auc': metrics['training']['auc'], 'val_auc': metrics['validation']['auc'],
            'train_bce': metrics['training']['bce'], 'val_bce': metrics['validation']['bce'],
            'classification_epochs': 9, 'seed': 42,
            'training_seconds': training_seconds, 'adaptation_seconds': adaptation_seconds,
            'total_training_seconds': training_seconds + adaptation_seconds,
            'evaluation_seconds': evaluation_seconds}


def write_results(output_dir, name, rows):
    write_csv(Path(output_dir) / name, list(rows[0]), rows)


def load_reused_classifier(run_dir, config, prepared, augmentation, consistency):
    """Read a completed run with matching training data and hyperparameters."""
    run_dir = Path(run_dir).resolve()
    candidates = [run_dir / 'training', run_dir / 'classifier',
                  run_dir / 'size_8000/training', run_dir / 'supervised_only/training']
    stage = next((path for path in candidates if (path / 'config.json').is_file()), None)
    if stage is None:
        raise ValueError(f'No completed classifier was found in {run_dir}.')
    actual = read_json(stage / 'config.json')
    expected = {key: config[key] for key in ('seed', 'max_length', 'batch_size',
        'gradient_accumulation', 'eval_batch_size', 'precision', 'attention_backend',
        'weight_decay', 'warmup_epochs', 'learning_rate')}
    expected.update(augmentation_probability=augmentation, consistency_weight=consistency,
                    training_rows=8000)
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f'Reused classifier setting differs: {key}')
    for name in ('train_stage.py', 'training_utils.py'):
        if actual.get('source_hashes', {}).get(name) != sha256(ROOT / 'src' / name):
            raise ValueError(f'Reused classifier training implementation differs: {name}')
    if actual.get('train_sha256') != sha256(Path(prepared) / 'train_8000.csv'):
        raise ValueError('Reused classifier training rows or row order differ.')
    if actual.get('fixed_schedule_contents', {}).get('learning_rates') != read_json(ROOT / 'configs/schedule.json')['learning_rates']:
        raise ValueError('Reused classifier learning-rate schedule differs.')
    if read_json(stage / 'metrics.json')['completed_epochs'] != 9:
        raise ValueError('Reused classifier must have completed nine epochs.')
    runtime = read_json(stage / 'runtime.json')
    checkpoint = stage / 'best_model'
    if sha256(checkpoint / 'model.safetensors') != runtime['checkpoint_sha256']:
        raise ValueError('Reused classifier weights differ from its completed run.')
    return checkpoint, runtime['training_seconds'], stage


def matching_heads(left, right):
    key = 'initial_classifier_head_sha256'
    a, b = read_json(Path(left) / 'backbone_load_report.json'), read_json(Path(right) / 'backbone_load_report.json')
    if a[key] != b[key]:
        raise ValueError('Compared classifiers used different head initializations.')


def prepare(args, config):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = {'experiment': args.experiment, 'configuration': config,
                'data_dir': str(args.data_dir.resolve()), 'device': args.device,
                'supervised_run': str(args.supervised_run.resolve()) if args.supervised_run else None,
                'baseline_run': str(args.baseline_run.resolve()) if args.baseline_run else None,
                'validation_ids_sha256': sha256(args.validation_ids),
                'schedule_sha256': sha256(ROOT / 'configs/schedule.json'),
                'source_sha256': {str(path.relative_to(ROOT)): sha256(path)
                                 for path in [ROOT / 'train.py', ROOT / 'src/data.py',
                                              ROOT / 'src/train_stage.py', ROOT / 'src/training_utils.py']}}
    config_path, prepared = output / 'run_config.json', output / 'prepared'
    if config_path.exists():
        if read_json(config_path) != protocol:
            raise ValueError('Choose a new output directory for a different experiment configuration.')
        manifest = read_json(prepared / 'split_manifest.json')
        for name, expected in manifest['files_sha256'].items():
            if sha256(prepared / name) != expected:
                raise ValueError(f'Prepared data changed: {name}')
        for name, expected in manifest['source_files_sha256'].items():
            source = args.validation_ids if name == 'fixed_validation_ids.csv' else args.data_dir / name
            if sha256(source) != expected:
                raise ValueError(f'Source data changed: {name}')
    else:
        if any(output.iterdir()):
            raise FileExistsError('Choose an empty output directory for the experiment.')
        manifest = prepare_data(args.data_dir, args.validation_ids, prepared,
                                args.experiment, config['seed'])
        write_json(config_path, protocol)
    return output, prepared, manifest


def main(args):
    config = read_json(ROOT / 'configs' / (args.experiment + '.json'))
    if args.precision:
        config['precision'] = args.precision
    if args.attention_backend:
        config['attention_backend'] = args.attention_backend
    root, prepared, manifest = prepare(args, config)
    if args.prepare_only:
        print(json.dumps({'status': 'prepared', 'experiment': args.experiment,
                          'directory': str(prepared), 'manifest': manifest}, indent=2))
        return
    validation = prepared / 'fixed_validation.csv'
    model = config['model_name_or_path']
    if args.experiment == 'learning-curve':
        rows = []
        for size in TRAINING_SIZES:
            directory = root / f'size_{size}'
            training = prepared / f'train_{size}.csv'
            checkpoint, seconds = run_stage(config, 'refit', model, training,
                                            directory / 'training', args.device)
            measured, eval_seconds = evaluate(config, checkpoint, training, validation,
                                              directory / 'evaluation', args.device)
            rows.append(result_row(f'size_{size}', measured, seconds,
                                   evaluation_seconds=eval_seconds, training_rows=size))
            write_results(root, 'learning_curve.csv', rows)
        write_json(root / 'metrics.json', {'status': 'complete', 'experiment': args.experiment,
                                          'results': rows})
        return

    if args.experiment == 'unlabeled':
        training = prepared / 'train_8000.csv'
        if args.supervised_run:
            supervised, supervised_seconds, supervised_stage = load_reused_classifier(
                args.supervised_run, config, prepared, 0.0, 0.0)
            actual = read_json(supervised_stage / 'config.json')
            if actual['model_name_or_path'] != model:
                raise ValueError('The supervised comparison must start from pretrained GPT-2 small.')
        else:
            supervised_stage = root / 'supervised_only/training'
            supervised, supervised_seconds = run_stage(config, 'refit', model, training,
                                                         supervised_stage, args.device)
        measured, eval_seconds = evaluate(config, supervised, training, validation,
                                          root / 'supervised_only/evaluation', args.device)
        rows = [result_row('supervised_only', measured, supervised_seconds,
                           evaluation_seconds=eval_seconds)]
        write_results(root, 'comparison.csv', rows)
        adapted, adaptation_seconds = run_stage(config, 'lm', model, prepared / 'unlabeled.csv',
            root / 'adaptation', args.device, prepared / 'lm_validation_ids.csv')
        stage = root / 'with_unlabeled/training'
        checkpoint, seconds = run_stage(config, 'refit', adapted, training, stage, args.device)
        matching_heads(supervised_stage, stage)
        measured, eval_seconds = evaluate(config, checkpoint, training, validation,
                                          root / 'with_unlabeled/evaluation', args.device)
        rows.append(result_row('with_unlabeled', measured, seconds, adaptation_seconds, eval_seconds))
        write_results(root, 'comparison.csv', rows)
        write_json(root / 'metrics.json', {'status': 'complete', 'experiment': args.experiment,
                                          'results': rows})
        return

    adapted, adaptation_seconds = run_stage(config, 'lm', model, prepared / 'unlabeled.csv',
        root / 'adaptation', args.device, prepared / 'lm_validation_ids.csv')
    training = prepared / ('train_all.csv' if args.experiment == 'final' else 'train_8000.csv')
    if args.experiment == 'consistency':
        if args.baseline_run:
            baseline, baseline_seconds, baseline_stage = load_reused_classifier(
                args.baseline_run, config, prepared, config['augmentation_probability'], 0.0)
            prior_model = Path(read_json(baseline_stage / 'config.json')['model_name_or_path'])
            if sha256(prior_model / 'model.safetensors') != sha256(adapted / 'model.safetensors'):
                raise ValueError('The reused comparison must use the same adapted language-model weights.')
        else:
            baseline_stage = root / 'baseline/training'
            baseline, baseline_seconds = run_stage(config, 'refit', adapted, training,
                baseline_stage, args.device, augmentation=config['augmentation_probability'])
        measured, eval_seconds = evaluate(config, baseline, training, validation,
            root / 'baseline/evaluation', args.device, format_variants=True)
        rows = [result_row('format_augmentation', measured, baseline_seconds,
                           adaptation_seconds, eval_seconds)]
        stage = root / 'candidate/training'
        checkpoint, seconds = run_stage(config, 'refit', adapted, training, stage,
            args.device, augmentation=config['augmentation_probability'],
            consistency=config['consistency_weight'])
        matching_heads(baseline_stage, stage)
        measured, eval_seconds = evaluate(config, checkpoint, training, validation,
            root / 'candidate/evaluation', args.device, format_variants=True,
            baseline_dir=root / 'baseline/evaluation')
        rows.append(result_row('format_consistency', measured, seconds, adaptation_seconds, eval_seconds))
        write_results(root, 'comparison.csv', rows)
        write_json(root / 'metrics.json', {'status': 'complete', 'experiment': args.experiment,
            'adaptation_shared_between_arms': True, 'results': rows,
            'format_comparison': 'candidate/evaluation/comparison.csv'})
        return

    checkpoint, seconds = run_stage(config, 'refit', adapted, training, root / 'classifier',
                                    args.device, augmentation=config['augmentation_probability'])
    if args.experiment == 'final':
        command = [sys.executable, str(ROOT / 'src/train_stage.py'), '--stage', 'predict',
                   '--checkpoint', str(checkpoint), '--predict_csv', str(prepared / 'test.csv'),
                   '--sample_submission', str(prepared / 'sample_submission.csv'),
                   '--prediction_output', str(root / 'submission.csv'), '--device', args.device,
                   '--precision', config['precision'], '--attention_backend', config['attention_backend'],
                   '--max_length', str(config['max_length']), '--eval_batch_size', str(config['eval_batch_size']),
                   '--seed', str(config['seed'])]
        subprocess.run(command, check=True)
        write_json(root / 'metrics.json', {'status': 'complete', 'experiment': args.experiment,
            'training_rows': 10000, 'classification_epochs': 9,
            'adaptation_seconds': adaptation_seconds, 'training_seconds': seconds,
            'checkpoint': str(checkpoint), 'submission': 'submission.csv'})
    else:
        measured, eval_seconds = evaluate(config, checkpoint, training, validation,
                                          root / 'evaluation', args.device)
        row = result_row('format_augmentation', measured, seconds, adaptation_seconds, eval_seconds)
        write_results(root, 'comparison.csv', [row])
        write_json(root / 'metrics.json', {'status': 'complete', 'experiment': args.experiment,
                                          'results': [row]})


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=EXPERIMENTS, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--validation-ids', type=Path, default=ROOT / 'configs/fixed_validation_ids.csv')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--precision', choices=['auto', 'bf16', 'fp16', 'fp32'])
    parser.add_argument('--attention-backend', choices=['flash', 'math', 'flash_or_math'])
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--supervised-run', type=Path, help='Completed learning-curve run to reuse for the supervised comparison.')
    parser.add_argument('--baseline-run', type=Path, help='Completed diagnostics run to reuse for the consistency comparison.')
    args = parser.parse_args()
    if args.supervised_run and args.experiment != 'unlabeled':
        parser.error('--supervised-run applies to the unlabeled experiment.')
    if args.baseline_run and args.experiment != 'consistency':
        parser.error('--baseline-run applies to the consistency experiment.')
    return args


if __name__ == '__main__':
    main(arguments())
