"""Compare raw, augmented, and consistency arms; export one development-selected candidate."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from compare_results import ARMS, CANDIDATE_ARMS, EXPERIMENT, WINNER_RULE, select_development_winner


MODEL_NAME = 'openai-community/gpt2-large'


def sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(part)
    return hasher.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def child_path(root, relative):
    root = Path(root).resolve()
    result = (root / relative).resolve()
    if root not in result.parents:
        raise ValueError(f'Expected a path inside {root}: {relative}')
    return result


def read_csv(path):
    with Path(path).open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        columns, rows = reader.fieldnames, list(reader)
    if not columns or 'id' not in columns or not rows:
        raise ValueError(f'A nonempty id column is required: {path}')
    values = [row['id'] for row in rows]
    if len(set(values)) != len(values) or any(not value for value in values):
        raise ValueError(f'IDs must be nonempty and unique: {path}')
    return columns, rows


def ids(path):
    return [row['id'] for row in read_csv(path)[1]]


def write_csv(path, columns, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def event(output_dir, name, **details):
    record = {'event': name, 'status': 'running', 'time_utc': datetime.now(timezone.utc).isoformat(), **details}
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + '\n')
        stream.flush()
    atomic_json(output_dir / 'progress.json', record)
    print(json.dumps(record), flush=True)


def run_command(command, log_path, name):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as log:
        process = subprocess.Popen([str(item) for item in command], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(f'[{name}] {line}', end='', flush=True)
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if code:
        raise RuntimeError(f'{name} exited with status {code}; see {log_path}.')


def model_hashes(directory):
    directory = Path(directory)
    if not (directory / 'config.json').is_file() or not list(directory.glob('*.safetensors')):
        raise ValueError(f'Model weights or configuration are missing: {directory}')
    if any(path.is_dir() or path.is_symlink() for path in directory.iterdir()):
        raise ValueError(f'Unexpected model-directory contents: {directory}')
    return {path.name: sha256(path) for path in sorted(directory.iterdir()) if path.is_file()}


def verify_files(root, records):
    for relative, expected in records.items():
        path = child_path(root, relative)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f'A frozen or completed artifact changed: {path}')


def inspect_prepared(root, allow_small_data=False):
    manifest = read_json(root / 'split_manifest.json')
    if manifest.get('seed') != 42:
        raise ValueError('All preparation and training stages must use seed 42.')
    verify_files(root, manifest['files'])
    _, full = read_csv(root / 'train.csv')
    _, development = read_csv(root / 'development.csv')
    _, holdout = read_csv(root / 'fixed_validation.csv')
    _, test = read_csv(root / 'test.csv')
    sample_columns, sample = read_csv(root / 'sample_submission.csv')
    if sample_columns != ['id', 'label'] or set(row['id'] for row in sample) != set(row['id'] for row in test):
        raise ValueError('The sample submission must contain id,label and exactly the test IDs.')
    if not allow_small_data and [len(full), len(development), len(holdout), len(test)] != [10000, 8000, 2000, 3000]:
        raise ValueError('Expected 10000 labeled, 8000 development, 2000 holdout, and 3000 test rows.')
    full_by_id = {row['id']: row for row in full}
    dev_ids, hold_ids = {row['id'] for row in development}, {row['id'] for row in holdout}
    if dev_ids & hold_ids or dev_ids | hold_ids != set(full_by_id):
        raise ValueError('Development and fixed validation must partition all labeled IDs.')
    for row in development + holdout:
        if row['text'] != full_by_id[row['id']]['text'] or row['label'] != full_by_id[row['id']]['label']:
            raise ValueError('A split changed an original parsed text or label value.')
    if [row['id'] for row in holdout] != manifest['fixed_validation']['ids']:
        raise ValueError('Fixed validation ID order changed.')
    all_inner = []
    for fold in range(5):
        fit = set(ids(root / f'folds/fold_{fold}_train_ids.csv'))
        inner_list = ids(root / f'folds/fold_{fold}_validation_ids.csv')
        inner = set(inner_list)
        if fit & inner or fit | inner != dev_ids or fit & hold_ids or inner & hold_ids:
            raise ValueError(f'Invalid development fold boundary: {fold}')
        if len(fit) != 4 * len(development) // 5 or len(inner) != len(development) // 5:
            raise ValueError('Each fold must use four fifths for training and one fifth for validation.')
        for partition in (fit, inner):
            if {int(full_by_id[key]['label']) for key in partition} != {0, 1}:
                raise ValueError('Every classifier partition must contain both label classes.')
        all_inner.extend(inner_list)
    if len(all_inner) != len(dev_ids) or set(all_inner) != dev_ids:
        raise ValueError('The five inner validation folds must cover development rows exactly once.')
    _, adaptation = read_csv(root / 'unlabeled_development.csv')
    adaptation_ids = {row['id'] for row in adaptation}
    lm_validation = set(ids(root / 'unlabeled_validation_ids.csv'))
    if not lm_validation < adaptation_ids:
        raise ValueError('Language-model validation IDs must be a proper nonempty adaptation subset.')
    adaptation_text = [row['text'] for row in adaptation]
    if len(set(adaptation_text)) != len(adaptation_text) or set(adaptation_text) & {row['text'] for row in full}:
        raise ValueError('Language-model adaptation must exclude exact duplicates within unlabeled and against all labeled text.')
    return {'full_rows': len(full), 'development_rows': len(development), 'fixed_validation_rows': len(holdout),
            'test_rows': len(test), 'lm_train_rows': len(adaptation) - len(lm_validation),
            'lm_validation_rows': len(lm_validation), 'raw_text_policy': manifest['raw_text_policy'],
            'files_sha256': {**manifest['files'], 'split_manifest.json': sha256(root / 'split_manifest.json')}}


def resolve_speed(path, model_name_or_path=None, max_length=None, allow_small_data=False):
    benchmark = read_json(path)
    if benchmark.get('status') != 'complete':
        raise ValueError('Runtime settings must come from a completed configuration.')
    if not allow_small_data and (benchmark.get('model') != model_name_or_path or benchmark.get('max_length') != max_length):
        raise ValueError('Runtime settings must match the model and maximum token length.')
    chosen = benchmark.get('chosen', {})
    for stage in ('lm', 'classifier'):
        settings = chosen.get(stage, {})
        for key in ('batch_size', 'gradient_accumulation', 'eval_batch_size'):
            if not isinstance(settings.get(key), int) or isinstance(settings[key], bool) or settings[key] < 1:
                raise ValueError(f'Missing or invalid runtime setting: chosen.{stage}.{key}')
        if not isinstance(settings.get('gradient_checkpointing'), bool):
            raise ValueError(f'Invalid gradient-checkpointing setting for {stage}.')
        if settings.get('precision') not in {'auto', 'bf16', 'fp16', 'fp32'}:
            raise ValueError(f'Invalid precision for {stage}.')
        if settings.get('attention_backend') not in {'flash', 'math', 'flash_or_math'}:
            raise ValueError(f'Invalid attention backend for {stage}.')
    return {stage: {key: chosen[stage][key] for key in ('batch_size', 'gradient_accumulation',
             'eval_batch_size', 'gradient_checkpointing', 'precision', 'attention_backend')}
             for stage in ('lm', 'classifier')}


def stage_receipt(directory, expected_fingerprint, require_model=True):
    receipt = read_json(directory / 'complete.json')
    if receipt.get('status') != 'complete' or receipt.get('fingerprint') != expected_fingerprint:
        raise ValueError(f'A stage completion receipt belongs to different inputs or settings: {directory}')
    verify_files(directory, receipt['files_sha256'])
    model = directory / 'training/best_model'
    if model.exists():
        removal_path = directory / 'model_removed.json'
        if not require_model and removal_path.exists():
            removed = read_json(removal_path)
            if removed['fingerprint'] != expected_fingerprint or removed['model_files_sha256'] != receipt['model_files_sha256']:
                raise ValueError('The partially removed model has no matching verified disposal receipt.')
            # An interrupted disposal can leave only a verified subset of files.
            for path in model.iterdir():
                if (not path.is_file() or path.is_symlink() or path.name not in receipt['model_files_sha256']
                        or sha256(path) != receipt['model_files_sha256'][path.name]):
                    raise ValueError(f'A partially removed checkpoint changed: {model}')
        elif model_hashes(model) != receipt['model_files_sha256']:
            raise ValueError(f'A completed checkpoint changed: {model}')
    elif require_model:
        raise ValueError(f'The completed stage model needed for the next step is missing: {model}')
    else:
        removed = read_json(directory / 'model_removed.json')
        if removed['fingerprint'] != expected_fingerprint or removed['model_files_sha256'] != receipt['model_files_sha256']:
            raise ValueError('The removed temporary model has no matching verified disposal receipt.')
    return receipt


def train_stage(args, configuration, name, stage, model_path, train_csv, *, val_ids=None,
                schedule=None, augmentation_probability=0.0, consistency_weight=0.0, require_model=True):
    verify_files(Path(__file__).parent, configuration['source_sha256'])
    directory = args.output_dir / name
    training = directory / 'training'
    settings = configuration['speed']['lm' if stage == 'lm' else 'classifier']
    identity = {'run_fingerprint': configuration['run_fingerprint'], 'name': name, 'stage': stage,
        'model': str(model_path), 'model_files_sha256': model_hashes(model_path) if Path(model_path).is_dir() else None,
        'train_csv_sha256': sha256(train_csv), 'validation_ids_sha256': sha256(val_ids) if val_ids else None,
        'schedule_sha256': sha256(schedule) if schedule else None, 'speed': settings,
        'augmentation_probability': augmentation_probability, 'consistency_weight': consistency_weight}
    stage_fingerprint = fingerprint(identity)
    if (directory / 'complete.json').exists():
        receipt = stage_receipt(directory, stage_fingerprint, require_model=require_model and stage != 'classifier')
        event(args.output_dir, 'completed_stage_verified', stage=name)
        return training, receipt
    if training.exists():
        raise RuntimeError(f'{name} was interrupted before completion. No optimizer state is saved; use fresh run directories.')
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / 'stage_config.json', identity)
    command = [sys.executable, '-u', Path(__file__).with_name('train_stage.py'), '--stage', stage,
        '--model_name_or_path', model_path, '--train_csv', train_csv, '--output_dir', training,
        '--seed', 42, '--device', args.device, '--max_length', args.max_length,
        '--learning_rate', args.lm_learning_rate if stage == 'lm' else args.learning_rate,
        '--max_epochs', args.lm_max_epochs if stage == 'lm' else args.max_epochs,
        '--min_epochs', 1 if stage == 'lm' else args.min_epochs,
        '--min_learning_rate', args.min_learning_rate, '--weight_decay', args.weight_decay,
        '--warmup_epochs', args.warmup_epochs, '--min_delta', args.min_delta,
        '--lr_patience', args.lr_patience, '--stop_patience', args.stop_patience,
        '--augmentation_probability', augmentation_probability, '--consistency_weight', consistency_weight]
    for key, value in settings.items():
        if key == 'gradient_checkpointing':
            command.append('--gradient_checkpointing' if value else '--no-gradient_checkpointing')
        else:
            command += ['--' + key, value]
    if val_ids:
        command += ['--val_ids', val_ids]
    if schedule:
        command += ['--fixed_schedule', schedule]
    event(args.output_dir, 'training_stage_started', stage=name, training_csv=str(train_csv))
    verify_files(args.prepared_dir, configuration['prepared']['files_sha256'])
    run_command(command, directory / 'training.log', name)
    metrics = read_json(training / 'metrics.json')
    if metrics.get('status') != 'complete':
        raise RuntimeError(f'Training did not complete: {name}')
    required = ['stage_config.json', 'training/config.json', 'training/metrics.json',
                'training/history.json', 'training/selected_schedule.json', 'training/best_checkpoint.json']
    if stage == 'classifier':
        required.append('training/validation_predictions.csv')
        columns, prediction = read_csv(training / 'validation_predictions.csv')
        if columns != ['id', 'label', 'probability'] or [row['id'] for row in prediction] != ids(val_ids):
            raise ValueError('Selected inner-validation predictions do not match the frozen fold IDs.')
        expected_labels = {row['id']: int(row['label']) for row in read_csv(train_csv)[1]}
        if any(int(row['label']) != expected_labels[row['id']] for row in prediction):
            raise ValueError('Selected inner-validation prediction labels differ from their source.')
        score_predictions(prediction)
    for relative in required:
        if not (directory / relative).is_file():
            raise FileNotFoundError(f'A completed training stage omitted a required artifact: {relative}')
    artifacts = [path for path in directory.rglob('*') if path.is_file()
                 and 'best_model' not in path.parts and path.suffix in {'.json', '.csv', '.log'}
                 and path.name not in {'complete.json', 'model_removed.json'}]
    receipt = {'status': 'complete', 'fingerprint': stage_fingerprint, 'stage': name,
               'files_sha256': {str(path.relative_to(directory)): sha256(path) for path in sorted(artifacts)},
               'model_files_sha256': model_hashes(training / 'best_model')}
    atomic_json(directory / 'complete.json', receipt)
    event(args.output_dir, 'training_stage_complete', stage=name, best_epoch=metrics.get('best_epoch'))
    return training, receipt


def remove_stage_model(directory, receipt):
    model = directory / 'training/best_model'
    if model.exists():
        removal_path = directory / 'model_removed.json'
        if removal_path.exists():
            removal = read_json(removal_path)
            if removal['fingerprint'] != receipt['fingerprint'] or removal['model_files_sha256'] != receipt['model_files_sha256']:
                raise ValueError('The temporary-model disposal receipt differs from the completed stage.')
            # A previous interrupted deletion may have removed only some files.
            for path in model.iterdir():
                if not path.is_file() or path.name not in receipt['model_files_sha256'] or sha256(path) != receipt['model_files_sha256'][path.name]:
                    raise ValueError('A temporary model changed after its disposal was authorized.')
        elif model_hashes(model) != receipt['model_files_sha256']:
            raise ValueError('Refusing to remove a temporary checkpoint that differs from its completion receipt.')
        # The disposal receipt is written first so a completed deletion can be recognized after interruption.
        atomic_json(directory / 'model_removed.json', {'fingerprint': receipt['fingerprint'],
                    'model_files_sha256': receipt['model_files_sha256'], 'reason': 'temporary_stage_consumed'})
        shutil.rmtree(model)


def select_schedule(args, folds, arm):
    histories = [read_json(training / 'history.json') for training, _ in folds]
    best_epochs = [int(read_json(training / 'metrics.json')['best_epoch']) for training, _ in folds]
    epochs = int(statistics.median(best_epochs))
    if epochs < 1 or any(not history for history in histories):
        raise ValueError('Cannot select a schedule from empty or invalid fold results.')
    rates = [float(statistics.median([history[min(epoch, len(history)) - 1]['learning_rate']
                                     for history in histories])) for epoch in range(1, epochs + 1)]
    if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError('Selected learning rates must be positive and finite.')
    schedule = {'epochs': epochs, 'learning_rates': rates, 'warmup_epochs': args.warmup_epochs,
                'rule': 'Median of five best epochs; at each selected epoch, median of five observed learning rates, carrying the final observed rate for a shorter history.',
                'fold_best_epochs': best_epochs,
                'fold_history_sha256': [sha256(training / 'history.json') for training, _ in folds],
                'selection_partition': 'development_inner_folds_only'}
    selection_dir = args.output_dir / 'arms' / arm / 'selection'
    path = selection_dir / 'locked_schedule.json'
    if path.exists() and read_json(path) != schedule:
        raise ValueError('The already locked schedule differs from the verified fold results.')
    atomic_json(path, schedule)
    all_predictions = []
    for fold, (training, _) in enumerate(folds):
        all_predictions.extend([{**row, 'fold': fold} for row in read_csv(training / 'validation_predictions.csv')[1]])
    by_id = {row['id']: row for row in all_predictions}
    development_ids = ids(args.prepared_dir / 'development.csv')
    if len(by_id) != len(all_predictions) or set(by_id) != set(development_ids):
        raise ValueError('Inner validation predictions must cover development rows once.')
    write_csv(selection_dir / 'inner_validation_predictions.csv',
              ['id', 'label', 'probability', 'fold'], [by_id[key] for key in development_ids])
    atomic_json(selection_dir / 'summary.json', {'arm': arm, **ARMS[arm], 'schedule': schedule,
        'inner_selection_metrics': score_predictions(all_predictions),
        'interpretation': 'Inner-validation predictions used to select fold checkpoints and the refit schedule.'})
    return path


def historical_baseline_schedule():
    """Verify the self-contained historical raw schedule against its original hash."""
    value = read_json(Path(__file__).with_name('baseline_schedule.json'))
    original = {key: item for key, item in value.items() if key != 'historical_source'}
    serialized = (json.dumps(original, indent=2, ensure_ascii=True, allow_nan=False) + '\n').encode('utf-8')
    if hashlib.sha256(serialized).hexdigest() != value['historical_source']['schedule_sha256']:
        raise ValueError('The historical baseline schedule does not match its recorded original hash.')
    if value['epochs'] != 2 or value['learning_rates'] != [1e-5, 1e-5] or value['warmup_epochs'] != 0.3:
        raise ValueError('Expected the archived two-epoch raw baseline schedule.')
    return value


def lock_baseline_schedule(args, value):
    directory = args.output_dir / 'arms/raw_control/selection'
    path = directory / 'locked_schedule.json'
    if path.exists() and read_json(path) != value:
        raise ValueError('The historical raw baseline schedule changed after locking.')
    atomic_json(path, value)
    atomic_json(directory / 'summary.json', {'arm': 'raw_control', **ARMS['raw_control'], 'schedule': value,
                'selection_source': 'historical_v006_development_cross_validation',
                'fresh_cross_validation_performed': False, 'eligible_for_new_candidate_selection': False,
                'interpretation': 'Fit a fresh 8000-row raw control from the shared adapted model using the prior raw schedule. Select the two new candidates using their new development folds.'})
    return path


def score_predictions(rows):
    import numpy as np
    from sklearn.metrics import log_loss, roc_auc_score
    labels = np.asarray([int(row['label']) for row in rows])
    probability = np.asarray([float(row['probability']) for row in rows])
    if set(labels) != {0, 1} or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('Scoring requires both label classes and finite probabilities in [0, 1].')
    return {'auc': float(roc_auc_score(labels, probability)),
            'bce': float(log_loss(labels, probability, labels=[0, 1])),
            'human_fpr_at_0_5': float(np.mean(probability[labels == 0] >= 0.5))}


def inference_stage(args, configuration, name, checkpoint, input_csv, *, labeled=False, sample=None):
    verify_files(Path(__file__).parent, configuration['source_sha256'])
    verify_files(args.prepared_dir, configuration['prepared']['files_sha256'])
    directory = args.output_dir / name
    identity = {'run_fingerprint': configuration['run_fingerprint'], 'name': name,
                'model_files_sha256': model_hashes(checkpoint), 'input_sha256': sha256(input_csv),
                'sample_sha256': sha256(sample) if sample else None, 'labeled': labeled}
    stage_fingerprint = fingerprint(identity)
    if (directory / 'complete.json').exists():
        receipt = read_json(directory / 'complete.json')
        if receipt['fingerprint'] != stage_fingerprint or receipt['status'] != 'complete':
            raise ValueError('Inference completion receipt belongs to a different model or input.')
        verify_files(directory, receipt['files_sha256'])
        event(args.output_dir, 'completed_inference_verified', stage=name)
        return read_json(directory / 'metrics.json')
    directory.mkdir(parents=True, exist_ok=True)
    prediction_path = directory / ('probabilities.csv' if labeled else 'submission.csv')
    speed = configuration['speed']['classifier']
    command = [sys.executable, '-u', Path(__file__).with_name('predict.py'),
               '--checkpoint', checkpoint, '--predict_csv', input_csv, '--output', prediction_path,
               '--device', args.device, '--precision', speed['precision'],
               '--attention_backend', speed['attention_backend'], '--batch_size', speed['eval_batch_size'],
               '--max_length', args.max_length]
    if sample:
        command += ['--sample_submission', sample]
    event(args.output_dir, 'inference_started', stage=name)
    run_command(command, directory / 'inference.log', name)
    _, probabilities = read_csv(prediction_path)
    summary = {'status': 'complete', 'rows': len(probabilities), 'model_files_sha256': identity['model_files_sha256'],
               'input_sha256': identity['input_sha256'], 'prediction_sha256': sha256(prediction_path)}
    if labeled:
        _, source = read_csv(input_csv)
        if [row['id'] for row in probabilities] != [row['id'] for row in source]:
            raise ValueError('Fixed-validation prediction order differs from the original held-out ID order.')
        predictions = [{'id': target['id'], 'label': int(target['label']), 'probability': float(predicted['label'])}
                       for target, predicted in zip(source, probabilities)]
        write_csv(directory / 'validation_predictions.csv', ['id', 'label', 'probability'], predictions)
        summary.update(metrics=score_predictions(predictions), role='fixed_validation_of_8000_row_refit_only',
                       used_for_selection=False, final_10000_model_evaluated=False)
    else:
        summary.update(role='test_predictions_from_final_10000_row_model', kaggle_submitted=False)
    atomic_json(directory / 'metrics.json', summary)
    artifact_names = [prediction_path.name, 'metrics.json'] + (['validation_predictions.csv'] if labeled else [])
    atomic_json(directory / 'complete.json', {'status': 'complete', 'fingerprint': stage_fingerprint,
                'files_sha256': {name: sha256(directory / name) for name in artifact_names}})
    event(args.output_dir, 'inference_complete', stage=name, rows=len(probabilities))
    return summary


def evaluate_arm(args, configuration, arm, training, receipt, format_inputs):
    """Score all prespecified formats before disposing of an arm's 8000-row model."""
    from evaluate_formats import summarize_formats
    directory = args.output_dir / 'arms' / arm / 'evaluation'
    identity = {'run_fingerprint': configuration['run_fingerprint'], 'arm': arm,
                'model_files_sha256': receipt['model_files_sha256'],
                'format_input_sha256': {name: sha256(path) for name, path in format_inputs.items()}}
    expected = fingerprint(identity)
    if (directory / 'complete.json').exists():
        completed = read_json(directory / 'complete.json')
        if completed.get('fingerprint') != expected or completed.get('status') != 'complete':
            raise ValueError('The completed format evaluation belongs to different inputs or model.')
        verify_files(directory, completed['files_sha256'])
        return read_json(directory / 'metrics.json')
    model = training / 'best_model'
    if model_hashes(model) != receipt['model_files_sha256']:
        raise ValueError('The holdout-fit model changed before format evaluation.')
    results = {}
    for variant, input_csv in format_inputs.items():
        results[variant] = inference_stage(args, configuration, f'arms/{arm}/evaluation/{variant}',
                                          model, input_csv, labeled=True)
    summary = summarize_formats(directory)
    summary.update(arm=arm, **ARMS[arm],
                   run_fingerprint=configuration['run_fingerprint'], model_files_sha256=receipt['model_files_sha256'],
                   locked_schedule=read_json(args.output_dir / 'arms' / arm / 'selection/locked_schedule.json'),
                   role='fixed_validation_of_8000_row_refit_only', used_for_selection=False,
                   final_10000_model_evaluated=False,
                   validation_caveat='These fixed 2000 rows have been examined in earlier experiments; they are a repeated development comparison, not a new untouched test set.')
    atomic_json(directory / 'metrics.json', summary)
    artifacts = [path for path in directory.rglob('*') if path.is_file() and path.suffix in {'.csv', '.json'}]
    atomic_json(directory / 'complete.json', {'status': 'complete', 'fingerprint': expected,
                'files_sha256': {str(path.relative_to(directory)): sha256(path) for path in sorted(artifacts)}})
    event(args.output_dir, 'arm_evaluation_complete', arm=arm, original_metrics=results['original']['metrics'])
    return summary


def export_final(args, configuration, source, winner):
    allowed_names = {'config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
                     'special_tokens_map.json', 'vocab.json', 'merges.txt', 'added_tokens.json',
                     'raw_text_config.json', 'training_metadata.json', 'model.safetensors.index.json'}
    files = [path for path in source.iterdir() if path.name in allowed_names
             or re.fullmatch(r'model(?:-\d+-of-\d+)?\.safetensors', path.name)]
    if not any(path.suffix == '.safetensors' for path in files):
        raise ValueError('The final classifier has no safetensors weights to export.')
    receipt_path = args.output_dir / 'export_manifest.json'
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt['run_fingerprint'] != configuration['run_fingerprint']:
            raise ValueError('The existing final export belongs to a different run.')
        if model_hashes(args.export_dir) != receipt['files_sha256']:
            raise ValueError('The existing final model export has changed.')
        return receipt
    args.export_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=args.export_dir.name + '.staging-', dir=args.export_dir.parent))
    try:
        for path in sorted(files):
            shutil.copyfile(path, temporary / path.name)
            if sha256(temporary / path.name) != sha256(path):
                raise IOError(f'Final model copy verification failed: {path.name}')
        metadata_path = temporary / 'training_metadata.json'
        metadata = read_json(metadata_path) if metadata_path.exists() else {}
        metadata['experiment'] = {'run_fingerprint': configuration['run_fingerprint'],
            'training_rows': configuration['prepared']['full_rows'],
            'training_arm': winner['selected_arm'], **ARMS[winner['selected_arm']],
            'inference_text_policy': 'raw_parsed_values_unchanged',
            'candidate_status': 'New candidate selected between augmentation and augmentation plus consistency using development folds only; the historical final model remains a separate fallback.',
            'candidate_selection': winner,
            'supervised_initialization': 'fresh_classification_head_from_selected_adapted_language_model',
            'schedule_source': 'five_inner_folds_of_8000_development_rows',
            'evaluation_scope': 'Fixed-validation metrics describe the separate 8000-row refit. The final prediction model is fitted on all 10000 labeled rows.',
            'locked_schedule': read_json(args.output_dir / 'arms' / winner['selected_arm'] / 'selection/locked_schedule.json')}
        atomic_json(metadata_path, metadata)
        hashes = model_hashes(temporary)
        if args.export_dir.exists():
            if model_hashes(args.export_dir) != hashes:
                raise FileExistsError('An existing export differs from this final model; it will not be overwritten.')
        else:
            temporary.rename(args.export_dir)
        if model_hashes(args.export_dir) != hashes:
            raise IOError('The committed final model export failed verification.')
        receipt = {'status': 'complete', 'run_fingerprint': configuration['run_fingerprint'],
                   'directory': str(args.export_dir), 'model_count': 1, 'files_sha256': hashes,
                   'runtime_model_files_sha256': model_hashes(source)}
        atomic_json(receipt_path, receipt)
        return receipt
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_finalization(args, configuration, record):
    if record.get('run_fingerprint') != configuration['run_fingerprint'] or record.get('status') != 'complete':
        raise ValueError('The completed experiment belongs to different code, input data, or settings.')
    verify_files(args.output_dir, record['files_sha256'])
    if model_hashes(args.export_dir) != record['exported_model_files_sha256']:
        raise ValueError('The final exported model has changed since completion.')


def finalize_cleanup(args, configuration, record):
    verify_finalization(args, configuration, record)
    for name in ['adaptation', *[f'arms/{arm}/selection/fold_{i}' for arm in CANDIDATE_ARMS for i in range(5)],
                 *[f'arms/{arm}/holdout_fit' for arm in ARMS], 'final_fit']:
        directory = args.output_dir / name
        remove_stage_model(directory, read_json(directory / 'complete.json'))
    atomic_json(args.output_dir / 'complete.json', record)
    failure = args.output_dir / 'failure.json'
    if failure.exists():
        failure.unlink()
    event(args.output_dir, 'experiment_complete', status='complete', final_model=str(args.export_dir),
          submission=str(args.output_dir / 'final_prediction/submission.csv'), model_count=1,
          fixed_validation_scope='8000_row_refit_only', kaggle_submitted=False)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('prepared_dir', 'output_dir', 'export_dir'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--runtime_config', '--benchmark_json', dest='benchmark_json', type=Path, required=True)
    parser.add_argument('--model_name_or_path', default=MODEL_NAME)
    parser.add_argument('--device', choices=['cuda', 'cpu', 'auto'], default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--lm_max_epochs', type=int, default=3)
    parser.add_argument('--lm_learning_rate', type=float, default=1e-5)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--min_epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--min_learning_rate', type=float, default=2.5e-6)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_epochs', type=float, default=0.3)
    parser.add_argument('--min_delta', type=float, default=0.0005)
    parser.add_argument('--lr_patience', type=int, default=2)
    parser.add_argument('--stop_patience', type=int, default=3)
    parser.add_argument('--allow_small_data', action='store_true', help='Use small integration fixtures.')
    parser.add_argument('--preflight_only', action='store_true')
    return parser.parse_args()


def main(args):
    for key in ('prepared_dir', 'output_dir', 'export_dir', 'benchmark_json'):
        setattr(args, key, getattr(args, key).resolve())
    if args.seed != 42 or not 1 <= args.lm_max_epochs <= 3 or not 1 <= args.min_epochs <= args.max_epochs <= 10:
        raise ValueError('Use seed42, at most3 LM epochs, and valid classifier limits up to10 epochs.')
    if (args.output_dir == args.export_dir or args.output_dir in args.export_dir.parents
            or args.export_dir in args.output_dir.parents):
        raise ValueError('Runtime results and the final model export must be in separate directory trees.')
    prepared = inspect_prepared(args.prepared_dir, args.allow_small_data)
    speed = resolve_speed(args.benchmark_json, args.model_name_or_path, args.max_length, args.allow_small_data)
    keys = ('model_name_or_path', 'device', 'seed', 'max_length', 'lm_max_epochs', 'lm_learning_rate',
            'max_epochs', 'min_epochs', 'learning_rate', 'min_learning_rate', 'weight_decay',
            'warmup_epochs', 'min_delta', 'lr_patience', 'stop_patience', 'allow_small_data')
    from evaluate_formats import FORMAT_RULES
    baseline = historical_baseline_schedule()
    configuration = {'experiment': EXPERIMENT,
        'protocol': {key: getattr(args, key) for key in keys}, 'prepared': prepared,
        'speed': speed, 'benchmark_sha256': sha256(args.benchmark_json), 'export_dir': str(args.export_dir),
        'source_sha256': {path.name: sha256(path) for path in sorted(Path(__file__).parent.iterdir())
                          if path.is_file() and path.suffix in {'.py', '.txt', '.json'}},
        'arms': ARMS, 'candidate_arms': list(CANDIDATE_ARMS), 'winner_rule': WINNER_RULE,
        'raw_schedule_source': baseline['historical_source'], 'format_diagnostics': FORMAT_RULES,
        'pairing': {'shared_adaptation': True, 'same_seed': 42, 'same_candidate_fold_ids': True,
                    'same_speed_settings': True, 'same_training_policy': True,
                    'selection_metric': 'original_inner_validation_auc',
                    'schedule_rule': 'B/C use the same new fivefold rule; raw control reuses its archived two-epoch schedule without new cross-validation.',
                    'randomness': 'Shared classifier initialization, data-order seed, and ID/epoch augmentation draws. Consistency forward passes use dropout disabled.'},
        'data_policy': 'Original parsed CSV text is preserved. Adaptation is shared and unaugmented. B/C apply the same stochastic partial newline replacement during supervised training; C adds weighted Bernoulli Jensen-Shannon consistency on changed documents. Validation and final inference use original text. Full format collapses are diagnostic only.'}
    configuration['run_fingerprint'] = fingerprint(configuration)
    if args.preflight_only:
        print(json.dumps({'status': 'preflight_passed', 'configuration': configuration}), flush=True)
        return configuration
    lock = (Path(tempfile.gettempdir()) / f'gpt2-large-adaptation-gpu-{os.getuid()}.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise RuntimeError('Another adaptation experiment is active in this runtime.') from exc
    try:
        config_path = args.output_dir / 'run_config.json'
        if config_path.exists():
            if read_json(config_path) != configuration:
                raise ValueError('Existing results use different inputs, source files, or settings; use fresh directories.')
        else:
            if args.output_dir.exists() and any(args.output_dir.iterdir()):
                raise FileExistsError('A new run needs an empty runtime output directory.')
            if args.export_dir.exists():
                raise FileExistsError('A new run needs a new final model export directory.')
            atomic_json(config_path, configuration)
        args._run_owned = True
        for name in ('complete.json', 'finalization.json'):
            if (args.output_dir / name).exists():
                record = read_json(args.output_dir / name)
                finalize_cleanup(args, configuration, record)
                return read_json(args.output_dir / 'metrics.json')
        started = time.perf_counter()
        event(args.output_dir, 'experiment_started', run_fingerprint=configuration['run_fingerprint'])
        lm, _ = train_stage(args, configuration, 'adaptation', 'lm', args.model_name_or_path,
            args.prepared_dir / 'unlabeled_development.csv', val_ids=args.prepared_dir / 'unlabeled_validation_ids.csv')
        adapted = lm / 'best_model'
        # Freeze all schedules and the B/C candidate choice before any fixed-validation inference.
        schedules = {'raw_control': lock_baseline_schedule(args, baseline)}
        event(args.output_dir, 'schedule_locked', arm='raw_control', source='historical_v006',
              schedule=read_json(schedules['raw_control']))
        for arm in CANDIDATE_ARMS:
            folds = []
            for fold in range(5):
                training, receipt = train_stage(args, configuration, f'arms/{arm}/selection/fold_{fold}',
                    'classifier', adapted, args.prepared_dir / 'development.csv',
                    val_ids=args.prepared_dir / f'folds/fold_{fold}_validation_ids.csv',
                    **ARMS[arm])
                folds.append((training, receipt))
                remove_stage_model(training.parent, receipt)
            schedules[arm] = select_schedule(args, folds, arm)
            event(args.output_dir, 'schedule_locked', arm=arm, schedule=read_json(schedules[arm]))
        winner = select_development_winner(args.output_dir)
        winner_path = args.output_dir / 'selection/winner.json'
        if winner_path.exists() and read_json(winner_path) != winner:
            raise ValueError('The locked candidate choice differs from the verified development results.')
        atomic_json(winner_path, winner)
        event(args.output_dir, 'candidate_locked', selected_arm=winner['selected_arm'],
              candidate_mean_auc=winner['candidate_mean_auc'], rule=WINNER_RULE)
        from evaluate_formats import prepare_format_inputs
        format_inputs = prepare_format_inputs(args.prepared_dir / 'fixed_validation.csv', args.output_dir / 'evaluation_inputs')
        arm_results = {}
        for arm in ARMS:
            evaluation_complete = (args.output_dir / 'arms' / arm / 'evaluation/complete.json').exists()
            holdout_model, receipt = train_stage(args, configuration, f'arms/{arm}/holdout_fit', 'refit', adapted,
                args.prepared_dir / 'development.csv', schedule=schedules[arm],
                **ARMS[arm], require_model=not evaluation_complete)
            arm_results[arm] = evaluate_arm(args, configuration, arm, holdout_model, receipt, format_inputs)
            remove_stage_model(holdout_model.parent, receipt)
        from compare_results import compare
        comparison = compare(args.output_dir)
        atomic_json(args.output_dir / 'comparison.json', comparison)
        final_arm = winner['selected_arm']
        schedule = schedules[final_arm]
        final_model, _ = train_stage(args, configuration, 'final_fit', 'refit', adapted,
            args.prepared_dir / 'train.csv', schedule=schedule, **ARMS[final_arm])
        test_result = inference_stage(args, configuration, 'final_prediction', final_model / 'best_model',
            args.prepared_dir / 'test.csv', sample=args.prepared_dir / 'sample_submission.csv')
        export = export_final(args, configuration, final_model / 'best_model', winner)
        result = {'status': 'complete', 'run_fingerprint': configuration['run_fingerprint'],
            'experiment': configuration['experiment'], 'adaptation': read_json(lm / 'metrics.json'),
            'arms': arm_results, 'comparison': comparison, 'final_arm': final_arm, 'candidate_selection': winner,
            'locked_schedule': read_json(schedule), 'final_training': read_json(final_model / 'metrics.json'),
            'test_prediction': test_result, 'final_model_directory': str(args.export_dir), 'saved_model_count': 1,
            'evaluation_scope': 'Fixed-validation metrics describe the separate 8000-row refit. The final prediction model is fitted on all 10000 labeled rows.',
            'elapsed_seconds_this_process': time.perf_counter() - started, 'kaggle_submitted': False}
        atomic_json(args.output_dir / 'metrics.json', result)
        artifacts = [path for path in args.output_dir.rglob('*') if path.is_file()
                     and 'best_model' not in path.parts and path.suffix in {'.json', '.csv'}
                     and path.name not in {'progress.json', 'failure.json', 'model_removed.json', 'finalization.json'}]
        record = {'status': 'complete', 'run_fingerprint': configuration['run_fingerprint'],
                  'files_sha256': {str(path.relative_to(args.output_dir)): sha256(path) for path in sorted(artifacts)},
                  'exported_model_files_sha256': export['files_sha256']}
        atomic_json(args.output_dir / 'finalization.json', record)
        finalize_cleanup(args, configuration, record)
        return result
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    args = arguments()
    try:
        main(args)
    except Exception as exc:
        if getattr(args, '_run_owned', False):
            atomic_json(args.output_dir / 'failure.json', {'status': 'failed', 'error': str(exc),
                        'traceback': traceback.format_exc()})
            event(args.output_dir, 'experiment_failed', status='failed', error=str(exc))
        raise
