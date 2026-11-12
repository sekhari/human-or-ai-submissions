"""Run raw LM adaptation and classifiers with optional training-only format augmentation."""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import itertools
import json
import math
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from sklearn.metrics import roc_auc_score

from training_utils import (MODEL_NAME, RAW_TEXT_CONFIG, atomic_json, attention_context,
    autocast_context, batch_weight, choose_device, choose_precision, classifier_epoch_data, encode_frame,
    forward_loss, forward_consistency_loss, load_model, make_loader, make_tokenizer, read_frame, read_ids,
    save_model, set_learning_rate, set_seed, sha256_file, split_frame, training_augmentation_config,
    training_consistency_config)


@dataclass
class TrainingPolicy:
    learning_rate: float
    min_learning_rate: float
    min_delta: float
    lr_patience: int
    stop_patience: int
    min_epochs: int
    best_score: float = -math.inf
    best_epoch: int = 0
    significant_best_score: float = -math.inf
    bad_epochs: int = 0
    reductions: int = 0

    def observe(self, score, epoch):
        if not math.isfinite(score):
            raise ValueError('Inner-validation AUC must be finite.')
        improved = score > self.best_score
        if improved:
            self.best_score, self.best_epoch = score, epoch
        significant = not math.isfinite(self.significant_best_score) or score - self.significant_best_score >= self.min_delta
        if significant:
            self.significant_best_score, self.bad_epochs = score, 0
        else:
            self.bad_epochs += 1
        reduced = False
        if (not significant and self.learning_rate > self.min_learning_rate * (1 + 1e-12)
                and self.reductions < 2 and self.bad_epochs >= self.lr_patience and epoch >= self.min_epochs):
            self.learning_rate = max(self.min_learning_rate, self.learning_rate * 0.5)
            self.reductions += 1
            self.bad_epochs = 0
            reduced = True
        stop = (not reduced and self.learning_rate <= self.min_learning_rate * (1 + 1e-12)
                and self.bad_epochs >= self.stop_patience and epoch >= self.min_epochs)
        return {'improved': improved, 'significant_improvement': significant,
                'learning_rate_reduced': reduced, 'stop': stop,
                'bad_epochs': self.bad_epochs, 'reductions': self.reductions,
                'next_learning_rate': self.learning_rate}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['lm', 'classifier', 'refit', 'predict'], required=True)
    parser.add_argument('--model_name_or_path', default=MODEL_NAME)
    parser.add_argument('--train_csv', type=Path)
    parser.add_argument('--val_ids', type=Path)
    parser.add_argument('--fixed_schedule', type=Path)
    parser.add_argument('--output_dir', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--predict_csv', type=Path)
    parser.add_argument('--prediction_output', type=Path)
    parser.add_argument('--sample_submission', type=Path)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--gradient_accumulation', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--gradient_checkpointing', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--precision', choices=['auto', 'bf16', 'fp16', 'fp32'], default='auto')
    parser.add_argument('--attention_backend', choices=['flash', 'math', 'flash_or_math'], default='flash_or_math')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--augmentation_probability', type=float, default=0.0)
    parser.add_argument('--consistency_weight', type=float, default=0.0)
    parser.add_argument('--max_epochs', type=int)
    parser.add_argument('--min_epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--min_learning_rate', type=float, default=2.5e-6)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_epochs', type=float, default=0.3)
    parser.add_argument('--min_delta', type=float, default=0.0005)
    parser.add_argument('--lr_patience', type=int, default=2)
    parser.add_argument('--stop_patience', type=int, default=3)
    return parser.parse_args()


def read_schedule(path):
    schedule = json.loads(Path(path).read_text(encoding='utf-8'))
    epochs, rates = schedule.get('epochs'), schedule.get('learning_rates')
    if type(epochs) is not int or not 1 <= epochs <= 10 or not isinstance(rates, list) or len(rates) != epochs:
        raise ValueError('A fixed schedule requires 1 through 10 epochs and exactly one rate per epoch.')
    if any(isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError('Fixed learning rates must be finite positive numbers.')
    if schedule.get('warmup_epochs', 0.3) != 0.3:
        raise ValueError('The fixed classifier schedule must use 0.3 warmup epochs.')
    return schedule


@torch.inference_mode()
def evaluate_lm(model, loader, device, precision, backend):
    model.eval()
    total_loss, total_tokens, documents = 0.0, 0, 0
    for batch in loader:
        loss, tokens = forward_loss(model, batch, 'lm', device, precision, backend)
        total_loss += float(loss.cpu())
        total_tokens += tokens
        documents += len(batch['input_ids'])
    if not total_tokens or not math.isfinite(total_loss):
        raise FloatingPointError('LM validation returned invalid loss or no targets.')
    nll = total_loss / total_tokens
    return {'nll': nll, 'perplexity': math.exp(nll) if nll < 700 else None,
            'target_tokens': total_tokens, 'documents': documents, 'loss_sum': total_loss}


@torch.inference_mode()
def evaluate_classifier(model, loader, device, precision, backend, labels=None):
    model.eval()
    pieces, loss_sum = [], 0.0
    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        with attention_context(device, backend), autocast_context(device, precision):
            logits = model(**inputs).logits.squeeze(-1)
        if 'labels' in batch:
            loss_sum += float(F.binary_cross_entropy_with_logits(logits.float(), batch['labels'].to(device), reduction='sum').cpu())
        pieces.append(torch.sigmoid(logits.float()).cpu().numpy())
    probabilities = np.concatenate(pieces).astype(np.float64)
    if not np.isfinite(probabilities).all():
        raise FloatingPointError('Classification returned non-finite probabilities.')
    metrics = None
    if labels is not None:
        labels = np.asarray(labels)
        if set(labels) != {0, 1}:
            raise ValueError('AUC evaluation requires both classes.')
        metrics = {'auc': float(roc_auc_score(labels, probabilities)), 'bce': loss_sum / len(labels),
                   'human_fpr_at_0_5': float(np.mean(probabilities[labels == 0] >= 0.5))}
    return probabilities, metrics


def run_prediction(checkpoint, predict_csv, prediction_output, sample_submission=None,
                   device='auto', eval_batch_size=4, max_length=512, precision='auto',
                   attention_backend='flash_or_math', seed=42):
    checkpoint = Path(checkpoint)
    metadata = json.loads((checkpoint / 'training_metadata.json').read_text())
    if metadata.get('stage') not in {'classifier', 'refit'} or metadata.get('raw_text_config') != RAW_TEXT_CONFIG:
        raise ValueError('Prediction requires a selected raw-text classification checkpoint.')
    if metadata.get('max_length') != max_length:
        raise ValueError('Prediction max_length must match the saved model.')
    if json.loads((checkpoint / 'raw_text_config.json').read_text()) != RAW_TEXT_CONFIG:
        raise ValueError('Saved raw-text handling does not match this predictor.')
    set_seed(seed)
    target_device = choose_device(device)
    actual_precision = choose_precision(precision, target_device)
    frame = read_frame(predict_csv)
    if sample_submission is not None:
        ids = read_ids(sample_submission)
        if len(ids) != len(frame) or set(ids) != set(frame.id):
            raise ValueError('Sample-submission and prediction IDs must match exactly.')
        frame = frame.set_index('id').loc[ids].reset_index()
    tokenizer = make_tokenizer(checkpoint)
    model, loading = load_model(checkpoint, 'classifier', tokenizer, target_device)
    if loading['missing_keys'] or loading['unexpected_keys']:
        raise RuntimeError('Prediction must load every trained classifier parameter.')
    dataset = encode_frame(frame[['id', 'text']], tokenizer, max_length, 'classifier')
    probabilities, _ = evaluate_classifier(model, make_loader(dataset, tokenizer, eval_batch_size),
                                            target_device, actual_precision, attention_backend)
    output = Path(prediction_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    pd.DataFrame({'id': frame.id, 'label': probabilities}).to_csv(temporary, index=False)
    temporary.replace(output)
    result = {'status': 'complete', 'stage': 'predict', 'rows': len(frame),
              'output': str(output), 'sha256': sha256_file(output),
              'precision': actual_precision, 'attention_backend': attention_backend}
    atomic_json(output.with_suffix(output.suffix + '.metadata.json'), result)
    print(json.dumps(result), flush=True)
    return result


def validate_arguments(args):
    # Keep programmatic callers (including the unchanged benchmark) compatible.
    args.augmentation_probability = getattr(args, 'augmentation_probability', 0.0)
    args.consistency_weight = getattr(args, 'consistency_weight', 0.0)
    training_augmentation_config(args.augmentation_probability, args.seed)
    training_consistency_config(args.consistency_weight)
    if args.consistency_weight > 0 and (args.stage not in {'classifier', 'refit'} or args.augmentation_probability <= 0):
        raise ValueError('Consistency requires nonzero classifier/refit training augmentation; never LM or prediction.')
    if args.stage in {'lm', 'predict'} and args.augmentation_probability != 0:
        raise ValueError('Format augmentation applies only to classifier/refit training, never LM or prediction.')
    if args.stage == 'predict':
        if not args.checkpoint or not args.predict_csv or not args.prediction_output:
            raise ValueError('Prediction requires checkpoint, predict_csv, and prediction_output.')
        return
    if not args.train_csv or not args.output_dir:
        raise ValueError('Training requires train_csv and output_dir.')
    if args.stage in {'lm', 'classifier'} and (not args.val_ids or args.fixed_schedule):
        raise ValueError('Selection stages require val_ids and prohibit fixed_schedule.')
    if args.stage == 'refit' and (not args.fixed_schedule or args.val_ids):
        raise ValueError('Refitting requires fixed_schedule and prohibits validation IDs.')
    for key in ('batch_size', 'gradient_accumulation', 'eval_batch_size', 'min_epochs', 'lr_patience', 'stop_patience'):
        if getattr(args, key) < 1:
            raise ValueError(f'{key} must be positive.')
    if not 2 <= args.max_length <= 1024 or not 0 < args.min_learning_rate <= args.learning_rate:
        raise ValueError('Invalid maximum length or learning-rate settings.')
    if args.warmup_epochs != 0.3 or args.weight_decay < 0 or args.min_delta <= 0:
        raise ValueError('Use 0.3 warmup epochs, nonnegative weight decay, and positive min_delta.')
    if args.max_epochs is None:
        args.max_epochs = 3 if args.stage == 'lm' else 10
    limit = 3 if args.stage == 'lm' else 10
    if not 1 <= args.max_epochs <= limit:
        raise ValueError(f'This stage permits 1 through {limit} epochs.')
    if args.stage == 'classifier' and args.min_epochs > args.max_epochs:
        raise ValueError('min_epochs cannot exceed max_epochs for classifier selection.')


def main(args):
    validate_arguments(args)
    if args.stage == 'predict':
        return run_prediction(args.checkpoint, args.predict_csv, args.prediction_output,
                              args.sample_submission, args.device, args.eval_batch_size,
                              args.max_length, args.precision, args.attention_backend, args.seed)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.run.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / 'config.json').exists() or (out / 'best_model').exists():
        raise FileExistsError('Choose an unused output directory; optimizer resumption is intentionally unsupported.')
    args._run_owned = True
    set_seed(args.seed)
    device = choose_device(args.device)
    precision = choose_precision(args.precision, device)
    task = 'lm' if args.stage == 'lm' else 'classifier'
    augmentation_config = training_augmentation_config(args.augmentation_probability, args.seed)
    consistency_config = training_consistency_config(args.consistency_weight)
    schedule = read_schedule(args.fixed_schedule) if args.stage == 'refit' else None
    epochs = schedule['epochs'] if schedule else args.max_epochs
    if schedule:
        training, validation = read_frame(args.train_csv, labeled=True), None
        if set(training.label) != {0, 1}:
            raise ValueError('Refitting requires both label classes.')
    else:
        training, validation = split_frame(args.train_csv, args.val_ids, labeled=task == 'classifier')
    configuration = {key: str(value.resolve()) if isinstance(value, Path) else value
                     for key, value in vars(args).items() if not key.startswith('_')}
    configuration.update(training_rows=len(training), validation_rows=0 if validation is None else len(validation),
                         actual_precision=precision, device=str(device), raw_text_config=RAW_TEXT_CONFIG,
                         training_augmentation_config=augmentation_config,
                         training_consistency_config=consistency_config,
                         train_sha256=sha256_file(args.train_csv),
                         val_ids_sha256=sha256_file(args.val_ids) if args.val_ids else None,
                         fixed_schedule_contents=schedule,
                         fixed_schedule_sha256=sha256_file(args.fixed_schedule) if args.fixed_schedule else None,
                         source_hashes={p.name: sha256_file(p) for p in Path(__file__).parent.glob('*.py')},
                         deterministic_algorithms=True,
                         tokenization=('clean_cached_once; changed_training_texts_retokenized_each_epoch'
                                       if augmentation_config['enabled'] else 'cached_once_per_partition'))
    atomic_json(out / 'config.json', configuration)
    training[['id']].to_csv(out / 'train_ids.csv', index=False)
    if validation is not None:
        validation[['id']].to_csv(out / 'val_ids.csv', index=False)
    environment = {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
                   'transformers': transformers.__version__, 'device': str(device), 'precision': precision,
                   'cuda_version': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
                   'attention_backend': args.attention_backend, 'deterministic_algorithms': True}
    atomic_json(out / 'environment.json', environment)
    print(json.dumps({'event': 'setup', 'stage': args.stage, 'training_rows': len(training),
                      'validation_rows': configuration['validation_rows'], 'epochs': epochs, **environment}), flush=True)
    tokenizer = make_tokenizer(args.model_name_or_path)
    training_data = encode_frame(training, tokenizer, args.max_length, task)
    validation_loader = (make_loader(encode_frame(validation, tokenizer, args.max_length, task),
                         tokenizer, args.eval_batch_size) if validation is not None else None)
    model, loading = load_model(args.model_name_or_path, task, tokenizer, device,
                               args.gradient_checkpointing, require_adapted=task == 'classifier' and args.model_name_or_path != MODEL_NAME)
    if args.max_length > model.config.n_positions:
        raise ValueError('Requested sequence length exceeds the checkpoint positional capacity.')
    if task == 'classifier':
        loading['initial_classifier_head_sha256'] = hashlib.sha256(
            model.score.weight.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        loading['initial_position_embedding_sha256'] = hashlib.sha256(
            model.transformer.wpe.weight.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        loading['pretrained_revision'] = getattr(model.config, '_commit_hash', None)
    atomic_json(out / 'backbone_load_report.json', loading)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and precision == 'fp16')
    policy = TrainingPolicy(args.learning_rate, args.min_learning_rate, args.min_delta,
                            args.lr_patience, args.stop_patience, args.min_epochs)
    updates_per_epoch = math.ceil(math.ceil(len(training) / args.batch_size) / args.gradient_accumulation)
    warmup_updates = int(updates_per_epoch * args.warmup_epochs)
    history, global_step, skipped_updates = [], 0, 0
    augmentation_history = []
    best_epoch, best_nll, baseline = 0, math.inf, None
    checkpoint = out / 'best_model'
    started = time.perf_counter()
    if task == 'lm':
        baseline = evaluate_lm(model, validation_loader, device, precision, args.attention_backend)
        history.append({'epoch': 0, 'baseline_only': True, 'validation': baseline,
                        'validation_nll': baseline['nll'], 'training_loss': None, 'learning_rate': None})
        atomic_json(out / 'baseline_metrics.json', baseline)
        atomic_json(out / 'history.json', history)
        print(json.dumps({'event': 'lm_baseline', **baseline}), flush=True)
    stop_reason = 'fixed_schedule_complete' if schedule else 'max_epochs'
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        epoch_data, augmentation_report = training_data, None
        if task == 'classifier':
            epoch_data, augmentation_report = classifier_epoch_data(
                training, training_data, tokenizer, args.max_length, seed=args.seed, epoch=epoch,
                augmentation_probability=args.augmentation_probability,
                consistency_weight=args.consistency_weight)
            augmentation_history.append(augmentation_report)
            atomic_json(out / 'augmentation_counts.json', augmentation_history)
            print(json.dumps({'event': 'training_augmentation', 'stage': args.stage,
                              **augmentation_report}), flush=True)
        loader = make_loader(epoch_data, tokenizer, args.batch_size, shuffle=True, seed=args.seed + epoch)
        epoch_rate = float(schedule['learning_rates'][epoch - 1]) if schedule else policy.learning_rate
        set_learning_rate(optimizer, epoch_rate, global_step, warmup_updates)
        model.train()
        loss_sum, units, documents = 0.0, 0, 0
        consistency_sum, consistency_documents = 0.0, 0
        print(json.dumps({'event': 'epoch_start', 'stage': args.stage, 'epoch': epoch, 'learning_rate': epoch_rate}), flush=True)
        iterator = iter(loader)
        while True:
            group = list(itertools.islice(iterator, args.gradient_accumulation))
            if not group:
                break
            denominator = sum(batch_weight(batch, task) for batch in group)
            if denominator <= 0:
                raise ValueError('An accumulation group has no loss targets.')
            optimizer.zero_grad(set_to_none=True)
            for batch in group:
                loss, count = forward_loss(model, batch, task, device, precision, args.attention_backend)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite training loss.')
                # Checkpoint recomputation must use the same deterministic backend.
                with attention_context(device, args.attention_backend):
                    scaler.scale(loss / denominator).backward()
                loss_sum += float(loss.detach().cpu())
                if args.consistency_weight > 0 and 'consistency_original' in batch:
                    js_sum, paired_count = forward_consistency_loss(
                        model, batch, device, precision, args.attention_backend)
                    if not torch.isfinite(js_sum):
                        raise FloatingPointError('Non-finite consistency loss.')
                    with attention_context(device, args.attention_backend):
                        scaler.scale(args.consistency_weight * js_sum / denominator).backward()
                    consistency_sum += float(js_sum.detach().cpu())
                    consistency_documents += paired_count
                    del js_sum
                units += count
                documents += len(batch['input_ids'])
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                if not scaler.is_enabled():
                    raise FloatingPointError('Non-finite gradient norm in unscaled training.')
                scaler.update(new_scale=scaler.get_scale() / 2)
                updated = False
            else:
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                updated = scaler.get_scale() >= old_scale
            if updated:
                global_step += 1
                set_learning_rate(optimizer, epoch_rate, global_step, warmup_updates)
            else:
                skipped_updates += 1
            optimizer.zero_grad(set_to_none=True)
            if updated and (global_step == 1 or global_step % 25 == 0):
                progress = {'status': 'training', 'stage': args.stage, 'epoch': epoch, 'max_epochs': epochs,
                            'examples_seen_this_epoch': documents, 'training_rows': len(training),
                            'loss_units': 'next_tokens' if task == 'lm' else 'documents',
                            'training_loss': loss_sum / units, 'global_step': global_step,
                            'learning_rate': optimizer.param_groups[0]['lr'], 'amp_skipped_updates': skipped_updates,
                            'elapsed_seconds': time.perf_counter() - started}
                if task == 'classifier':
                    progress.update(supervised_bce=loss_sum / units,
                        consistency_js_sum=consistency_sum, consistency_documents=consistency_documents,
                        mean_js_changed_documents=consistency_sum / consistency_documents if consistency_documents else 0.0,
                        mean_js_all_documents=consistency_sum / documents,
                        total_objective=(loss_sum + args.consistency_weight * consistency_sum) / documents)
                atomic_json(out / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
        if documents != len(training):
            raise AssertionError('An epoch did not consume every training document exactly once.')
        row = {'epoch': epoch, 'training_loss': loss_sum / units,
               'loss_units': 'next_tokens' if task == 'lm' else 'documents',
               'training_target_tokens': units if task == 'lm' else None,
               'learning_rate': epoch_rate, 'optimizer_updates': global_step,
               'amp_skipped_updates': skipped_updates}
        if augmentation_report is not None:
            row['training_augmentation'] = augmentation_report
            row.update(supervised_bce=loss_sum / units,
                       consistency_js_sum=consistency_sum, consistency_documents=consistency_documents,
                       mean_js_changed_documents=consistency_sum / consistency_documents if consistency_documents else 0.0,
                       mean_js_all_documents=consistency_sum / documents,
                       total_objective=(loss_sum + args.consistency_weight * consistency_sum) / documents)
            if args.consistency_weight > 0 and consistency_documents != augmentation_report['changed']:
                raise AssertionError('Consistency did not consume every changed training document exactly once.')
        save_selected = False
        if task == 'lm':
            metrics = evaluate_lm(model, validation_loader, device, precision, args.attention_backend)
            save_selected = metrics['nll'] < best_nll
            if save_selected:
                best_nll, best_epoch = metrics['nll'], epoch
            row.update(validation=metrics, validation_nll=metrics['nll'], improved=save_selected,
                       best_epoch=best_epoch, next_learning_rate=epoch_rate)
        elif not schedule:
            probabilities, metrics = evaluate_classifier(model, validation_loader, device, precision,
                                                          args.attention_backend, validation.label.to_numpy())
            predictions = validation[['id', 'label']].copy()
            predictions['probability'] = probabilities
            predictions.to_csv(out / f'validation_epoch_{epoch}.csv', index=False)
            control = policy.observe(metrics['auc'], epoch)
            save_selected, best_epoch = control['improved'], policy.best_epoch
            row.update(validation=metrics, selection_score=metrics['auc'], validation_bce=metrics['bce'],
                       best_epoch=best_epoch, **control)
            if save_selected:
                predictions.to_csv(out / 'validation_predictions.csv', index=False)
        else:
            best_epoch = epochs
            row['next_learning_rate'] = schedule['learning_rates'][epoch] if epoch < epochs else None
        if save_selected:
            metadata = {'stage': args.stage, 'task': task, 'epoch': epoch, 'seed': args.seed,
                        'max_length': args.max_length, 'precision': precision,
                        'attention_backend': args.attention_backend, 'training_rows': len(training),
                        'validation_rows': len(validation), 'validation': metrics,
                        'selection_metric': 'minimum_token_weighted_nll' if task == 'lm' else 'maximum_auc',
                        'model_name_or_path': args.model_name_or_path,
                        'training_augmentation_config': augmentation_config,
                        'training_consistency_config': consistency_config}
            records = save_model(model, tokenizer, checkpoint, metadata)
            atomic_json(out / 'checkpoint_manifest.json', {'epoch': epoch, 'directory': 'best_model', 'files': records})
            print(json.dumps({'event': 'best_checkpoint_saved', 'stage': args.stage, 'epoch': epoch,
                              'validation': metrics}), flush=True)
        row['elapsed_seconds'] = time.perf_counter() - epoch_started
        history.append(row)
        atomic_json(out / 'history.json', history)
        print(json.dumps({'event': 'epoch_complete', 'stage': args.stage, **row}), flush=True)
        if task == 'classifier' and not schedule and control['stop']:
            stop_reason = 'validation_plateau_at_minimum_learning_rate'
            break
    completed = [row for row in history if row['epoch'] > 0]
    if schedule:
        metadata = {'stage': 'refit', 'task': 'classifier', 'epoch': epochs, 'seed': args.seed,
                    'max_length': args.max_length, 'precision': precision,
                    'attention_backend': args.attention_backend, 'training_rows': len(training),
                    'validation_rows': 0, 'fixed_schedule': schedule,
                    'model_name_or_path': args.model_name_or_path,
                    'training_augmentation_config': augmentation_config,
                    'training_consistency_config': consistency_config}
        records = save_model(model, tokenizer, checkpoint, metadata)
        atomic_json(out / 'checkpoint_manifest.json', {'epoch': epochs, 'directory': 'best_model', 'files': records})
        selected_schedule = schedule
    else:
        selected_schedule = {'epochs': best_epoch,
            'learning_rates': [row['learning_rate'] for row in completed[:best_epoch]],
            'warmup_epochs': args.warmup_epochs,
            'selected_by': 'minimum_lm_validation_nll' if task == 'lm' else 'maximum_inner_auc',
            'selection_history_sha256': sha256_file(out / 'history.json')}
    atomic_json(out / 'selected_schedule.json', selected_schedule)
    atomic_json(out / 'best_checkpoint.json', {'directory': 'best_model', 'epoch': best_epoch, 'max_length': args.max_length})
    result = {'status': 'complete', 'stage': args.stage, 'mode': args.stage, 'stop_reason': stop_reason,
              'model': args.model_name_or_path, 'seed': args.seed, 'completed_epochs': len(completed),
              'best_epoch': best_epoch, 'training_rows': len(training),
              'validation_rows': configuration['validation_rows'], 'checkpoint_directory': 'best_model',
              'precision': precision, 'attention_backend': args.attention_backend,
              'training_augmentation_config': augmentation_config,
              'training_consistency_config': consistency_config,
              'best_selection_score': policy.best_score if task == 'classifier' and not schedule else None,
              'best_validation_nll': best_nll if task == 'lm' else None,
              'baseline_validation_nll': baseline['nll'] if baseline else None,
              'adaptation_improves_baseline': best_nll < baseline['nll'] if baseline else None,
              'history': history, 'amp_skipped_updates': skipped_updates,
              'elapsed_seconds': time.perf_counter() - started}
    atomic_json(out / 'metrics.json', result)
    atomic_json(out / 'progress.json', result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    args = arguments()
    try:
        main(args)
    except Exception as exc:
        if getattr(args, '_run_owned', False):
            atomic_json(args.output_dir / 'failure.json', {'error': str(exc), 'traceback': traceback.format_exc()})
            atomic_json(args.output_dir / 'progress.json', {'status': 'failed', 'error': str(exc)})
        raise
