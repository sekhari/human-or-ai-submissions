"""Train a GPT-2 classifier and select a checkpoint by validation AUROC."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import pickle
import platform
import random
import shutil
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from common import (MODEL_NAME, atomic_json, autocast_context, choose_device, encode_frame,
                    file_sha256, load_data, make_loader, make_tokenizer,
                    predict_probabilities, save_model, set_seed, flush_file)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train_csv', type=Path, required=True)
    p.add_argument('--val_ids', type=Path, default=Path(__file__).parent / 'val_ids.csv')
    p.add_argument('--output_dir', type=Path, default=Path('results'))
    p.add_argument('--test_csv', type=Path)
    p.add_argument('--sample_submission', type=Path)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--max_length', type=int, default=512)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--eval_batch_size', type=int, default=4)
    p.add_argument('--gradient_accumulation', type=int, default=4)
    p.add_argument('--learning_rate', type=float, default=2e-5)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--warmup_ratio', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto', 'mps', 'cuda', 'cpu'], default='auto')
    p.add_argument('--save_every_steps', type=int, default=50)
    p.add_argument('--limit_train_rows', type=int, default=0)
    p.add_argument('--limit_val_rows', type=int, default=0)
    return p.parse_args()


def subset(frame, count, seed):
    if count == 0 or count >= len(frame):
        return frame
    idx, _ = train_test_split(np.arange(len(frame)), train_size=count,
                              random_state=seed, stratify=frame.label)
    return frame.iloc[np.sort(idx)].reset_index(drop=True)


def capture_rng(device):
    result = {'python': random.getstate(), 'numpy': np.random.get_state(),
              'torch': torch.get_rng_state()}
    if device.type == 'mps':
        result['device'] = torch.mps.get_rng_state()
    elif device.type == 'cuda':
        result['device'] = torch.cuda.get_rng_state_all()
    return result


def restore_rng(value, device):
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch'])
    if device.type == 'mps':
        torch.mps.set_rng_state(value['device'])
    elif device.type == 'cuda':
        torch.cuda.set_rng_state_all(value['device'])


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def resume_records(directory):
    """Read committed checkpoint records, including the backup manifest."""
    records = []
    errors = []
    for name in ('manifest.json', 'manifest.backup.json'):
        path = directory / name
        if not path.exists():
            continue
        try:
            manifest = json.loads(path.read_text())
            if manifest['schema_version'] != 1:
                raise ValueError('Unsupported checkpoint manifest version.')
            for record in manifest['checkpoints']:
                if Path(record['file']).name != record['file']:
                    raise ValueError('Checkpoint names must not contain a directory.')
                if record['file'] not in {r['file'] for r in records}:
                    records.append(record)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f'{name}: {exc}')
    return records, errors


def valid_record(directory, record):
    path = directory / record['file']
    return (path.is_file() and path.stat().st_size == record['bytes']
            and file_sha256(path) == record['sha256'])


def load_resume(directory, signature):
    if not directory.exists():
        return None
    records, errors = resume_records(directory)
    for index, record in enumerate(records):
        try:
            if not valid_record(directory, record):
                raise ValueError('Size or SHA-256 verification failed.')
            state = torch.load(directory / record['file'], map_location='cpu', weights_only=False)
            if state['signature'] != signature:
                raise RuntimeError('The checkpoint belongs to a different training configuration.')
            if index or errors:
                print(json.dumps({'event':'checkpoint_recovery','file':record['file'],
                                  'earlier_checkpoint_errors':errors}), flush=True)
            return state
        except RuntimeError as exc:
            if 'different training configuration' in str(exc):
                raise
            errors.append(f"{record['file']}: {exc}")
        except (OSError, ValueError, KeyError, EOFError, pickle.UnpicklingError) as exc:
            errors.append(f"{record['file']}: {exc}")
    if any(directory.iterdir()):
        raise RuntimeError('No valid committed training checkpoint was found. ' + '; '.join(errors))
    return None


def save_resume(directory, value):
    """Commit a verified checkpoint before replacing either manifest."""
    directory.mkdir(parents=True, exist_ok=True)
    old_records, _ = resume_records(directory)
    previous = next((r for r in old_records if valid_record(directory, r)), None)
    filename = f"step_{value['global_step']:06d}_{uuid.uuid4().hex[:12]}.pt"
    destination = directory / filename
    temporary = directory / (filename + '.tmp')
    with tempfile.TemporaryDirectory(prefix='gpt2-checkpoint-') as scratch:
        staged = Path(scratch) / filename
        torch.save(cpu_tree(value), staged)
        with staged.open('rb') as stream:
            os.fsync(stream.fileno())
        expected_hash = file_sha256(staged)
        expected_bytes = staged.stat().st_size
        with staged.open('rb') as source, temporary.open('wb') as target:
            shutil.copyfileobj(source, target, length=4*1024*1024)
            flush_file(target)
        if temporary.stat().st_size != expected_bytes or file_sha256(temporary) != expected_hash:
            raise IOError('The written checkpoint failed size or SHA-256 verification.')
        os.replace(temporary, destination)
    record = {'file':filename,'bytes':expected_bytes,'sha256':expected_hash,
              'epoch':value['epoch'],'next_batch':value['next_batch'],
              'global_step':value['global_step']}
    manifest = {'schema_version':1,'checkpoints':[record] + ([previous] if previous else [])}
    atomic_json(directory / 'manifest.json', manifest)
    if json.loads((directory / 'manifest.json').read_text()) != manifest:
        raise IOError('Checkpoint manifest verification failed.')
    atomic_json(directory / 'manifest.backup.json', manifest)
    keep = {r['file'] for r in manifest['checkpoints']}
    for old in directory.glob('step_*.pt'):
        if old.name not in keep:
            old.unlink()
    print(json.dumps({'event':'checkpoint_saved','path':str(destination),**record}), flush=True)


def main(args):
    for key in ('epochs', 'max_length', 'batch_size', 'eval_batch_size',
                'gradient_accumulation', 'save_every_steps'):
        if getattr(args, key) < 1:
            raise ValueError(f'{key} must be positive.')
    if args.max_length > 1024 or not 0 <= args.warmup_ratio < 1:
        raise ValueError('Invalid sequence length or warmup ratio.')
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError('Invalid optimizer settings.')
    if args.limit_train_rows < 0 or args.limit_val_rows < 0:
        raise ValueError('Row limits must be nonnegative.')
    if args.sample_submission and not args.test_csv:
        raise ValueError('sample_submission requires test_csv.')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock_name = hashlib.sha256(str(out).encode()).hexdigest()
    lock = (Path(tempfile.gettempdir()) / f'gpt2-{lock_name}.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('A training process already owns this output directory.')
    device = choose_device(args.device)
    if device.type == 'cuda':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    set_seed(args.seed)
    amp_enabled = device.type == 'cuda'
    train, val = load_data(args.train_csv, args.val_ids)
    train = subset(train, args.limit_train_rows, args.seed)
    val = subset(val, args.limit_val_rows, args.seed)
    if set(train.label.unique()) != {0, 1} or set(val.label.unique()) != {0, 1}:
        raise ValueError('Both partitions must contain both classes.')
    source_hashes = {p.name: file_sha256(p) for p in Path(__file__).parent.glob('*.py')}
    configuration = {k: str(v.resolve()) if isinstance(v, Path) else v
                     for k, v in vars(args).items() if not k.startswith('_')}
    configuration.update(model=MODEL_NAME, device=str(device), precision='fp16_autocast' if amp_enabled else 'float32',
                         training_rows=len(train), validation_rows=len(val), source_hashes=source_hashes,
                         train_sha256=file_sha256(args.train_csv), val_ids_sha256=file_sha256(args.val_ids),
                         torch=torch.__version__, transformers=transformers.__version__)
    for key in ('test_csv', 'sample_submission'):
        if getattr(args, key):
            configuration[key + '_sha256'] = file_sha256(getattr(args, key))
    signature = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    config_path = out / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError('The output directory contains a different training configuration.')
    atomic_json(config_path, configuration)
    if (out / 'metrics.json').exists():
        completed = json.loads((out / 'metrics.json').read_text())
        best_path = out / completed['checkpoint_directory']
        required = [out / 'validation_predictions.csv', out / 'best_checkpoint.json',
                    best_path / 'config.json', best_path / 'model.safetensors', best_path / 'tokenizer.json']
        if args.test_csv:
            required.append(out / 'submission.csv')
        if completed.get('status') != 'complete' or any(not p.is_file() for p in required):
            raise RuntimeError('Completed-run metadata exists, but required result files are missing.')
        print('Training is already complete.', flush=True)
        return
    args._run_owned = True
    (out / 'failure.json').unlink(missing_ok=True)
    train[['id']].to_csv(out / 'train_ids.csv', index=False)
    val[['id']].to_csv(out / 'val_ids.csv', index=False)
    environment = {'python':sys.version, 'platform':platform.platform(),
        'torch':torch.__version__, 'transformers':transformers.__version__, 'device':str(device),
        'cuda_version':torch.version.cuda, 'precision':configuration['precision'],
        'gpu':torch.cuda.get_device_name(0) if amp_enabled else None,
        'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory if amp_enabled else None}
    atomic_json(out / 'environment.json', environment)
    print(json.dumps({'event':'setup','model':MODEL_NAME,'device':str(device),
                     'training_rows':len(train),'validation_rows':len(val),'epochs':args.epochs}), flush=True)
    tokenizer = make_tokenizer(MODEL_NAME)
    train_data = encode_frame(train, tokenizer, args.max_length)
    val_loader = make_loader(encode_frame(val, tokenizer, args.max_length), tokenizer,
                             args.eval_batch_size)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1,
                                                               use_safetensors=True)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.float().to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    batches_per_epoch = math.ceil(len(train) / args.batch_size)
    updates_per_epoch = math.ceil(batches_per_epoch / args.gradient_accumulation)
    scheduler = get_linear_schedule_with_warmup(optimizer,
        num_warmup_steps=int(updates_per_epoch * args.epochs * args.warmup_ratio),
        num_training_steps=updates_per_epoch * args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    resume_path = out / 'resume'
    epoch_start, next_batch, global_step, seen, loss_sum = 0, 0, 0, 0, 0.0
    skipped_updates = 0
    history = []
    best_auc, best_epoch = -1.0, 0
    state = load_resume(resume_path, signature)
    if state is not None:
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        epoch_start, next_batch = state['epoch'], state['next_batch']
        global_step, seen, loss_sum = state['global_step'], state['seen'], state['loss_sum']
        history, best_auc, best_epoch = state['history'], state['best_auc'], state['best_epoch']
        skipped_updates = state['skipped_updates']
        restore_rng(state['rng'], device)
        del state
        print(json.dumps({'event':'resume','epoch':epoch_start+1,'next_batch':next_batch}), flush=True)

    def checkpoint(epoch, cursor):
        save_resume(resume_path, {'signature':signature,'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
            'epoch':epoch,'next_batch':cursor,'global_step':global_step,'seen':seen,
            'loss_sum':loss_sum,'history':history,'best_auc':best_auc,'best_epoch':best_epoch,
            'skipped_updates':skipped_updates,'rng':capture_rng(device)})

    started = time.perf_counter()
    for epoch in range(epoch_start, args.epochs):
        model.train()
        loader = make_loader(train_data, tokenizer, args.batch_size, shuffle=True, seed=args.seed+epoch)
        optimizer.zero_grad(set_to_none=True)
        epoch_started = time.perf_counter()
        for batch_index, batch in enumerate(loader):
            if batch_index < next_batch:
                continue
            labels = batch.pop('labels').to(device=device, dtype=torch.float32)
            inputs = {k:v.to(device) for k,v in batch.items()}
            with autocast_context(device):
                logits = model(**inputs).logits.squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='sum')
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss.')
            group_start = (batch_index // args.gradient_accumulation) * args.gradient_accumulation
            group_examples = min(len(train), (group_start+args.gradient_accumulation)*args.batch_size) - group_start*args.batch_size
            scaler.scale(loss / group_examples).backward()
            loss_sum += float(loss.detach().cpu())
            seen += len(labels)
            if (batch_index+1) % args.gradient_accumulation == 0 or batch_index+1 == len(loader):
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(norm):
                    if not amp_enabled:
                        raise FloatingPointError('Non-finite gradient norm.')
                    scaler.update(new_scale=scaler.get_scale()/2)
                    updated = False
                else:
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    updated = scaler.get_scale() >= old_scale
                if updated:
                    scheduler.step()
                    global_step += 1
                else:
                    skipped_updates += 1
                    print(json.dumps({'event':'amp_update_skipped','epoch':epoch+1,
                        'batch':batch_index+1,'scale':scaler.get_scale()}), flush=True)
                optimizer.zero_grad(set_to_none=True)
                progress = {'status':'training','epoch':epoch+1,'epochs':args.epochs,
                    'examples_seen_this_epoch':seen,'training_rows':len(train),
                    'global_step':global_step,'training_loss':loss_sum/seen,
                    'elapsed_seconds_this_process':time.perf_counter()-started,
                    'best_validation_auc':None if best_epoch==0 else best_auc,
                    'amp_skipped_updates':skipped_updates}
                atomic_json(out / 'progress.json', progress)
                if updated and (global_step == 1 or global_step % 10 == 0):
                    print(json.dumps(progress), flush=True)
                if updated and (global_step == 1 or global_step % args.save_every_steps == 0):
                    checkpoint(epoch, batch_index+1)
        next_batch = 0
        atomic_json(out / 'progress.json', {'status':'evaluating','epoch':epoch+1,'epochs':args.epochs})
        probabilities = predict_probabilities(model, val_loader, device)
        auc = float(roc_auc_score(val.label, probabilities))
        predictions = val[['id','label']].copy()
        predictions['probability'] = probabilities
        predictions.to_csv(out / f'validation_epoch_{epoch+1}.csv', index=False)
        row = {'epoch':epoch+1,'training_loss':loss_sum/seen,'validation_auc':auc,
               'elapsed_seconds_this_attempt':time.perf_counter()-epoch_started,'amp_skipped_updates':skipped_updates}
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch+1
            destination = out / f'checkpoint_epoch_{epoch+1}'
            save_model(model, tokenizer, destination)
            atomic_json(destination / 'training_metadata.json', {'epoch':best_epoch,'validation_auc':best_auc,
                         'max_length':args.max_length,'positive_label':1})
        history.append(row)
        print(json.dumps({'event':'epoch_complete',**row}), flush=True)
        loss_sum, seen = 0.0, 0
        checkpoint(epoch+1, 0)
        atomic_json(out / 'history.json', history)

    best_path = out / f'checkpoint_epoch_{best_epoch}'
    atomic_json(out / 'best_checkpoint.json', {'directory':best_path.name,'epoch':best_epoch,
                'validation_auc':best_auc,'max_length':args.max_length})
    shutil.copyfile(out / f'validation_epoch_{best_epoch}.csv', out / 'validation_predictions.csv')
    if args.test_csv:
        del optimizer, scheduler, model
        import gc
        gc.collect()
        if device.type == 'mps':
            torch.mps.empty_cache()
        elif device.type == 'cuda':
            torch.cuda.empty_cache()
        from predict import run_prediction
        atomic_json(out / 'progress.json', {'status':'predicting','epochs':args.epochs,'best_epoch':best_epoch})
        run_prediction(best_path, args.test_csv, out / 'submission.csv',
                       args.sample_submission, str(device), args.eval_batch_size, args.max_length)
    result = {'status':'complete','model':MODEL_NAME,'epochs':args.epochs,'best_epoch':best_epoch,
              'best_validation_auc':best_auc,'training_rows':len(train),'validation_rows':len(val),
              'checkpoint':str(best_path),'checkpoint_directory':best_path.name,'history':history,
              'amp_skipped_updates':skipped_updates}
    atomic_json(out / 'metrics.json', result)
    atomic_json(out / 'progress.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    args = arguments()
    try:
        main(args)
    except Exception as exc:
        if getattr(args, '_run_owned', False):
            atomic_json(args.output_dir / 'failure.json', {'error':str(exc),'traceback':traceback.format_exc()})
            atomic_json(args.output_dir / 'progress.json', {'status':'failed','error':str(exc)})
        raise
