"""Raw-text data, deterministic attention, precision, and checkpoint utilities."""
from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

MODEL_NAME = 'openai-community/gpt2-large'
RAW_TEXT_CONFIG = {
    'text_transform': 'identity', 'newline_normalization': False,
    'encoding_repair': False, 'whitespace_collapse': False,
    'format_augmentation': False, 'truncation_side': 'right',
    'lm_append_eos': True, 'lm_padding_labels': 'attention_mask_zero_only',
}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_frame(path, labeled=False):
    csv.field_size_limit(max(csv.field_size_limit(), 16 * 1024 * 1024))
    # newline='' preserves CRLF and CR inside quoted fields exactly as supplied.
    with Path(path).open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'id', 'text'} | ({'label'} if labeled else set())
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Missing required columns in {path}: {sorted(required)}')
        records = list(reader)
    if not records:
        raise ValueError(f'No records found in {path}.')
    frame = pd.DataFrame(records)
    if frame.id.map(lambda x: not isinstance(x, str) or not x.strip()).any() or frame.id.duplicated().any():
        raise ValueError('IDs must be nonempty and unique.')
    if not frame.text.map(lambda x: isinstance(x, str) and len(x) > 0).all():
        raise ValueError('Every text field must be a nonempty string.')
    if labeled:
        values = pd.to_numeric(frame.label, errors='coerce')
        if values.isna().any() or not values.isin([0, 1]).all():
            raise ValueError('Classification labels must be 0 or 1.')
        frame['label'] = values.astype(np.int64)
    return frame


def read_ids(path):
    with Path(path).open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if 'id' not in (reader.fieldnames or []):
            raise ValueError('The validation-ID file requires an id column.')
        ids = [row['id'] for row in reader]
    if not ids or len(set(ids)) != len(ids) or any(not value.strip() for value in ids):
        raise ValueError('Validation IDs must be nonempty and unique.')
    return ids


def split_frame(path, validation_ids, labeled=False):
    frame = read_frame(path, labeled=labeled)
    ids = read_ids(validation_ids)
    missing = set(ids) - set(frame.id)
    if missing:
        raise ValueError(f'Validation IDs not found in source CSV: {sorted(missing)[:5]}')
    validation = frame.set_index('id').loc[ids].reset_index()
    training = frame.loc[~frame.id.isin(ids)].reset_index(drop=True)
    if training.empty:
        raise ValueError('The training partition is empty.')
    if labeled and (set(training.label) != {0, 1} or set(validation.label) != {0, 1}):
        raise ValueError('Both classification partitions require both label classes.')
    return training, validation


def choose_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(name)
    if device.type not in {'cpu', 'cuda'}:
        raise ValueError('Supported devices are cpu and cuda.')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU was requested but is unavailable.')
    return device


def choose_precision(requested, device):
    if device.type != 'cuda':
        if requested not in {'auto', 'fp32'}:
            raise ValueError('CPU execution requires auto or fp32 precision.')
        return 'fp32'
    if requested == 'auto':
        return 'bf16' if torch.cuda.is_bf16_supported() else 'fp16'
    if requested == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('This GPU/runtime does not support bfloat16.')
    return requested


def set_seed(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def autocast_context(device, precision):
    if device.type == 'cuda' and precision != 'fp32':
        dtype = torch.bfloat16 if precision == 'bf16' else torch.float16
        return torch.autocast('cuda', dtype=dtype)
    return contextlib.nullcontext()


def attention_context(device, backend):
    if device.type != 'cuda':
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    choices = {'flash': [SDPBackend.FLASH_ATTENTION], 'math': [SDPBackend.MATH],
               'flash_or_math': [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]}
    # Restrict execution to the deterministic flash or math attention paths.
    return sdpa_kernel(choices[backend])


def make_tokenizer(source):
    tokenizer = AutoTokenizer.from_pretrained(str(source), use_fast=True)
    if tokenizer.eos_token_id is None:
        raise ValueError('GPT-2 requires an end-of-sequence token.')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'right'
    return tokenizer


class EncodedRows(Dataset):
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        return dict(self.rows[index])


def encode_frame(frame, tokenizer, max_length, task):
    if not 2 <= max_length <= 1024:
        raise ValueError('GPT-2 max_length must be between 2 and 1024.')
    rows = []
    texts = frame.text.tolist()
    for start in range(0, len(texts), 256):
        encoded = tokenizer(texts[start:start + 256], truncation=True,
                            max_length=max_length - 1 if task == 'lm' else max_length,
                            padding=False, return_attention_mask=True,
                            return_token_type_ids=False, add_special_tokens=False)
        for offset, tokens in enumerate(encoded['input_ids']):
            if task == 'lm':
                tokens = tokens + [tokenizer.eos_token_id]
            if not tokens or (task == 'lm' and len(tokens) < 2):
                raise ValueError('A document produced too few tokens for this training objective.')
            row = {'input_ids': tokens, 'attention_mask': [1] * len(tokens)}
            if task == 'classifier' and 'label' in frame:
                row['labels'] = float(frame.iloc[start + offset].label)
            rows.append(row)
    return EncodedRows(rows)


class RawCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, examples):
        labels = [example['labels'] for example in examples] if 'labels' in examples[0] else None
        features = [{k: v for k, v in example.items() if k != 'labels'} for example in examples]
        batch = self.tokenizer.pad(features, padding=True, return_tensors='pt')
        if labels is not None:
            batch['labels'] = torch.tensor(labels, dtype=torch.float32)
        return batch


def make_loader(dataset, tokenizer, batch_size, shuffle=False, seed=42):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(seed), collate_fn=RawCollator(tokenizer))


def lm_labels(input_ids, attention_mask):
    # EOS is also the padding token ID; mask by attention, never by token ID.
    return input_ids.masked_fill(attention_mask == 0, -100)


def lm_loss_sum(logits, input_ids, attention_mask):
    labels = lm_labels(input_ids, attention_mask)[:, 1:].contiguous()
    count = int((labels != -100).sum().item())
    if count == 0:
        raise ValueError('No next-token targets remain after attention masking.')
    loss = F.cross_entropy(logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
                           labels.reshape(-1), ignore_index=-100, reduction='sum')
    return loss, count


def batch_weight(batch, task):
    if task == 'lm':
        return int(batch['attention_mask'][:, 1:].sum().item())
    return len(batch['labels'])


def forward_loss(model, batch, task, device, precision, backend):
    inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
    with attention_context(device, backend), autocast_context(device, precision):
        logits = model(**inputs).logits
    if task == 'lm':
        return lm_loss_sum(logits, inputs['input_ids'], inputs['attention_mask'])
    targets = batch['labels'].to(device=device, dtype=torch.float32)
    return F.binary_cross_entropy_with_logits(logits.squeeze(-1).float(), targets, reduction='sum'), len(targets)


def load_model(source, task, tokenizer, device, checkpointing=False, require_adapted=False):
    source = str(source)
    metadata = None
    if require_adapted:
        metadata_path = Path(source) / 'training_metadata.json'
        if not metadata_path.is_file():
            raise ValueError('Classifier training requires the local adapted LM checkpoint.')
        metadata = json.loads(metadata_path.read_text())
        if metadata.get('stage') != 'lm' or metadata.get('raw_text_config') != RAW_TEXT_CONFIG:
            raise ValueError('The source must be a raw-text adapted LM checkpoint.')
    factory = AutoModelForCausalLM if task == 'lm' else AutoModelForSequenceClassification
    options = {'attn_implementation': 'sdpa', 'use_safetensors': True, 'output_loading_info': True}
    if task == 'classifier':
        options['num_labels'] = 1
    model, loading = factory.from_pretrained(source, **options)
    if model.config.model_type != 'gpt2':
        raise ValueError('This pipeline supports GPT-2 checkpoints only.')
    allowed_missing = {'score.weight'} if task == 'classifier' else set()
    allowed_unexpected = {'lm_head.weight'} if task == 'classifier' else set()
    missing = set(loading.get('missing_keys', []))
    unexpected = set(loading.get('unexpected_keys', []))
    if (missing - allowed_missing or unexpected - allowed_unexpected
            or loading.get('mismatched_keys') or loading.get('error_msgs')):
        raise RuntimeError(f'Checkpoint loading failed strict backbone verification: {loading}')
    if require_adapted and missing != {'score.weight'}:
        raise RuntimeError('Expected a new classifier head and a completely loaded adapted backbone.')
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.float().to(device)
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    else:
        model.gradient_checkpointing_disable()
    report = {'source': source, 'task': task, 'missing_keys': sorted(missing),
              'unexpected_keys': sorted(unexpected), 'strict_backbone_verified': True,
              'adapted_lm_metadata': metadata, 'parameter_count': sum(p.numel() for p in model.parameters()),
              'attention_implementation': 'sdpa', 'master_parameter_dtype': 'float32'}
    return model, report


def set_learning_rate(optimizer, target, completed_updates, warmup_updates):
    rate = target * min(1.0, (completed_updates + 1) / max(1, warmup_updates))
    for group in optimizer.param_groups:
        group['lr'] = rate
    return rate


def save_model(model, tokenizer, path, metadata):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    previous = path.with_name(path.name + '.previous')
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    # Keep tied LM weights tied after the CPU copy; do not export optimizer state.
    state, storage_copies = {}, {}
    for name, tensor in model.state_dict().items():
        key = (tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()))
        if key not in storage_copies:
            storage_copies[key] = tensor.detach().cpu().contiguous()
        state[name] = storage_copies[key]
    model.save_pretrained(temporary, state_dict=state, safe_serialization=True, max_shard_size='10GB')
    tokenizer.save_pretrained(temporary)
    atomic_json(temporary / 'raw_text_config.json', RAW_TEXT_CONFIG)
    atomic_json(temporary / 'training_metadata.json', {**metadata, 'raw_text_config': RAW_TEXT_CONFIG})
    source = Path(__file__).resolve().parent
    for name in ('training_utils.py', 'train_stage.py', 'requirements.txt'):
        if (source / name).is_file():
            shutil.copy2(source / name, temporary / name)
    if previous.exists():
        shutil.rmtree(previous)
    if path.exists():
        path.rename(previous)
    try:
        temporary.rename(path)
    except BaseException:
        if previous.exists() and not path.exists():
            previous.rename(path)
        raise
    if previous.exists():
        shutil.rmtree(previous)
    return {p.name: {'bytes': p.stat().st_size, 'sha256': sha256_file(p)}
            for p in sorted(path.iterdir()) if p.is_file()}
