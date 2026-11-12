"""Evaluate a trained GPT-2 classifier under whitespace transformations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    project, work = args.project, args.work
    work.mkdir(parents=True, exist_ok=False)
    local_input = work / 'inputs'
    local_input.mkdir()
    results = project / 'results'
    metadata_files = ['metrics.json', 'best_checkpoint.json', 'config.json',
                      'val_ids.csv', 'validation_predictions.csv']
    for name in metadata_files:
        shutil.copyfile(results / name, local_input / name)
    shutil.copyfile(args.data / 'train.csv', local_input / 'train.csv')
    shutil.copyfile(project / 'code/common.py', local_input / 'common.py')
    metrics = json.loads((local_input / 'metrics.json').read_text())
    reference = json.loads((local_input / 'best_checkpoint.json').read_text())
    config = json.loads((local_input / 'config.json').read_text())
    assert metrics['status'] == 'complete'
    assert reference['epoch'] == metrics['best_epoch'] == 3
    name = reference['directory']
    assert Path(name).name == name
    assert name == metrics['checkpoint_directory']
    checkpoint = local_input / name
    checkpoint.mkdir()
    print(json.dumps({'event': 'copy_model', 'selected_epoch': reference['epoch']}), flush=True)
    for source in (results / name).iterdir():
        if source.is_file():
            shutil.copyfile(source, checkpoint / source.name)
    model_metadata = json.loads((checkpoint / 'training_metadata.json').read_text())
    assert model_metadata['epoch'] == 3 and model_metadata['max_length'] == 512
    assert model_metadata['positive_label'] == 1
    assert sha256(local_input / 'train.csv') == config['train_sha256']
    assert sha256(local_input / 'common.py') == config['source_hashes']['common.py']
    sys.path.insert(0, str(local_input))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    import numpy as np
    import pandas as pd
    import torch
    import transformers
    from sklearn.metrics import roc_auc_score
    from transformers import AutoModelForSequenceClassification
    from common import (autocast_context, encode_frame, load_data, make_loader,
                        make_tokenizer, set_seed)

    assert torch.cuda.is_available(), 'A CUDA GPU is required for this evaluation.'
    set_seed(config['seed'])
    device = torch.device('cuda')
    _, validation = load_data(local_input / 'train.csv', local_input / 'val_ids.csv')
    assert len(validation) == 2000 and validation['id'].nunique() == 2000
    assert validation['label'].value_counts().to_dict() == {0: 1000, 1: 1000}
    saved = pd.read_csv(local_input / 'validation_predictions.csv', dtype={'id': str})
    assert saved['id'].tolist() == validation['id'].tolist()
    assert saved['label'].tolist() == validation['label'].tolist()
    max_length = config['max_length']
    batch_size = config['eval_batch_size']
    assert max_length == reference['max_length'] == 512 and batch_size == 4
    tokenizer = make_tokenizer(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, local_files_only=True, use_safetensors=True)
    assert model.config.model_type == 'gpt2' and model.config.num_labels == 1
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.float().to(device).eval()
    original_text = validation['text'].tolist()
    labels = validation['label'].to_numpy()
    transforms = {
        'original': lambda text: text,
        'spaces_only': lambda text: re.sub(r'[ \t]+', ' ', text),
        'newlines_only': lambda text: re.sub(r'[\r\n]+', ' ', text),
        'all_whitespace': lambda text: re.sub(r'\s+', ' ', text).strip(),
    }
    definitions = {
        'original': 'Unchanged validation text.',
        'spaces_only': 'Collapse each run of ASCII spaces and tabs to one space; retain line breaks.',
        'newlines_only': 'Replace each run of CR/LF line-break characters with one space; retain other whitespace.',
        'all_whitespace': 'Collapse every run of Unicode whitespace to one ASCII space and trim the edges.',
    }
    predictions, lengths, changed, token_changed = {}, {}, {}, {}
    records = []
    original_tokens = None
    consistency = None
    started = time.perf_counter()
    for variant, transform in transforms.items():
        texts = [transform(text) for text in original_text]
        assert all(re.sub(r'\s+', '', before) == re.sub(r'\s+', '', after)
                   for before, after in zip(original_text, texts))
        changed[variant] = np.array([a != b for a, b in zip(original_text, texts)])
        frame = validation.copy()
        frame['text'] = texts
        full_lengths = []
        for start in range(0, len(texts), 256):
            full = tokenizer(texts[start:start + 256], truncation=False, padding=False,
                             return_attention_mask=False, return_token_type_ids=False,
                             verbose=False)
            full_lengths.extend(map(len, full['input_ids']))
        lengths[variant] = np.array(full_lengths)
        dataset = encode_frame(frame, tokenizer, max_length)
        tokens = [row['input_ids'] for row in dataset.rows]
        if original_tokens is None:
            original_tokens = tokens
        token_changed[variant] = np.array([a != b for a, b in zip(original_tokens, tokens)])
        loader = make_loader(dataset, tokenizer, batch_size)
        probabilities = []
        phase_started = time.perf_counter()
        print(json.dumps({'event': 'variant_start', 'variant': variant,
                          'changed_texts': int(changed[variant].sum())}), flush=True)
        with torch.inference_mode():
            for index, batch in enumerate(loader):
                inputs = {key: value.to(device) for key, value in batch.items() if key != 'labels'}
                with autocast_context(device):
                    logits = model(**inputs).logits.squeeze(-1)
                probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
                if (index + 1) % 125 == 0:
                    print(json.dumps({'event': 'prediction_progress', 'variant': variant,
                                      'rows': min((index + 1) * batch_size, len(frame))}), flush=True)
        probability = np.concatenate(probabilities).astype(np.float64)
        assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
        predictions[variant] = probability
        auc = float(roc_auc_score(labels, probability))
        if variant == 'original':
            saved_auc = float(roc_auc_score(labels, saved['probability']))
            consistency = {
                'saved_auc': saved_auc,
                'rerun_auc': auc,
                'auc_difference': auc - saved_auc,
                'max_probability_difference': float(np.abs(probability - saved['probability'].to_numpy()).max()),
            }
            assert abs(saved_auc - metrics['best_validation_auc']) < 1e-10
            assert abs(consistency['auc_difference']) <= 1e-5, consistency
            assert consistency['max_probability_difference'] <= 0.002, consistency
            print(json.dumps({'event': 'original_consistency', **consistency}), flush=True)
        delta = probability - predictions['original']
        record = {
            'variant': variant, 'rows': len(frame), 'auc': auc,
            'auc_delta_vs_original': auc - float(roc_auc_score(labels, predictions['original'])),
            'changed_texts': int(changed[variant].sum()),
            'changed_human_texts': int(changed[variant][labels == 0].sum()),
            'changed_ai_texts': int(changed[variant][labels == 1].sum()),
            'changed_token_sequences': int(token_changed[variant].sum()),
            'truncated_texts': int((lengths[variant] > max_length).sum()),
            'mean_absolute_probability_change': float(np.abs(delta).mean()),
            'max_absolute_probability_change': float(np.abs(delta).max()),
            'mean_probability_change_human': float(delta[labels == 0].mean()),
            'mean_probability_change_ai': float(delta[labels == 1].mean()),
            'label_flips_at_0_5': int(((probability >= 0.5) != (predictions['original'] >= 0.5)).sum()),
            'inference_seconds': time.perf_counter() - phase_started,
        }
        records.append(record)
        print(json.dumps({'event': 'variant_complete', **record}), flush=True)

    common_untruncated = np.logical_and.reduce([value <= max_length for value in lengths.values()])
    assert len(np.unique(labels[common_untruncated])) == 2
    common_original_auc = float(roc_auc_score(labels[common_untruncated], predictions['original'][common_untruncated]))
    for record in records:
        variant = record['variant']
        record['common_untruncated_rows'] = int(common_untruncated.sum())
        record['common_untruncated_auc'] = float(roc_auc_score(labels[common_untruncated], predictions[variant][common_untruncated]))
        record['common_untruncated_auc_delta'] = record['common_untruncated_auc'] - common_original_auc
    paired = validation[['id', 'label']].copy()
    paired['common_untruncated'] = common_untruncated
    for variant in transforms:
        paired[variant + '_probability'] = predictions[variant]
        paired[variant + '_full_tokens'] = lengths[variant]
        paired[variant + '_text_changed'] = changed[variant]
        paired[variant + '_tokens_changed'] = token_changed[variant]
    output = work / 'results'
    output.mkdir()
    paired.to_csv(output / 'paired_predictions.csv', index=False)
    pd.DataFrame(records).to_csv(output / 'metrics.csv', index=False)
    report = {
        'status': 'complete', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'model': metrics['model'], 'selected_epoch': reference['epoch'],
        'validation_rows': len(validation), 'positive_label': 1,
        'max_length': max_length, 'batch_size': batch_size, 'precision': 'fp16_autocast',
        'model_sha256': sha256(checkpoint / 'model.safetensors'),
        'train_sha256': sha256(local_input / 'train.csv'),
        'validation_ids_sha256': sha256(local_input / 'val_ids.csv'),
        'saved_predictions_sha256': sha256(local_input / 'validation_predictions.csv'),
        'common_source_sha256': sha256(local_input / 'common.py'),
        'script_sha256': sha256(__file__),
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'gpu': torch.cuda.get_device_name(0), 'definitions': definitions,
        'original_consistency': consistency, 'metrics': records,
        'elapsed_seconds': time.perf_counter() - started,
        'limitations': [
            'This is a paired sensitivity diagnostic on an already-used validation set, not an independent generalization estimate.',
            'Whitespace transformations preserve non-whitespace characters but can alter paragraph structure and tokenization.',
            'Full-set changes include altered token truncation; the common untruncated subset removes that source of variation.',
            'No test-set labels are available, so this diagnostic cannot establish the cause of the public-leaderboard gap.',
        ],
    }
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    shutil.copyfile(__file__, output / 'format_sensitivity.py')
    readme = '# Formatting sensitivity\n\nInference-only paired evaluation using the selected epoch-3 GPT-2 model and the same 2,000 validation IDs.\n\n'
    readme += '\n'.join(f'- **{name}**: {definition}' for name, definition in definitions.items())
    readme += '\n\nAll transformations preserve the sequence of non-whitespace characters. See summary.json for provenance, checks, metrics and limitations; paired_predictions.csv contains per-ID probabilities and token lengths. No model weights or training checkpoints are included.\n'
    (output / 'README.md').write_text(readme)
    archive_path = Path(shutil.make_archive(str(work / 'format_sensitivity_results'), 'zip', output))
    destination = project / 'format_sensitivity' / work.name
    destination.mkdir(parents=True, exist_ok=False)
    for source in output.iterdir():
        shutil.copyfile(source, destination / source.name)
    shutil.copyfile(archive_path, destination / archive_path.name)
    print(pd.DataFrame(records)[['variant', 'auc', 'auc_delta_vs_original', 'changed_texts',
                               'common_untruncated_auc', 'common_untruncated_auc_delta']].to_string(index=False), flush=True)
    print(json.dumps({'event': 'complete', 'output_directory': str(destination),
                      'local_archive': str(archive_path), 'summary': report}), flush=True)


if __name__ == '__main__':
    main()
