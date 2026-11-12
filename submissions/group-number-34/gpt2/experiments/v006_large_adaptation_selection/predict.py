"""Predict from the final adapted GPT-2 classifier without modifying input text."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path


def read_rows(path):
    with Path(path).open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def run_prediction(checkpoint, predict_csv, output, sample_submission=None,
                   device='cuda', precision='auto', attention_backend='flash_or_math',
                   batch_size=8, max_length=512):
    """Delegate raw-text model inference and verify the resulting IDs and probabilities."""
    checkpoint, predict_csv, output = map(Path, (checkpoint, predict_csv, output))
    columns, inputs = read_rows(predict_csv)
    if not columns or not {'id', 'text'}.issubset(columns) or not inputs:
        raise ValueError('The input CSV must contain nonempty id,text rows.')
    expected_ids = [row['id'] for row in inputs]
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError('Prediction input IDs must be unique.')
    if sample_submission:
        sample_columns, sample = read_rows(sample_submission)
        if sample_columns != ['id', 'label']:
            raise ValueError('The submission template must contain exactly id,label.')
        template_ids = [row['id'] for row in sample]
        if len(set(template_ids)) != len(template_ids) or set(template_ids) != set(expected_ids):
            raise ValueError('Template IDs must match prediction input IDs exactly.')
        expected_ids = template_ids
    command = [sys.executable, '-u', str(Path(__file__).with_name('train_stage.py')),
               '--stage', 'predict', '--checkpoint', str(checkpoint),
               '--predict_csv', str(predict_csv), '--prediction_output', str(output),
               '--device', device, '--precision', precision,
               '--attention_backend', attention_backend, '--eval_batch_size', str(batch_size),
               '--max_length', str(max_length)]
    if sample_submission:
        command += ['--sample_submission', str(sample_submission)]
    subprocess.run(command, check=True)
    prediction_columns, predictions = read_rows(output)
    if prediction_columns != ['id', 'label'] or [row['id'] for row in predictions] != expected_ids:
        raise ValueError('Prediction IDs or columns differ from the expected output.')
    for row in predictions:
        probability = float(row['label'])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('Every prediction must be a finite probability in [0, 1].')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--predict_csv', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample_submission', type=Path)
    parser.add_argument('--device', choices=['cuda', 'cpu', 'auto'], default='cuda')
    parser.add_argument('--precision', choices=['auto', 'bf16', 'fp16', 'fp32'], default='auto')
    parser.add_argument('--attention_backend', choices=['flash', 'math', 'flash_or_math'], default='flash_or_math')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_length', type=int, default=512)
    run_prediction(**vars(parser.parse_args()))
