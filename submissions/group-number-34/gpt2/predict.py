"""Write AI-authorship probabilities from a trained GPT-2 classifier."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--test-csv', type=Path, required=True)
    parser.add_argument('--sample-submission', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu', 'auto'], default='cuda')
    parser.add_argument('--precision', choices=['bf16', 'fp16', 'fp32', 'auto'], default='bf16')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--attention-backend', choices=['flash_or_math', 'flash', 'math'], default='flash_or_math')
    args = parser.parse_args()
    command = [sys.executable, '-u', str(Path(__file__).parent / 'src/train_stage.py'),
               '--stage', 'predict', '--checkpoint', str(args.checkpoint),
               '--predict_csv', str(args.test_csv), '--sample_submission', str(args.sample_submission),
               '--prediction_output', str(args.output), '--device', args.device,
               '--precision', args.precision, '--eval_batch_size', str(args.batch_size),
               '--max_length', '512', '--attention_backend', args.attention_backend,
               '--seed', '42']
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
