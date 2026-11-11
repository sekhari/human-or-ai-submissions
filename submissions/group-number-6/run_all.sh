#!/usr/bin/env bash
# Reproduce every experiment in the report, then the Kaggle submission.
# Run from this folder with the Kaggle CSVs in ./data/. Total ~1.5 h on a 2-core laptop.
# Steps are sequential on purpose: running them in parallel oversubscribes the CPU and
# distorts the reported fit times.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs results figures

python experiments/00_make_split.py                       | tee logs/q0.log   # split + leakage check
python experiments/01_compare_models.py                   | tee logs/q1.log   # Q1 (~20 min)
python experiments/02_learning_curve.py --model C_style_gbm | tee logs/q2_C.log # Q2
python experiments/02_learning_curve.py --model B_char_lr   | tee logs/q2_B.log
python experiments/03_unlabeled.py --model B_char_lr      | tee logs/q3_B.log   # Q3
python experiments/03_unlabeled.py --model C_style_gbm    | tee logs/q3_C.log
python experiments/05_improvement.py                      | tee logs/q5.log   # Q5 (~30 min)
python experiments/04_error_analysis.py --config ensemble+norm | tee logs/q4.log # Q4

python train.py --validate                                 | tee logs/final_validate.log
python train.py --model-out models/final.joblib            | tee logs/final_train.log
python predict.py --model models/final.joblib --out submission.csv
