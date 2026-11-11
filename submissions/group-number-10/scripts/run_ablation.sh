#!/bin/bash
# Formatting ablation at the grid-best setting (lr 3e-5, 4-epoch schedule).
set -u
PY=.venv/bin/python
$PY -u -m experiments.distilbert_train --lr 3e-5 --epochs 4 --model-name distilbert-base-cased --newline-marker "[NL]" --notes ablation
$PY -u -m experiments.distilbert_train --lr 3e-5 --epochs 4 --newline-marker "[NL]" --notes ablation
$PY -u -m experiments.distilbert_train --lr 3e-5 --epochs 4 --model-name distilbert-base-cased --notes ablation
echo ABLATION_DONE
