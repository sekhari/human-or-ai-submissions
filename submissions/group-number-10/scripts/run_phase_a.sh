#!/bin/bash
# Phase A (validation split): double-space marker, seed variance, DAPT, lr re-check. Base: lr 3e-5, 4-epoch schedule.
set -u
PY=.venv/bin/python
T="$PY -u -m experiments.distilbert_train --epochs 4 --notes phaseA"
$T --lr 3e-5 --newline-marker "[NL]" --space-marker "[DS]" --seed 42
$T --lr 3e-5 --newline-marker "[NL]" --seed 43
$T --lr 3e-5 --newline-marker "[NL]" --seed 44
$T --lr 3e-5 --newline-marker "[NL]" --space-marker "[DS]" --seed 43
$T --lr 3e-5 --newline-marker "[NL]" --space-marker "[DS]" --seed 44
$PY -u -m experiments.distilbert_mlm --corpus split --newline-marker "[NL]" && \
  $T --lr 3e-5 --newline-marker "[NL]" --seed 42 --model-name checkpoints/mlm/dapt-split-nl
$T --lr 2e-5 --newline-marker "[NL]" --seed 42
$T --lr 5e-5 --newline-marker "[NL]" --seed 42
echo PHASE_A_DONE
