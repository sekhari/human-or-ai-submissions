#!/bin/bash
# Test-time augmentation (none + unwrap) on file 15's four models (aug 0.5, full 10k, seeds 42-45). Prediction only.
set -u
cd ~/aml-kaggle/team-10
export HF_HUB_OFFLINE=1
systemctl --user is-active -q llm.service && { echo "TTA_ABORT llm.service is running $(date)"; exit 1; }
R=checkpoints/full/deberta-v3-large_nl_wu0.1_aug0.5_lr1e-05_ep3_bs16_len512_seed
echo "TTA_START $(date)"
.venv/bin/python -u -m experiments.tta --checkpoints ${R}42_full --limit 16 --name smoke || { echo "TTA_FAIL smoke $(date)"; exit 1; }
.venv/bin/python -u -m experiments.tta --checkpoints ${R}42_full ${R}43_full ${R}44_full ${R}45_full \
  --name deberta-v3-large_nl_wu0.1_aug0.5_lr1e-05_ep3_full_seeds42-45_tta-unwrap || { echo "TTA_FAIL $(date)"; exit 1; }
echo "TTA_DONE $(date)"
