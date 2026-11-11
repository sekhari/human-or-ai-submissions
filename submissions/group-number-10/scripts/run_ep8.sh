#!/bin/bash
# Training length: DeBERTa-large + [NL] + layout augmentation 0.5, 8k split, seed 42, 8 epochs.
# Records validation AUROC after every epoch (results/metrics/<run>.json history), then the line-break stress
# test on the saved checkpoint (best epoch by validation). Compare with the 3-epoch run (val 0.99942).
set -u
cd ~/aml-kaggle/team-10
export HF_HUB_OFFLINE=1
while pgrep -f "[r]un_tta.sh|[r]un_aug_more.sh|[r]un_layout_aug" >/dev/null; do sleep 30; done
systemctl --user is-active -q llm.service && { echo "EP8_ABORT llm.service is running $(date)"; exit 1; }
capped() {  # capped <module> <args...>
  .venv/bin/python -u -c 'import runpy, sys, torch
torch.cuda.set_per_process_memory_fraction(0.85)
sys.argv = sys.argv[1:]
runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)' "$@"
}
RUN=deberta-v3-large_nl_wu0.1_aug0.5_lr1e-05_ep8_bs16_len512_seed42
echo "EP8_START $(date)"
capped experiments.distilbert_train --model-name models/deberta-v3-large --lr 1e-5 --newline-marker [NL] \
  --warmup-ratio 0.1 --layout-aug 0.5 --epochs 8 --seed 42 --notes "training length: 8 epochs" \
  || { echo "EP8_FAIL train $(date)"; exit 1; }
echo "EP8_VAL_DONE $(date)"
capped experiments.format_stress --checkpoints checkpoints/deberta/$RUN || { echo "EP8_FAIL stress $(date)"; exit 1; }
echo "EP8_DONE $(date)"
