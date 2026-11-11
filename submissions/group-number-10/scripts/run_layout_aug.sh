#!/bin/bash
# Layout augmentation for DeBERTa-large: half of the training passages get a random line layout each epoch
# (src/layout.py), so line breaks stop being a shortcut (the stress test showed poem-like short lines flip
# 821 of 1,000 AI validation passages to "human").
#   1) 8k split, seed 42, --layout-aug 0.5 -> val AUROC
#   2) stress test on the new checkpoint -> robustness to layout changes
#   3) gate: val >= 0.9990 and short_lines AUROC >= 0.95 -> full 10k, seeds 42 + 43 (Kaggle file)
# Queued after run_electra.sh (one job at a time).
set -u
cd ~/aml-kaggle/team-10
export HF_HUB_OFFLINE=1
while pgrep -f "[r]un_electra.sh|[r]un_format_stress.sh" >/dev/null; do sleep 30; done
systemctl --user is-active -q llm.service && { echo "AUG_ABORT llm.service is running $(date)"; exit 1; }
capped() {  # capped <module> <args...>
  .venv/bin/python -u -c 'import runpy, sys, torch
torch.cuda.set_per_process_memory_fraction(0.85)
sys.argv = sys.argv[1:]
runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)' "$@"
}
LARGE="--model-name models/deberta-v3-large --lr 1e-5 --newline-marker [NL] --epochs 3 --warmup-ratio 0.1 --layout-aug 0.5"
RUN=deberta-v3-large_nl_wu0.1_aug0.5_lr1e-05_ep3_bs16_len512_seed42
echo "AUG_START $(date)"
capped experiments.distilbert_train $LARGE --limit 64 --epochs 1 --no-checkpoint || { echo "AUG_FAIL smoke $(date)"; exit 1; }
capped experiments.distilbert_train $LARGE --seed 42 --notes "layout augmentation 0.5" || { echo "AUG_FAIL val run $(date)"; exit 1; }
echo "AUG_VAL_DONE $(date)"
capped experiments.format_stress --checkpoints checkpoints/deberta/$RUN || { echo "AUG_FAIL stress $(date)"; exit 1; }
echo "AUG_STRESS_DONE $(date)"
.venv/bin/python - "$RUN" <<"EOP" || { echo "AUG_STOP gate not passed $(date)"; exit 0; }
import json, sys
run = sys.argv[1]
val = json.load(open(f"results/metrics/{run}.json"))["val_auc"]
short = json.load(open(f"results/format_stress/{run}.json"))["results"]["short_lines"]["auroc"]
print(f"gate: val_auc={val:.5f} (need >= 0.9990), short_lines auroc={short:.5f} (need >= 0.95)")
sys.exit(0 if val >= 0.9990 and short >= 0.95 else 1)
EOP
capped experiments.distilbert_full $LARGE --seeds 42 43 || echo "AUG_FAIL full $(date)"
echo "AUG_DONE $(date)"
