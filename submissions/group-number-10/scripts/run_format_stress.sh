#!/bin/bash
# Line-break stress test on DeBERTa-large (8k split, seeds 42 + 43). Inference only, one job at a time.
set -u
cd ~/aml-kaggle/team-10
export HF_HUB_OFFLINE=1
while pgrep -f "[r]un_more.sh|[r]un_binoculars.sh" >/dev/null; do sleep 30; done
systemctl --user is-active -q llm.service && { echo "STRESS_ABORT llm.service is running $(date)"; exit 1; }
capped() {  # capped <module> <args...>
  .venv/bin/python -u -c 'import runpy, sys, torch
torch.cuda.set_per_process_memory_fraction(0.85)
sys.argv = sys.argv[1:]
runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)' "$@"
}
C=checkpoints/deberta/deberta-v3-large_nl_wu0.1_lr1e-05_ep3_bs16_len512_seed
echo "STRESS_START $(date)"
capped experiments.format_stress --checkpoints ${C}42 --limit 16 || { echo "STRESS_FAIL smoke $(date)"; exit 1; }
capped experiments.format_stress --checkpoints ${C}42 ${C}43 || echo "STRESS_FAIL $(date)"
echo "STRESS_DONE $(date)"
