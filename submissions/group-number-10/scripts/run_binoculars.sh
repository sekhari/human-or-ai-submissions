#!/bin/bash
# Zero-shot Binoculars scores (Falcon-7B observer, Falcon-7B-instruct performer) for test.csv and train.csv.
# Queued after run_more.sh and the model download (one job at a time). Smoke test on 32 passages first.
set -u
cd ~/aml-kaggle/team-10
export HF_HUB_OFFLINE=1
while pgrep -f "[r]un_more.sh|[d]l_falcon.sh" >/dev/null; do sleep 30; done
grep -q DL_DONE dl_falcon.log || { echo "BINO_ABORT download incomplete $(date)"; exit 1; }
systemctl --user is-active -q llm.service && { echo "BINO_ABORT llm.service is running $(date)"; exit 1; }
capped() {  # capped <module> <args...>
  .venv/bin/python -u -c 'import runpy, sys, torch
torch.cuda.set_per_process_memory_fraction(0.85)
sys.argv = sys.argv[1:]
runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)' "$@"
}
echo "BINO_START $(date)"
capped experiments.binoculars --sets train --limit 32 || { echo "BINO_FAIL smoke $(date)"; exit 1; }
echo "BINO_SMOKE_DONE $(date)"
capped experiments.binoculars --sets test train || echo "BINO_FAIL $(date)"
echo "BINO_DONE $(date)"
