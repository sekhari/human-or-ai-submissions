#!/bin/bash
# DeBERTa-v3-large WITHOUT the [NL] newline marker (never trained before; every DeBERTa run used [NL]).
#   0) check whether the DeBERTa tokenizer keeps line breaks at all
#   1) 8k split, seed 42 -> val AUROC;  2) line-break stress test
#   3) gate: val >= 0.9990 -> full 10k, seeds 42 + 43 (Kaggle file, compare with files 5 and 13)
# Queued after run_ep8.sh (one job at a time).
set -u
cd ~/aml-kaggle/team-10
export HF_HUB_OFFLINE=1
while pgrep -f "[r]un_ep8.sh|[r]un_tta.sh" >/dev/null; do sleep 30; done
systemctl --user is-active -q llm.service && { echo "NONL_ABORT llm.service is running $(date)"; exit 1; }
capped() {  # capped <module> <args...>
  .venv/bin/python -u -c 'import runpy, sys, torch
torch.cuda.set_per_process_memory_fraction(0.85)
sys.argv = sys.argv[1:]
runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)' "$@"
}
LARGE="--model-name models/deberta-v3-large --lr 1e-5 --epochs 3 --warmup-ratio 0.1"
RUN=deberta-v3-large_wu0.1_lr1e-05_ep3_bs16_len512_seed42
echo "NONL_START $(date)"
.venv/bin/python - <<"EOP"
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("models/deberta-v3-large")
a, b = tok("one two\nthree")["input_ids"], tok("one two three")["input_ids"]
print("tokenizer check: newline", "IGNORED (same ids as a space)" if a == b else f"KEPT ({a} vs {b})")
EOP
capped experiments.distilbert_train $LARGE --seed 42 --notes "DeBERTa-large without [NL]" || { echo "NONL_FAIL val run $(date)"; exit 1; }
echo "NONL_VAL_DONE $(date)"
capped experiments.format_stress --checkpoints checkpoints/deberta/$RUN || { echo "NONL_FAIL stress $(date)"; exit 1; }
echo "NONL_STRESS_DONE $(date)"
.venv/bin/python - "$RUN" <<"EOP" || { echo "NONL_STOP gate not passed $(date)"; exit 0; }
import json, sys
val = json.load(open(f"results/metrics/{sys.argv[1]}.json"))["val_auc"]
print(f"gate: val_auc={val:.5f} (need >= 0.9990)")
sys.exit(0 if val >= 0.9990 else 1)
EOP
capped experiments.distilbert_full $LARGE --seeds 42 43 || { echo "NONL_FAIL full $(date)"; exit 1; }
echo "NONL_DONE $(date)"
