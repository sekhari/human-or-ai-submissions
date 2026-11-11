#!/bin/bash
# Follow-ups on pseudo-labeling, queued after run_curve_holdout.sh (one job at a time):
#   A) DistilBERT+[NL] + 8k pseudo-labels, seeds 42/43/44 (baselines exist; val less saturated, known seed noise)
#   C) DeBERTa-large full-data + pseudo-labels, seed 43 -> 2-seed average with seed 42 (Kaggle candidate)
#   B) DeBERTa-large 8k split, seed 43, without and with pseudo-labels (is the seed-42 gain repeatable?)
# DistilBERT stays fp32 to match its baselines. DeBERTa runs (C, B) use bf16 if the bf16 test matched fp32
# (val >= 0.9995), else fp32; B's two runs always share one precision, so the pair stays comparable.
set -u
cd ~/aml-kaggle/team-10
export HF_HUB_OFFLINE=1
while pgrep -f "[r]un_pseudo.sh|[r]un_bf16_large.sh|[r]un_lr_sweep.sh|[r]un_curve_holdout.sh" >/dev/null; do sleep 30; done
capped() {  # capped <module> <args...>
  .venv/bin/python -u -c 'import runpy, sys, torch
torch.cuda.set_per_process_memory_fraction(0.85)
sys.argv = sys.argv[1:]
runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)' "$@"
}
PREC=""
.venv/bin/python - <<"EOP" && PREC="--bf16"
import json, sys
try:
    auc = json.load(open("results/metrics/deberta-v3-large_nl_wu0.1_bf16_lr1e-05_ep3_bs16_len512_seed42.json"))["val_auc"]
except FileNotFoundError:
    sys.exit(1)
sys.exit(0 if auc >= 0.9995 else 1)
EOP
LARGE="--model-name models/deberta-v3-large --lr 1e-5 --newline-marker [NL] --epochs 3 --warmup-ratio 0.1 $PREC"
echo "MORE_START $(date) deberta precision=${PREC:-fp32}"
for s in 42 43 44; do
  capped experiments.distilbert_train --lr 3e-5 --epochs 4 --newline-marker [NL] --seed $s \
    --extra-train results/pseudo/large8k.csv --no-checkpoint --notes "pseudo-labels from DeBERTa-large 8k" \
    || echo "MORE_FAIL distilbert seed $s"
done
echo "MORE_A_DONE $(date)"
# distilbert_full names runs by config, so a bf16 seed 43 cannot reuse the fp32 seed-42 file: train seed 43
# alone, then average the two seed files explicitly.
capped experiments.distilbert_full $LARGE --seeds 43 --extra-train results/pseudo/largefull4.csv \
  || echo "MORE_FAIL full seed 43"
.venv/bin/python - "$PREC" <<"EOP" || echo "MORE_FAIL 2-seed average"
import sys, pandas as pd
from predict import check_against_sample
from src.data import DATA_DIR
tag = "_bf16" if sys.argv[1] else ""
d = "results/submissions/"
a = pd.read_csv(d + "deberta-v3-large_nl_wu0.1_lr1e-05_ep3_bs16_len512_seed42_pl-largefull4_full.csv")
b = pd.read_csv(d + f"deberta-v3-large_nl_wu0.1{tag}_lr1e-05_ep3_bs16_len512_seed43_pl-largefull4_full.csv")
assert a["id"].equals(b["id"])
out = pd.DataFrame({"id": a["id"], "label": (a["label"] + b["label"]) / 2})
check_against_sample(out, DATA_DIR / "sample_submission.csv")
name = d + "deberta-v3-large_nl_wu0.1_lr1e-05_ep3_bs16_len512_pl-largefull4_full_seeds42-43.csv"
out.to_csv(name, index=False)
print(f"wrote {name} (seed 43 precision: {'bf16' if tag else 'fp32'}); seed Spearman {a['label'].corr(b['label'], method='spearman'):.3f}", flush=True)
EOP
echo "MORE_C_DONE $(date)"
capped experiments.distilbert_train $LARGE --seed 43 --notes "seed-43 baseline for pseudo-label comparison" \
  || echo "MORE_FAIL large seed 43 baseline"
capped experiments.distilbert_train $LARGE --seed 43 --extra-train results/pseudo/large8k.csv \
  --notes "self-training: + confident unlabeled.csv pseudo-labels (8k large seed42, t=0.02)" \
  || echo "MORE_FAIL large seed 43 pseudo"
echo "MORE_DONE $(date)"
