# Reproduce Q4

These are measured results from the supplied Human or AI? competition training CSV.
Source: https://github.com/vr382/human-or-ai-submissions/tree/group-16
Source commit: db769ff154a6fb5c49d9cc7be355d887af7c9d63

Use Python 3.12 and install the recorded dependencies:

```sh
python -m pip install -r requirements.txt
python q4_analysis.py --q1-dir q1_source --train /path/to/train.csv --val-ids results/val_ids.csv --out rerun_results
```

Only the labeled train.csv is needed. It is not included. The original Q1 source files are copied unchanged in q1_source/. The archive includes derived evaluation results, but no raw training CSV, model weights, or full passages.

The original repository omitted its saved validation IDs. We reconstructed its seed-42 stratified split from the original order in the supplied competition CSV, then saved IDs here. This reproduces the reported validation AUROC to three decimals but cannot independently prove the historical split was identical. The evaluation script seeds NumPy before fitting LIBLINEAR; small version-dependent numerical differences can occur.

Results: training AUROC 1.000000; validation AUROC 0.966246. Counts at threshold 0.5: TN=936, FP=64, FN=145, TP=855. Length bins were specified before subgroup evaluation. Five most confidently wrong cases of each type were selected for qualitative analysis; these extremes are not a representative sample of errors.

summary.json records data/code hashes, library versions and settings. Feature contributions are additive AI log-odds terms, not causal explanations. The report's proposed ablations have not been run. Rerunning writes error_evidence.json with full selected passages for local inspection; keep that file local rather than committing it.

AI assistance: Codex prepared the evaluation script, computed the reported metrics, inspected selected errors, and drafted the Q4 answer. The team should review the code, passages, and interpretations before submission and update its Q6.3 disclosure accordingly.
