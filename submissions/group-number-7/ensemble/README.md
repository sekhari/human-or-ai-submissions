# Q5 — Probability-based ensembles

**Team member:** Yuchen (Wayne) Yang, yy2458. This experiment compares Model 2 alone with equal-weight probability mixtures of Models 1–3. Unequal weights are clearly designated **exploratory** because their selection reused the reported validation set; they are not independent evidence of improvement. The team retained Model 2 as its final Kaggle entry.

The `run.py` script works with 2,000-row **validation prediction files**, not competition text or test labels. It checks ID alignment and equality of ground-truth labels before computing AUROC.

```bash
# Run from submissions/group-number-7/
python ensemble/run.py \
    --model1 outputs/model1_eval/validation_predictions.csv \
    --model2 outputs/model2_eval/validation_predictions.csv \
    --model3 /path/to/model3_val_predictions.csv \
    --outdir outputs/q5_ensemble
```

Outputs: an aggregate `ensemble_summary.csv` and a **LOCAL_ONLY** predictions CSV. The aligned validation predictions and full official Kaggle data must not be committed to the public course repository. The notebook `q5_ensemble_analysis.ipynb` provides an alternative, interactive analysis, where the locally cached file names are `model1_reproduced_validation_predictions.csv`, `validation_predictions.csv`, and `model3_val_predictions.csv`.

The original group experiment used a Model 1 reconstruction and Yokie's recorded Model 2 validation probabilities, producing standalone AUROCs M1=0.936077, M2=0.970440 and M3=0.826659. Equal-weight mixtures did not exceed M2; an exploratory 20/60/20 mixture achieved 0.972428 but was selected on the same validation set. Fresh reruns across software versions may differ slightly. The runtime metric in `run.py` measures only aggregation of cached probabilities and **excludes model training, feature extraction, prediction, and AUROC evaluation**.
