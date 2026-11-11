# Current team handoff

Start with the [course README](README.md) for setup and reproduction commands.

The current accepted submission is **reference-035**, reported by Alex at **0.97960** on the public leaderboard. The earlier four-method submission scored 0.97956. The private score is unknown.

## Current recipe

| Component | Weight |
|---|---:|
| Original word/character/style reference | 0.350 |
| GPT-2 + GPT-2 Large feature classifier | 0.234 |
| DistilRoBERTa | 0.156 |
| DeBERTa-v3-base | 0.260 |

The models and quantile calibration are shared with the earlier submission; only the final weights changed. Base fitting uses 9,200 supplied labeled examples, with 800 separate examples for calibration. No extra labeled datasets are used. Scores are for ranking, not literal authorship probabilities.

`predict_robust.py` now defaults to `--recipe current`, including with the earlier saved model bundle. `--recipe baseline` reproduces the earlier 0.97956 recipe; `--recipe saved` explicitly uses the weights serialized inside a bundle. New final training saves the current weights. `train_robust.py --validation` defaults to the baseline recipe to reproduce the historical audit; pass `--recipe current` to evaluate the accepted reblend instead.

## Files to start with

- `Four_Method_Robust_Colab.ipynb`: self-contained fresh T4 training, prediction, checks, and model export; upload the original competition ZIP.
- `train_robust.py`, `predict_robust.py`, `reblend_robust.py`: training, full inference, and a fast CPU-only reblend.
- `robust_validation/current_recipe.json`: current weights, paired checks, and reported public scores.
- `ROBUST_ENSEMBLE.md`: the earlier four-method recipe's detailed validation record and timing/provenance. Its historical 0.375 reference weight is not the current default.

The older `train.py` and `predict.py` remain the team's original three-way control. Use the `*_robust.py` commands for this pipeline.

## Reuse a trained bundle

Install `requirements-robust.txt` alongside a suitable PyTorch installation (CUDA/T4 recommended), extract the existing trained bundle, and run:

```bash
USE_TF=0 USE_FLAX=0 python predict_robust.py --model final_model --input data/test.csv --output submission_current.csv --recipe current
```

This also scores a professor-provided later CSV with `id,text`, preserving its row order. No new fitting or recalibration occurs during inference. Public pretrained GPT-2 weights download automatically; offline copies can be supplied with `--offline-root`.

If you already have the SAME fitted model's baseline 0.375/0.225/0.15/0.25 predictions for that input file, the shortcut avoids repeating all neural inference:

```bash
USE_TF=0 USE_FLAX=0 python reblend_robust.py --model four_method_bundle.zip --input data/test.csv --baseline submission_four_methods.csv --output submission_current.csv
```

The shortcut computes only the reference score and reconstructs the combined contribution of the other components from the baseline CSV. It requires the same model, input IDs/order, and baseline recipe. Do not feed it the already-reblended 0.97960 CSV. It preserves the baseline file.

## Reproduce or improve training

In Colab: upload the notebook, select T4 GPU, Run all, and upload the original competition ZIP. Expect about 1.5–2 hours for a fresh complete run, plus downloads; the recorded final fit alone took 33.46 minutes with language features already cached. The notebook produces the CURRENT recipe by default and never submits anything.

Training features and fitted checkpoints are reusable. Keep them outside Git. The trainer supports `--small-features` and `--large-features` with matching dataset/model metadata. Source/configuration changes require a fresh output directory; reuse compatible feature files explicitly instead of recomputing them. Model ZIPs, feature caches, and original data are retained locally; this source handoff does not provide a hosted weights download. Obtain the existing fitted bundle from Alex or reproduce it with the notebook.

## Evidence and next improvements

The accepted reblend's original-text audit AUC is 0.999558 (earlier recipe: 0.999603), flattened AUC 0.994925 (0.994209), and short-line AUC 0.997057 (0.996970). All three paired news checks also improved. These are local validation fits, not private-test estimates; the public scores above were reported by Alex.

Known risks: the audit split has prior project exposure; content-development checks influenced selection; human poetry is poorly covered by supplied training labels. Synthetic wrapping does not establish real human-poetry performance. Keep data groups/near-duplicates together during evaluation, fit preprocessing only on each fitting partition, and separate model selection from evaluation. Use the supplied labels only; compare any new detector against this baseline on content and formatting checks before replacing it.

The 800-row calibration partition is excluded from supervised component fitting. It fits fixed component quantiles; inference never ranks against the other rows of the input file. Current weights are fixed for future data. The submitted CSV already includes all 3,000 test IDs; Kaggle's quoted 30/70 split changes which rows contribute to public/private scoring, not which predictions are in this file.
