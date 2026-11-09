# Start here

See [TEAM_HANDOFF.md](TEAM_HANDOFF.md) for the current 0.97960 recipe, training/inference commands, cached-feature reuse, and validation limitations.

- Already accepted predictions: `predictions/submission_current_best.csv`.
- Retrain on T4: `Four_Method_Robust_Colab.ipynb` (original competition ZIP required).
- Score with existing weights: `predict_robust.py --recipe current`.
- Fast conversion of this model's earlier baseline CSV: `reblend_robust.py`.

Model weights and original data are not in Git. Nothing is submitted automatically.
