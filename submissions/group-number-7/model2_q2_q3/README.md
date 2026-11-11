# Model 2 — Q2 learning curve and Q3 unlabeled-data experiment

This module extends the teammate-provided **character TF-IDF + calibrated linear SVM** implementation. It reproduces the selected configuration and preserves the canonical holdout. It is separate from Model 1 and does not change the original teammate source. It prepares local results only; it does not upload to Kaggle/GitHub or edit Overleaf.

## Source and scope

- Original implementation: `JD-TB1/cornell-tech-ai-human-model2`, revision `1b1d379964c8b2ceb75e4eef531bbf90623ad176`, as attributed in the supplied archive. The source repository was not independently fetched in this extension.
- The user supplied `cornell-model2.zip`. The unchanged `model2/model.py`, `model2/data.py` and `model2/__init__.py` are copied from that archive. `MODEL2_PROVENANCE.md` preserves its original attribution; its references to `records/` concern the supplied source project.
- `config/selected.json` is the teammate's selected configuration. `config/shared_split.csv` is the teammate's canonical ID-only split manifest.
- New work: `common.py`, `run_q2.py`, `run_q3.py`, `verify.py`, experiment protocol and documentation. Q2/Q3 contributor: Yaoan Xie (yx749), with OpenAI Codex assistance for implementation, checking, analysis and draft writing. Model 2 authorship remains attributed to its source authors.
- This is a Q2/Q3 module, not the complete team submission or a new final Kaggle model.

## Install and inputs

Verified with Python 3.12.6 on Apple Silicon macOS. Core numerical versions match the supplied Model 2 record (NumPy 2.4.6, SciPy 1.17.1, pandas 3.1.1, scikit-learn 1.9.1); the teammate's original Python version was 3.11.5. All exact installed dependencies are in `requirements.txt`.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

Obtain the original official competition CSVs from the [Kaggle data page](https://www.kaggle.com/competitions/cornell-tech-cs-5785-combined-2026-ai-vs-human-text/data). Place `train.csv` and `unlabeled.csv` in `data/`. The scripts verify their SHA-256 hashes. Do not resave or reorder the original inputs. Neither `test.csv` nor the starter is used. Do not include raw data or credentials in the course PR.

## Run in order

```bash
.venv/bin/python run_q2.py --data-dir data --output-dir outputs/q2
.venv/bin/python run_q3.py --data-dir data --q2-dir outputs/q2 --output-dir outputs/q3
.venv/bin/python verify.py --data-dir data --q2-dir outputs/q2 --q3-dir outputs/q3
```

All output directories must be new; existing runs are protected from overwrite. Quote paths containing spaces. For optional reproduction against the teammate's stored outputs, add `--reference-dir /path/to/original/artifacts/run-001/selected` to `run_q2.py`. This compares all ordered train/validation scores at 8,000 examples, with probability tolerance `1e-8`. The original archive is not otherwise required.

## Fixed methods

Model settings: character 2–4 grams, lowercase, sublinear TF, `min_df=2`, `max_features=100000`; `LinearSVC(C=1, dual="auto", max_iter=10000, random_state=42)`. Default TF-IDF normalization and smoothed IDF are retained. The calibrated teacher uses three shuffled stratified folds, seed 42, sigmoid calibration and `ensemble=False`. Features fit inside each internal fold; validation labels never fit or calibrate a model. Final training uses the original CSV order within the canonical partition, as the teammate's `partition()` does. Membership equals the agreed original-order 80/20 stratified seed-42 split, but row order differs from the random order returned directly by `train_test_split`. Preserve Model 2's order to reproduce calibration folds.

**Q2:** Use 500/1,000/2,000/5,000/8,000 genuinely labeled examples. Shuffle Human positions, then AI positions once with `numpy.random.default_rng(42)`; take half of each class at each size, preserving the original Model 2 partition order. Larger subsets contain smaller ones. Refit both TF-IDF and SVM/calibration from scratch at every size. Train AUROC uses that point's training subset, and validation AUROC uses the same 2,000 holdout examples. Probability and SVM-margin AUROCs are checked to agree. This is one run per size, not a multi-seed confidence interval.

**Q3:** Use the 8,000-example Q2 model as a calibrated teacher. Assign pseudo-labels to `unlabeled.csv` only when `max(P(AI),1-P(AI)) >= 0.95`. The threshold, one-round limit and unit pseudo-label weight were fixed before the new experiments in `EXPERIMENT_PROTOCOL.json`. Exclude normalized exact overlaps with labeled data, duplicate unlabeled texts, and unlabeled texts with hashed word-trigram similarity >=0.9 to validation texts; this hygiene step uses no validation labels. Refit the same TF-IDF + SVM settings from scratch on the 8,000 true labels plus selected pseudo-labels. Compare both models using **raw SVM decision-score AUROC** on the unchanged holdout. The student's probability calibration is deliberately not fitted to pseudo-labels; probability calibration is unnecessary for this ranking comparison. The baseline's probability/margin AUROCs are also verified equal. No calibrated student probabilities or new Kaggle submission are claimed.

**Overlap sensitivity:** Recompute the teammate's hashed word-trigram cosine diagnostic (`2**18` features; float32; no alternate signs; L2 norm). The main results always use the canonical 2,000 validation examples. Secondary results exclude validation IDs with similarity >=0.9 to any original training text, without refitting. This is an approximate diagnostic, not proof that all source/near-duplicate dependence has been eliminated.

## Outputs and verification

Q2 produces `learning_curve.csv`, `learning_curve.png/.pdf`, ordered split IDs, `validation_overlap.csv`, `summary.json`, and five `n_<size>/` folders with model checkpoints, subset IDs, calibration fold IDs and predictions.

Q3 produces `comparison.csv`, `pseudo_labels.csv` (teacher probabilities, pseudo-labels and selection flags for all 3,000 rows), `augmented_train_ids.csv` (official versus pseudo label provenance), overlap audit, student checkpoint, decision-score predictions and `summary.json`. Q3 training AUROC is scored only on the original 8,000 true labels, not on pseudo-label correctness.

`verify.py` reloads all models, checks settings, ordered IDs, nested balanced subsets, calibration fold assignments, fitted IDF against the actual fit texts, saved predictions and independent rank-sum AUROCs. It also regenerates teacher pseudo-label decisions and checks the augmented training labels and comparison table. Successful checks write `verification.json` into both run folders. Read `summary.json` as completion metadata only; independent validation is recorded separately. Failed runs may retain diagnostics and must be rerun to new output directories after correction.

Only source, configuration, ID-only split manifest and documentation belong in the code contribution. Do not commit `outputs/`, raw CSVs, fitted models, per-article predictions, environments, private archive contents or credentials. Aggregate report tables/figures can be integrated into the report after user/team review. Keep Model 1's earlier learning curve as historical work; the final Q2 now uses Model 2.

## Interpretation and disclosure

Pseudo-labels are model guesses, not verified ground truth. A confidence threshold does not establish 95% accuracy on the unlabeled distribution. A small holdout gain does not establish statistical reliability, and none of these results guarantees Kaggle improvement. Configuration selection and the holdout were inherited; this is a reused development set, not an independent new generalization test. No test labels or external model services are used. The user must review and understand the results before submission.

Technical references: [scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html) and [self-training](https://scikit-learn.org/stable/modules/semi_supervised.html#self-training).
