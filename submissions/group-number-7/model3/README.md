# Model 3 — ConvoKit-based stylometry + logistic regression

**Team member:** Yuchen (Wayne) Yang, yy2458. This reconstructed notebook documents the Model 3 experiment and is provided so an evaluator can rerun feature extraction and model training. It is an experimental model, not the selected Kaggle model.

## Run in Colab

1. Open `model3_convokit_stylometry.ipynb` in Google Colab.
2. Upload the *official, original-order* Kaggle `train.csv`, `test.csv`, `unlabeled.csv`, and `sample_submission.csv` to the current Colab working directory (these files are **not** stored in GitHub).
3. Run all cells in order. The notebook installs ConvoKit, spaCy, NLTK, scikit-learn, etc., downloads `en_core_web_sm`, and checks the fixed seed-42 stratified split.
4. Feature extraction from 10,000 texts may take several minutes. Extracted features are cached under `results/` to support resuming an interrupted run. Note: Colab's ephemeral runtime means caches must be separately saved if persistence is needed.
5. Expected output: `results/model3_val_predictions.csv` (`id,label,p_ai`), and optional `submission_model3.csv` (`id,label`) for Model 3, not the final group entry.

The notebook uses `process_text(mode='parse')` with spaCy and NLTK. Features include sentence length and variability, type-token ratio, word length, repeated bigrams, POS tag proportions, dependency distance, punctuation, uppercase ratio, and digits. `StandardScaler` and `LogisticRegression(C=1, max_iter=2000, random_state=42)` are fitted on the training partition. The validation split must use the exact same original labeled CSV and the same seed-42 protocol as Models 1 and 2.

Original recorded results: train AUROC **0.8427**, validation AUROC **0.8267**; ~447 seconds feature extraction for 8,000 training texts, ~107 seconds for 2,000 validation texts, and ~0.042 seconds for Logistic Regression fit. These are on the recorded Colab environment, not guaranteed timings on other machines.

**Provenance:** This notebook was reconstructed from the team's previously run workflow. Reported scores should be attributed to the original experiment rather than this clean, output-free copy.
