# Q5: punctuation ablations

## What was tested

Compare the selected Q1 Model A against (1) curly-quote normalization and (2) a word-only tokenizer. Fixed settings, the same saved 8,000/2,000 split, and training-only five-fold CV are used. The baseline is retained because both alternatives have lower mean CV AUROC. Negative results are reported honestly.

Source: https://github.com/vr382/human-or-ai-submissions/tree/group-16
Q1 source commit: db769ff154a6fb5c49d9cc7be355d887af7c9d63

## Reproduce

Use Python 3.12 and run from this q5 directory:

```sh
python -m pip install -r requirements.txt
python q5_experiment.py --train /path/to/competition/train.csv --val-ids results/val_ids.csv --q1-dir q1_source --out rerun_results
```

Only the competition's labeled train.csv is needed. It is not included. Q1 source files are unchanged snapshots; q5_experiment.py imports its model factory. The script computes CV first and saves selection.json before any validation scoring. The candidates and rule are documented in manifest.json.

Quote normalization maps U+2018/U+2019 to an ASCII apostrophe and U+201C/U+201D to an ASCII double quote. It is applied before the unchanged vectorizer, so lowercasing is preserved. The separate word-only ablation retains one-character words. No other hyperparameters are changed. NumPy is seeded before every fit. TF-IDF and model fitting are restricted to two computational threads.

## Results

- Baseline: CV AUROC 0.961877; validation AUROC 0.966246; FP=64, FN=145.
- Quote normalization: CV 0.960905; validation 0.965481; FP=54, FN=146.
- Word-only: CV 0.945498; validation 0.947583; FP=90, FN=155.
- Quote-normalization minus baseline validation AUROC: -0.000765. Descriptive paired bootstrap 95% interval: [-0.002589, +0.001044].

Timing is the median of three TF-IDF + classifier fits, with rotating candidate order. It excludes quote mapping, validation scoring, CV and bootstrap; wall-clock times vary by machine/load. Q1's historical timing is not used as the comparator.

The original Q1 repository did not contain validation IDs. These IDs were reconstructed for Q4 using its seed-42 split procedure and the original supplied CSV order. Q5 reuses those exact IDs. Data/code/ID hashes and versions are recorded in results/manifest.json. Q4 already inspected this validation set, so Q5 is exploratory development evidence, not independent confirmation. Bootstrap intervals do not capture training uncertainty or domain shift. No Kaggle test data or hidden labels were used.

## Files and GitHub

Upload this entire q5 directory under submissions/group-16/. The included .gitignore makes only results/*.csv exceptions to the repository's global CSV ignore rule; no raw training data is included. q1_source/ holds the exact source used for reproducibility. Q5_answers.md contains the report answer; results/ includes CV fold assignments, all timings, validation scores, subgroup checks, and the ten Q4 errors' follow-up scores.

AI assistance: Codex designed and ran these ablations, checked the measurements, and drafted Q5. The team should review the code and results and update its Q6.3 disclosure to reflect its own verification. Existing Q1/Q4 files do not need to be overwritten to add this folder.
