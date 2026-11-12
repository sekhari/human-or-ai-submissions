# Q2: training data size

Party in Thee Code. This adds the five required learning-curve runs.
The baseline, Q1 results and original train/validation split files are unchanged.

From the team folder:

```bash
python -m pip install -r requirements_q2.txt
python q2.py --train data/train.csv
```

Or open q2_learning_curve.ipynb beside baseline.py and q1.py. The notebook
calls the same run_q2 function and has executed outputs already. One entrypoint
is enough. Put the supplied train.csv in data/ only for local execution.
Raw data are not included in the upload package.

The model is the Q1 character TF-IDF + linear SVM with C=1, character 3-5-grams,
min_df=2, max_features=50000, sublinear TF, smoothed IDF and L2 normalization.
All settings stay fixed. Each size has a fresh vectorizer and classifier,
trained only on that subset. We use decision scores for AUROC, without the
probability-calibration step. Scores in curve_scores.csv are not probabilities.

Subsets contain equal human and AI counts. Seeded per-class permutations
produce nested sets of 500, 1000, 2000, 5000 and 8000 rows. We retain the shared
training order, so the 8000-row endpoint uses exactly the Q1 rows and order.
The same 2000 validation IDs are used throughout.

Files:
- q2.py: reproducible experiment and plotting code.
- q2_learning_curve.ipynb: executed notebook.
- requirements_q2.txt: baseline dependencies plus matplotlib.
- results/q2/learning_curve.csv: five-row metric table.
- results/q2/q2_learning_curve.pdf and .png: the figure.
- results/q2/subset_ids/: exact IDs used at each size.
- results/q2/curve_scores.csv: decision scores and labels for checking AUROC.
- results/q2/experiment_config.json and environment.json: settings and versions.
- results/q2/verification.json: output and split checks.

This is one nested sampling run, not a repeated-seed estimate. The plot has no
confidence bands. Q2 does not change the Q1 selected-model predictions or train
a final Kaggle model on all 10000 labels.

AI assistance: ChatGPT helped draft the Q2 code and text, execute the notebook,
check results and format the figure/LaTeX. The final disclosure should also
reflect the team's own checks and changes.
