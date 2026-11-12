# Human or AI? · Group 36

Kaggle team name: **group 36**.

The team used the ensemble model approach in [abishek_ensemble](abishek_ensemble/) and uploaded **prob_average_final.csv** (to Kaggle), the chosen model has a public AUROC in the team’s submission history (**0.92207**). We had models that performed better and received higher public AUROC (not by much) but chose this model for its simplicity and stability. The code is also available in the group's main directory as well as the submission [file](submission.csv).

Abishek’s available comparison notebook evaluates a five-model ensemble. Four logistic classifiers use Naive Bayes weights on different text representations. A fifth classifier uses gradient boosting on 67 writing-style features. In the saved experiment, the four text models each receive weight 1 and the style model receives weight 2. Essentially, the weighted average of the five predicted probabilities is the final P(AI). An earlier rank average based model is kept in abishek_ensemble/rank_average.ipynb along with its test probabilities under abishek_ensemble/submissions/rank_average_final.csv.

| Contributor | Model | Held-out AUROC | Experiments |
| --- | --- | ---: | --- |
| Abishek Chozhan (ac3388) | Five-model Weighted Probability Average | 0.9888 | [Final Ensemble Notebook](abishek_ensemble/ensemble_model.ipynb) · [Model comparison](abishek_ensemble/results/model_comparison.csv) |
| Liyan Yang (ly462) | Random Forest on 21 writing-style features | 0.989913 | [Experimental report](alma_rf/results/EXPERIMENT_REPORT.md) · [Run instructions](alma_rf/README.md) |
| Jianing Li (jl4837) | Word and character TF-IDF, chunked MiniLM and linear SVM | 0.974103 | [Experiments and run instructions](jianing_svm/README.md) · [Learning curve](jianing_svm/results/learning_curve/learning_curve.png) |

All three approaches use the same stratified partition of 8,000 training and 2,000 validation passages, with seed 42. The public scores in [the submission history](team_results/public_scores.csv) are keyed by filename. Jianing’s binary TF-IDF SVM scored 0.90977 on the public leaderboard and 0.977117 on validation.

## Reproduce the selected probability ensemble

From the team directory (`submissions/group-36/`), put the course files `train.csv`, `test.csv` and `sample_submission.csv` in `abishek_ensemble/data/`. Set up a Python 3.13 environment:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r abishek_ensemble/requirements.txt
```

Run the final notebook from the same team directory:

```bash
cd abishek_ensemble && jupyter nbconvert --to notebook --execute --inplace ensemble_model.ipynb
```

The notebook evaluates the five-model weighted probability average on the fixed validation split (seed 42), refits on 10,000 labeled passages, and writes `submissions/submission.csv` inside `abishek_ensemble/`. The selected submission achieved validation AUROC 0.9888 (rounded) and public AUROC 0.92207.

## Experimental evidence

- **Abishek.** The text models use word n-grams, within-word character n-grams, grammatical skeletons and cross-word character n-grams. The style model uses gradient boosting. Five-fold predictions on training data selected the style weight. Component model vocabularies use the 8,000 training texts before internal CV; the supervised NB ratios and regressions are refitted within each fold. The held-out validation texts are excluded from training. These experiments and data are reproduced from earlier versions of the model that use the rank-average version (with style weight 3): [Weight search](abishek_ensemble/results/style_weight_search.csv) · [Leave-one-model-out results](abishek_ensemble/results/ablation.csv) · [Validation predictions](abishek_ensemble/results/validation_scores.csv) · [ROC curves](abishek_ensemble/results/validation_roc.png)
- **Alma.** Representation and Random Forest settings were compared by training-only cross-validation. The report includes baseline comparisons, length subgroups and example errors. [RF results](alma_rf/results/)
- **Jianing.** Experiments compare term frequencies, frozen MiniLM and feature fusion. The learning-curve experiment refits vocabularies and IDF on each balanced subset. Validation AUROC increased from 0.890852 at 500 passages to 0.974103 at 8,000. [Training-size results](jianing_svm/results/learning_curve/learning_curve.csv)

[Experiment history](EXPERIMENT_HISTORY.md) · [Data split](jianing_svm/data/splits/seed42_stratified_8000_2000.csv)

## Unlabeled-data experiment

[Experiment and reproduction instructions](abi_unlabeled/README.md). Adding 3,000 unlabeled passages to vocabulary fitting in the reconstructed stack changed fixed-validation AUROC from **0.988825 to 0.988813**. [Supervised reconstruction and separate SVM control](group36_evidence/) · [Paired validation comparison](abi_unlabeled/results/comparison.json).

The team reports **0.92117 public AUROC** for the unlabeled-vocabulary submission. Our final submission has a public AUROC 0.92207.

## Run the reconstructed stack

From the team directory, place the course CSVs in data/ and use a Python 3.13 environment:

~~~sh
python -m pip install -r group36_evidence/requirements.txt
python group36_evidence/rebuild.py
python group36_evidence/summarize.py
python abi_unlabeled/experiment.py
python abi_unlabeled/compare.py
python abi_unlabeled/experiment.py --submission
~~~

The final command produces the unlabeled-vocabulary candidate (reported public AUROC 0.92117). The split and cross-validation seed is 42. The historical 0.92847 upload lacks a verified original generation command.

## AI assistance

- **Abishek Chozhan** used Claude (Anthropic) via Claude Code to discuss modeling options, write and debug notebook code, verify results and help draft report text. All code was run and all reported numbers were checked by the team.
- **Liyan Yang** used ChatGPT to help write Random Forest code, analyze results and plan subsequent experiments. She compared lexical and style representations and evaluated alternative tree settings.
- **Jianing Li** used Codex to help implement and debug SVM models, organize experiments, and draft and edit report text. She edited and ran notebooks in Jupyter and submitted predictions to Kaggle. Codex also executed the learning-curve, stacking-reconstruction and unlabeled-vocabulary experiments and helped check IDs and recompute metrics.
