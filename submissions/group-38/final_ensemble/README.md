# November 9 generalization experiments

The team selected the final BERT/CNN/tree/previous-model ensemble on November 9. User-reported public AUROC is approximately 0.97; exact precision and private AUROC are unknown. The user confirmed that this score belongs to submission_recommended.csv including BERT, superseding the earlier 0.956 attribution.

## Reproduce

Tested with Python 3.14.7 on an Apple M3 (16 GB); neural models use MPS and tree models use CPU. On macOS, XGBoost requires OpenMP (`brew install libomp`).

```sh
python -m pip install -r requirements.txt
python run.py --data /absolute/path/to/cornell-tech-cs-5785-combined-2026-ai-vs-human-text.zip --output results
```

The supplied ZIP has CSVs at its root. The runner rebuilds the original ensemble from the sibling `../improved_ensemble/` directory, extracts features, compares tree variants, trains a character CNN, checks a style-kernel SVM, tests self-training, fine-tunes a compact BERT and writes verified predictions under `results/deliverables/`. Training can take a while on a laptop. Scripts run locally and keep weights, datasets and predictions in the output folder. No competition text is sent to an external inference API.

## Selected final model

The final prediction is 28% original improved ensemble + 28% character CNN + 24% enriched-feature trees + 20% fine-tuned compact BERT. The tree component equally averages XGBoost depth 3, XGBoost depth 5 and CatBoost depth 6. `finalize.py` freezes these components and weights; unsuccessful SVM/pseudo-label experiments are excluded. The earlier MiniLM/Hybrid three-way blend is a separate historical model.

Original submitted CSV: `submission_recommended.csv`, 3,000 rows. SHA-256: `ce5ad51331883b22add000923e606a6cee1abdd87fc5fc112045a77060b05753`. Exact submitted predictions and trained weights remain local under the repository's artifact exclusions. Re-training on another runtime may produce slightly different predictions; the original CSV hash identifies the Kaggle-scored file.

Fixed validation AUROC: 0.999393; test-like validation subset AUROC: 0.999478. These are exploratory, repeatedly reused local validation estimates, not Kaggle scores. Public AUROC is approximately 0.97 as reported and attributed by the user; no exact leaderboard decimal is claimed. See `selected_model.json`, `RESULTS.md` and `experiment_records.json`.

## Data and validation

- Supplied labels: 10,000 passages. Additional unlabeled passages: 3,000. Test passages: 3,000, without authorship labels.
- Fixed 8,000/2,000 stratified split, seed 42, matching the shared team split.
- A separate text-domain classifier distinguishes 8,000 training texts from unlabeled test texts. Its top-scoring validation quartile forms the test-like stress subset. This auxiliary grouping uses no test authorship labels.
- Selection averages overall validation AUROC and test-like-quartile AUROC. This is an exploratory objective. Repeated selection on this fixed validation set can produce optimistic estimates.
- Data checks found zero exact text overlap between train, unlabeled and test files.

## Models

The original ensemble combines 75% style boosting, 15% style MLP and 10% word/character logistic regression. Its source is available in the sibling `../improved_ensemble/` directory.

New 781-dimensional fixed features extend the original 99 style statistics with function-word frequencies, word/sentence/paragraph-length distributions, vocabulary repetition, readability, POS counts and coarse POS bigrams. Feature extraction uses no authorship labels. CatBoost controls compare old/enriched features and XGBoost uses enriched features. Round counts are chosen on validation, then fixed for full-label refitting. A depth-8 CatBoost pilot was interrupted for high computation cost; it has no completed score.

The character CNN uses UTF-8 bytes, a 32-dimensional embedding, 64 filters for widths 3/5/7, max/mean pooling and a 128-unit hidden layer with dropout. Input keeps first 768 and last 256 bytes. AdamW uses lr=.001, weight decay=.01, batch 64. Six epochs are compared; the selected epoch is retrained on all supplied labels.

Style RBF SVM compares C in [1,10,100] on training-scaled original features clipped to +/-5. Sigmoid margins are bounded ranking scores; this exploratory model is excluded unless evidence supports it.

Self-training requires two training-only parent models to agree above .995 or below .005. Pseudo-example weights .2/.5 are compared. Validation labels never generate pseudo-labels. The final parents use all supplied labels. A pseudo-label model is eligible for blending only if it improves both validation checks over its parent.

Compact BERT fine-tunes all four layers plus its classification head, using 256 tokens (first/last 127), batch 32 and four candidate epochs. Encoder lr=3e-5, head lr=1e-4, 50-step warm-up, AdamW weight decay=.01. The selected epoch is refit on all supplied labels. Small blend contributions .05/.10/.20 are compared. Completed scores and final chosen weights appear in the result records.

## Pretrained resources and implementation references

- [NLTK averaged perceptron POS tagger](https://www.nltk.org/api/nltk.tag.perceptron.html): grammatical annotation. Its JSON data downloads into the local output directory.
- [prajjwal1/bert-mini](https://huggingface.co/prajjwal1/bert-mini), MIT license: generic pretrained base weights. The tokenizer uses the identical BERT vocabulary in distilbert-base-uncased; vocabulary SHA-256 is `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3`.
- [CatBoost documentation](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier), [XGBoost documentation](https://xgboost.readthedocs.io/en/stable/python/python_api.html).

Authorship training uses only supplied competition labels, including explicitly generated pseudo-labels from supplied unlabeled data. No external human-vs-AI labeled dataset is added.

## AI disclosure

Codex designed, implemented and ran these experiments and checked the submission files. Team members should inspect the code and results, understand the choices and disclose their actual participation accurately. Completed, failed and interrupted experiments are distinguished in the records.
