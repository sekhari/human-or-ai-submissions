# Adding a model to the classical track

Everything you need is on this page. One model = one file in `models/`, one config in `configs/`,
one line in the registry. `train.py` handles the split, the hyperparameter search, the metrics, the
results files and the testing history.

## 1. Get set up

1. Ask the fork owner to add you as a collaborator on `github.com/ZhenmanShen/human-or-ai-submissions`.
2. Clone the fork and branch off `main`, using your own git identity so the history shows who did what:

   ```bash
   git clone git@github.com:ZhenmanShen/human-or-ai-submissions.git
   cd human-or-ai-submissions
   git checkout main
   git checkout -b model/<your-name>-<model>
   git config user.name  "Your Name"
   git config user.email "you@cornell.edu"
   ```

3. Python 3.11: `pip install -r submissions/<team>/classical/requirements.txt`.
4. Download `train.csv`, `test.csv`, `unlabeled.csv`, `sample_submission.csv` from the Kaggle Data tab into
   `submissions/<team>/data/` (git-ignored), or anywhere with `--data-dir PATH` / `HOA_DATA_DIR`.
5. Check you have the same validation split as everyone else (the script compares against the committed file):

   ```bash
   cd submissions/<team>
   python common/make_split.py        # must print "OK: val_ids.csv reproduced exactly"
   ```

## 2. Write the model

```bash
cd classical
cp models/_template.py models/<model>.py
cp configs/template.yaml configs/<model>.yaml
```

- In `models/<model>.py`, rename the class and implement `fit(texts, y, unlabeled_texts=None)` and
  `predict_proba(texts)` (P(AI) in [0, 1]). The defaults for `save`/`load` pickle the object; override
  them if your model holds torch tensors (see `models/transformer_scratch.py`).
- Add one line to `REGISTRY` in `models/__init__.py`: `"<model>": "<model>:<ClassName>",`.
- In `configs/<model>.yaml` set `model: <model>`, the `params:` your `__init__` takes, and optionally a
  `search:` grid (tried with 5-fold CV inside the 8,000 training rows; the best mean AUROC wins).

## 3. Run it

```bash
python train.py --model <model>              # search (if any) -> fit on 8,000 -> score the fixed 2,000
python train.py --model <model> --final      # only for a Kaggle file: refit on 10,000 rows (new model: --letter <X>)
python train.py --model <model> --cv topic   # optional: inner CV that holds out whole topics (domain shift)
```

The first command writes `results/<model>/metrics.json`, `val_pred.csv` and `search.csv`, and adds or updates
your model's row in `submission_log.csv` / `SUBMISSIONS.md` ("not submitted"). The second writes
`kaggle_submissions/final_<X>_<model>.csv` (already validated) and adds a `final_<X>_<model>` row marked
"pending upload". `<X>` is the model's letter (D stylo_gbm, E tfidf_lr, F transformer_scratch; a new model takes the next free one).

## 4. Commit in steps, push, open a PR inside the fork

Use these message prefixes (graders read the history):

1. `feat(<model>): add model` - the model file, config and registry line, no results.
2. `exp(<model>): fixed-split validation + hyperparameter search` - `results/<model>/`, `submission_log.csv`,
   `SUBMISSIONS.md`. Put the val AUROC in the message, and what you learned if anything.
3. `sub(<model>): refit on 10k, write kaggle_submissions/final_<X>_<model>.csv` - only if it goes to Kaggle.

Run `git status` before each commit: no data, no `artifacts/`, no weights. Then push your branch and open a
pull request **in the fork** with base `main` (not to the course repo). Someone else on the team reviews and
merges it with `--no-ff`.

## 5. Kaggle uploads

Uploads go through the Kaggle team owner (5 per day for the whole team, so ask first). After an upload, record
the score in a `log(<model>): Kaggle public <score>` commit:

```bash
cd submissions/<team>
python common/update_submission_log.py --track classical --model final_<X>_<model> \
    --file classical/kaggle_submissions/final_<X>_<model>.csv --public <score> --author "Your Name"
```

## Rules

- **No pretrained anything** in `classical/`: no downloaded weights, embeddings or tokenizers. Tokenizers and
  vocabularies are learned here, from the training rows plus `unlabeled_texts` only.
- Never edit `common/val_ids.csv`.
- Never tune on the 2,000 validation rows: choose settings with the `search:` grid (inner CV). The validation
  score is reported, not optimised.
- Never commit competition data, `artifacts/`, model weights or Kaggle credentials.
- Fix the seed (train.py passes `seed` to your model; default 20261023).

## Open model ideas

Not taken yet, pick one and put your name next to it in your PR:

- char-CNN over raw characters (keeps case, punctuation and line breaks)
- BiLSTM over BPE tokens (train the BPE like `transformer_scratch` does)
- extra trees on the stylometric features (`features/stylometry.py`; random forest is done: `rf_stylo`)
- k-NN on TF-IDF vectors (cosine distance)
- perceptron / averaged perceptron / Pegasos SVM from class, on TF-IDF
- a self-training variant of `tfidf_lr` using `unlabeled_texts`
