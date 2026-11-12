# lm_stack

Language-model likelihood features plus a classifier, and the blend
with Usman's base512.

Pipeline: `normalize.py` -> `extract_features.py` (11 features from
Qwen2.5-0.5B and its Instruct version) -> `train.py` (logistic
regression, gradient boosting) -> `ensemble.py` (rank average with
base512).

Other files: `split.py` (shared 8,000 / 2,000 split), `data_check.py`
(prints character stats), `test_features.py`, `test_normalize.py`.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Put `train.csv`, `unlabeled.csv`, `test.csv` in `./data/` (not
committed). On a CUDA machine, install the matching PyTorch build
first. Tested on Python 3.12.10.

## Run

```bash
python extract_features.py --model-size 0.5B --dtype float32 \
  --batch-size 2 --splits train test unlabeled --out features
python train.py --seed 42
python ensemble.py --npz-dir preds/usman
```

`preds/usman/` needs `base512_main.npz`, `base512_final_test.npz`
and `deberta_main.npz`, made by `../train_deberta.py` (see the usage
lines in its docstring; base512 uses `--tag base512`).
Output: `blend/submission_w0.2.csv`.

Seed 42 everywhere. Plain `train.py` defaults to the older seed
20261023, which is a different validation set.

## Notes

- Pretrained weights: `Qwen/Qwen2.5-0.5B` and
  `Qwen/Qwen2.5-0.5B-Instruct`. No external LLM API was used.
- Known issue: `test_binoculars_matches_reference` misses its 1e-4
  tolerance by about 3e-4 on a GTX 1080 (float32). It passes on an
  Apple M2.

## AI assistance

Used Claude Code to write majority of the code, but did most of the architecture planning using Claude to help generate different model ideas. I picked the final model choice and mapped out the curation pipeline. I reviewed the code, ran the tests, extraction, and training myself, and checked Usman's predictions against his published AUROCs before blending.