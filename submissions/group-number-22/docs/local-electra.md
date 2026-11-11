# Local ELECTRA

Issue #16 fine-tunes `google/electra-small-discriminator` at revision
`fa8239aadc095e9164941d05878b98afe9b953c3` (Apache 2.0, Hugging Face Hub) in
notebook 16. Weights download once into ignored `artifacts/`, then load with
`local_files_only=True`. No remote inference.

## Env

Optional isolated env in `envs/electra`: torch 2.15.0, transformers 5.22.2,
huggingface-hub 2.4.0, tokenizers 0.23.3, safetensors 0.8.0, plus the root core
pins. The root `uv.lock` stays torch-free. Run:

```bash
uv sync --project envs/electra --locked
uv run --project envs/electra --locked jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=7200 notebooks/16_local_electra.ipynb
```

Device was Apple M5 MPS. MPS is not bitwise deterministic. We did not rerun to
measure the drift.

## Settings

Max 256 tokens, batch 16, AdamW lr 3e-5, weight decay .01 (none on bias and
LayerNorm), linear warmup over 10% of a three-epoch step count, then linear decay.
No grad clipping. Seed 42 for head init and shuffle. Score is logit1 minus logit0.

## Selection

Internal 6400/1600 inside original 8000, same split as #12 seed 42.

| Epoch | Train loss | Internal AUROC | Train seconds |
|---:|---:|---:|---:|
| 1 | 0.385750 | 0.977459 | 75.8 |
| 2 | 0.114606 | 0.980533 | 74.2 |
| 3 | 0.054574 | 0.983039 | 74.1 |

Epoch 3 wins. The pick was saved before frozen 2000 was scored.

## Confirm

A fresh model trained on original 8000 for 3 epochs (1500 steps, schedule sized
for 3 epochs). Training took 278.4 s. Frozen 2000 was scored once.

| Stress | Frozen 2000 AUROC |
|---|---:|
| original | 0.987481 |
| remove_blank_lines | 0.987481 |
| collapse_whitespace | 0.987481 |
| wrap80 | 0.987480 |

This is below the #15 classical blend (.998492 frozen). But formatting barely
moves it, because the WordPiece tokenizer drops whitespace and newlines. #12
measured classical soft losses of .082 to .103 under whitespace collapse.

## Blend

Not run. It waits for #17. The fixed plan is neural weights 0, .1, .25, .5, .75, 1
on standardized scores with an eligible-unlabeled reference. The objective is the
mean AUROC over the four stresses, with the guardrail original >= classical - .002.
Margins on frozen 2000 (all stresses), eligible unlabeled, and test are saved in
ignored artifacts for that step. No upload.

Record: `results/16_local_electra/20261110T223101338301Z.json`. Total runtime 593.6 s.
