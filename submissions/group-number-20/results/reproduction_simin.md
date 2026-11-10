# Reproduction check (Simin Xu, sx333)

I re-ran both final pipelines on my own laptop from commit `bd6486f`, following the README.
Runs: 2026-11-09. Comparison with the files uploaded to Kaggle: 2026-11-10.

**Result:** both pipelines reproduce. Validation AUROC, CI, perturbed AUROC and all 2,000
validation probabilities match the committed results. The two submission files match the files
uploaded to Kaggle up to about 1e-9 per row, with an identical ranking of all 3,000 test rows,
so their AUROC on any labels is identical.

## Machine and environment

| Item | Value |
|---|---|
| Computer | MacBook Pro (Mac16,1), Apple M4 (10 cores), 16 GB RAM |
| OS | macOS 26.6.2 (25G83) |
| Python | 3.13.5, fresh `venv`, pip 25.1.1 |
| Pinned packages | pandas 3.0.5, scipy 1.18.1, scikit-learn 1.9.1, matplotlib 3.11.2 |
| Unpinned packages (resolved) | numpy 2.5.4, joblib 1.6.0 |
| Data | Kaggle competition zip; train 10,000 rows (5,000 / 5,000), unlabeled 3,000, test 3,000 |

`pip install -r requirements.txt` worked without changes. No environment fixes were needed.

## Commands

Run from `submissions/group-number-20/`:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Final 1 (train.py was run twice; both runs printed identical results)
python src/train.py --train data/train.csv
python src/predict.py --test data/test.csv --out submission.csv

# Final 2
python src/train.py --no-format --dist-weight 0.5 --model-out model_v2.joblib --metrics-out results/final_v2.json
python src/predict.py --model model_v2.joblib --out submission_v2.csv

# Final 2 again with a fixed, different hash seed (outputs go to data/, not committed)
PYTHONHASHSEED=1 python src/train.py --no-format --dist-weight 0.5 --model-out data/model_v2_hash1.joblib --metrics-out data/final_v2_hash1.json
python src/predict.py --model data/model_v2_hash1.joblib --out data/submission_v2_hash1.csv
```

The README commands overwrite `results/final.json` and `results/final_v2.json`. I copied my
re-run files to `data/repro_final.json` and `data/repro_final_v2.json`, then restored the
committed versions with `git checkout`, so the committed files still hold the original run.
Comparisons used `src/check_repro.py`:

```bash
python src/check_repro.py --metrics results/final.json data/repro_final.json
python src/check_repro.py --metrics results/final_v2.json data/repro_final_v2.json
python src/check_repro.py --metrics data/repro_final_v2.json data/final_v2_hash1.json
python src/check_repro.py --sub ../../../submission.csv submission.csv        # original from Zhijie, kept outside the repo
python src/check_repro.py --sub ../../../submission_v2.csv submission_v2.csv
python src/check_repro.py --sub submission_v2.csv data/submission_v2_hash1.csv
```

## Validation results

| Metric | Final 1 committed | Final 1 my run | Final 2 committed | Final 2 my run |
|---|---:|---:|---:|---:|
| Train AUROC | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Validation AUROC | 0.995507 | 0.995507 | 0.986859 | 0.986859 |
| 95% bootstrap CI | 0.992866–0.997621 | same | 0.982770–0.990736 | same |
| Perturbed-format AUROC | 0.914223 | 0.914223 | 0.985986 | 0.985986 |
| Validation probabilities equal | – | 2,000 / 2,000 | – | 2,000 / 2,000 |
| Fit time on 8,000 rows | 46.7 s | 14.3 s (14.1 s first run) | 35.5 s | 20.3 s |

## Submission files

| Comparison | Rows | Exactly equal | Max abs. difference | Same ranking |
|---|---:|---:|---:|---|
| `submission.csv`: Kaggle upload vs my run | 3,000 | 0 | 2.1e-9 | yes |
| `submission_v2.csv`: Kaggle upload vs my run | 3,000 | 0 | 1.1e-9 | yes |
| `submission_v2.csv`: my run vs `PYTHONHASHSEED=1` | 3,000 | 3,000 | 0 | yes |

Both files have columns `id,label`, the same 3,000 ids as `sample_submission.csv` in the same
order, no missing values, and probabilities in [0, 1].

## Notes

1. **Precision of the validation check.** `train.py` stores validation probabilities rounded to
   6 decimals, so "2,000 / 2,000 equal" means equal to 6 decimals. Differences of about 1e-9, as
   in the test files, would not show up there.
2. **Why the Kaggle files differ by about 1e-9.** On my laptop, two runs with different hash seeds
   give bit-identical test predictions, so the pipeline has no run-to-run randomness. The small
   differences from the uploaded files appear only across machines. They are most likely
   floating-point differences in the numerical libraries and CPU. AUROC depends only on the
   ranking, and the ranking is identical, so these differences cannot change the Kaggle score.
3. **Hash seed.** `PYTHONHASHSEED` was unset in the normal runs, so Python used a random hash seed
   each time. The fixed-seed run matched exactly. This confirms that commit `44a6404` (ties broken
   alphabetically when ranking frequent words) made the masked-word model deterministic.
4. **`dist_weight` field.** The committed `final.json` has no `dist_weight` key because it was
   written before the `--dist-weight` option was added (`b699dc5`). The current code with the
   default weight 0 reproduces it exactly.
5. **Timing.** `fit_seconds` covers only the fit on 8,000 rows. On one machine, Final 2 (20.3 s)
   is slower than Final 1 (14.3 s), as expected because it fits one more model. The cloud
   timings in the report (35.5 s vs 46.7 s) show the opposite order, so those timings are noisy.
6. **Unpinned packages.** `numpy` and `joblib` are not pinned in `requirements.txt`. The versions
   I used are listed above.

## AI assistance

I ran the setup, training and prediction commands myself in my terminal and got the two
uploaded submission files from Zhijie. Claude (Claude Code, desktop app) explained each step,
ran the comparisons, wrote `src/check_repro.py` and drafted this file in English at my request.
