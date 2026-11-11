# One-pass self-training

Issue #7 runs C1 and the training-CV-selected C4 word teachers, each with a fresh
matched supervised control. Fit vocabulary and IDF on the original 8,000 verified
training rows only. Freeze them for one classifier refit with eligible unlabeled
rows at confidence at least .95. Hard pseudo labels use threshold .5, verified
weight 1, and pseudo weight .25. Neither confidence nor weights are retuned.

| Teacher C | Accepted human / AI | Pseudo weighted mass | Control validation AUROC | Treatment validation AUROC | Delta |
|---|---:|---:|---:|---:|---:|
| 1 | 0 / 8 | 2.00 | .932728 | .932718 | -.000010 |
| 4 | 61 / 190 | 62.75 | .941252 | .941257 | +.000005 |

Both changes are negligible. The 2,000-draw paired class-stratified bootstrap 95%
intervals are [-.000029, +.000006] for C1 and [-.000067, +.000077] for C4. They
condition on fitted models and reused validation rows, excluding training
variation and selection bias. Unlabeled-row correctness is unknown; confident
pseudo-label mistakes can reinforce the teacher. No model is adopted from this
ablation. Keep the negative C1 result too.

The full fresh-kernel run took 33.011 seconds. Each matched control reproduced its
teacher exactly. Vocabulary/IDF hashes stayed fixed. All six model reload checks
had zero probability difference. The safe run record is
`results/07_pseudo_labels/20261110T185517771685Z.json`. The separate acceptance
record was saved before validation. Per-arm scores are in the record; its top-level
validation AUROC is null because this experiment makes no single model selection.

```bash
uv run --locked jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=900 notebooks/07_pseudo_labels.ipynb
uv run --locked python tools/check_notebooks.py
```

The private pointer `artifacts/07_pseudo_labels/latest.json` names the teachers,
controls, treatments, ID-aligned predictions, source record, and artifact hashes.
Accepted pseudo-label rows and teacher scores remain ignored. The run metadata
counts 8,000 verified fitting rows; acceptance tables separately give each
augmented row count and effective weighted mass. No validation or test row is
pseudo-labeled, and no upload occurs. Teammate review is requested in the PR.
Notebook 15 still depends on teammate issue #13 as well as this experiment.
