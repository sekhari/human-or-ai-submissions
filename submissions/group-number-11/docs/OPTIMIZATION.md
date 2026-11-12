# Two-hour optimization session: November 11, 2026

See the [main README](../README.md) for the final entrypoints and the [baseline notes](BASELINE.md) for the initial controlled experiments. All paths below are relative to the repository root.

The baseline model and `output/submission_public_0.96962.csv` are retained. The public result 0.96962 was supplied by the team. All candidate data/model caches remain local under `artifacts/optimization/`, and experiment summaries are saved under `results/optimization/`.

## First candidate: robust_v1

The initial hypothesis was that eliminating sensitivity to paragraph formatting would improve transfer. We expanded local style features from 160 to 279 dimensions, trained native/canonical/format-augmented tree models, and compared whitespace-canonicalized character classifiers. Both classes received the same label-preserving transformations; IDs never enter features. The original fixed stratified 8,000/2,000 split is retained.

The finite blend grid required native validation AUC >= 0.994, then maximized the average of native AUC and worst transformed AUC. The selected model combined 75% augmented histogram boosting and 25% canonical character LR (C=16). Native validation AUC was 0.995598; transformed AUCs were 0.984877, 0.985047 and 0.985161.

**User-reported public result: 0.95899.** This candidate is rejected in favor of the 0.96962 baseline. Synthetic format robustness did not establish hidden-domain robustness. Public feedback is recorded as model-selection feedback, not hidden individual labels. The old baseline CSV/model are not overwritten.

To reproduce this comparison from the repository root:

```bash
python submissions/group-number-11/src/search_optimization.py --data DATA_DIR
python submissions/group-number-11/src/search_lexical.py --data DATA_DIR
python submissions/group-number-11/src/blend_optimization.py --data DATA_DIR --name robust_v1 --final
python submissions/group-number-11/src/predict_optimized.py \
  --test DATA_DIR/test.csv --sample DATA_DIR/sample_submission.csv \
  --model submissions/group-number-11/artifacts/optimization/robust_v1.joblib \
  --out submissions/group-number-11/output/submission_robust_v1.csv
```

`domain_validation.py` adds three-fold validation by 12 unsupervised lexical topic/style clusters. This is an observable grouping proxy, not known source/generator labels or an independent external dataset. Neither hidden test text nor labels participate in constructing these groups.

## Local frozen language-model features

The next hypothesis is that language predictability can add information beyond source-specific layout. The experiments use generic pretrained GPT-2 language-model weights locally, without external authorship labels or a remote prediction API. Supervised classifiers still use only the supplied training labels. No competition passage is sent to a hosted model endpoint.

The final environment is recorded in `requirements-optimization.txt`; it adds PyTorch/Transformers and CatBoost. The original baseline environment remains documented by `requirements.txt` and `environment.txt`.

Pinned weight sources:

- `openai-community/gpt2`, revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`.
- `openai-community/gpt2-medium`, revision `6dcaa7a952f72f9298047fd5137cd6e4f05f41da`.

`lm_features.py` extracts NLL distribution statistics and token-rank fractions from a frozen model. The active runs use at most 256 tokens per passage and fixed padding buckets. A prior 512-token attempt reached the laptop MPS memory limit and produced no complete cache or model result. Incremental checkpoints were added so a restart preserves completed passages. Cached features are checked against the exact supplied data hash before inference.

```bash
python submissions/group-number-11/src/lm_features.py --data DATA_DIR \
  --output submissions/group-number-11/artifacts/optimization/lm \
  --model-id openai-community/gpt2 \
  --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e \
  --max-tokens 256 --datasets train test
python submissions/group-number-11/src/search_language_models.py --data DATA_DIR \
  --lm-dirs lm --name language_small
```

Native-format LM features must be recomputed on transformed passages before claiming full-pipeline format robustness. Scripts flag style-only stress tests that keep native LM features fixed. Canonicalized LM extraction is invariant to whitespace transformations by construction.


## Confirmed improved candidate: language_v2

The team reported **0.98665 public AUROC** for `submission_language_v2.csv`, compared with 0.96962 for the first rebuild. The submission SHA256 is `426b463d3d9789aa1b0159f1fecb45c0603721e06be61790f633a2898da68515`. The matched controlled validation AUC is 0.999785, with 5 FP and 13 FN at threshold 0.5. A conditional 1,000-replicate bootstrap interval is [0.999595, 0.999924]; it excludes selection and distribution-shift uncertainty.

The selected model is 75% LightGBM on 279 style features + 26 frozen GPT-2 features, and 25% of the earlier ensemble. It uses paired original/collapsed style views, 31 leaves and 700 trees. In those feature-level augmentations, LM features retain original text, so synthetic style-view scores are not full end-to-end LM stress results.

After the negative public result, the next finite selection grid required native AUC >= 0.9975 and at least 25% baseline weight. It maximized .75 times AUC among texts without blank lines plus .25 times overall AUC. Plain and label-free within-format conditional-rank blending were compared at baseline weights .25/.5/.75; **plain blending won**. No individual hidden labels were accessed. Public feedback was used to reject robust_v1 and confirm the new language-feature candidate.

`reproduce_optimized.sh` provides a standalone training/feature extraction/search/refit/predict entrypoint. The baseline versions remain intact. The user-reported best and candidate checksums are in `results/optimization/current_best.json` and the candidate delivery manifests.


## Locked final result and handoff

The team reported **0.98861** public for the 10% RoBERTa logit hybrid, followed by **0.98921** for the final medium-language blend. Optimization stopped at the user's request. Final validation AUC is **0.999822**. The selected outer blend is 75% neural_v3 / 25% language_medium in logit space. The medium detector uses 104 small/medium LM statistics and contrasts, 279 style features, native-view LightGBM, and label-free within-format quantile alignment. Its small and medium contexts are 256 and 128 tokens respectively; the contrasts are learned statistics, not a same-context perplexity ratio.

RoBERTa uses generic `FacebookAI/roberta-base` revision `e2da8e2f811d1448a5b465c236feacd80ffbac7b`. Embeddings/lower eight blocks are frozen; upper four/head train only on the original 8,000 rows. AdamW rates are 2e-5/1e-4, weight decay .01, batch 16, random 192-token crops and seed 20261111. Validation selected step 750 from the budgeted run, which stopped at step 861. Reproduction stops at step 750 while retaining the original 1,000-step learning-rate schedule. Inference averages the first and last windows.

Final standard threshold-0.5 confusion counts are TN=996, FP=4, FN=12, TP=988. For five actual FP and five actual FN examples, the report declares review threshold 0.40 (FP=6, FN=10); Kaggle still receives continuous scores. Training AUC 1.0 is exactly certified by monotone clipped-logit bounds: every positive lower bound exceeds every negative upper bound. This is recorded transparently in `final_analysis.json` rather than represented as a full neural training-inference pass.

`predict_final.py` uses recorded weights without reading validation labels; it exactly regenerated the scored CSV SHA256 `ac75c035ad713459781168a1f475ef14bedff26e59657742f438391c70ddeef6`. `verify_final.py` checks its identity and validation provenance. `reproduce_final.sh` recreates all trained components/caches and the final report. The curated materials ZIP excludes competition data and checkpoints. Private score remains unknown.
