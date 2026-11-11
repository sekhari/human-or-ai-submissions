> Historical record of the earlier four-method recipe. Use README.md for the current submission and commands.

> **Current accepted recipe:** use [TEAM_HANDOFF.md](TEAM_HANDOFF.md) and `robust_validation/current_recipe.json`. The default inference weights are now `[0.35, 0.234, 0.156, 0.26]`, reported at public AUROC 0.97960. The report below records the earlier `[0.375, 0.225, 0.15, 0.25]` recipe, its fitted artifact, and its validation results. Historical hashes and checks describe that original artifact, not a new training run.

# Recorded four-method baseline validation

Open **Four_Method_Robust_Colab.ipynb** in Colab, select a **T4 GPU**, and Run all. Upload the original competition ZIP when prompted. The notebook is self-contained and needs no repository checkout. Save its prediction CSV and fitted model bundle before the runtime expires. Nothing is submitted to Kaggle.

This adds a tested candidate alongside the committed pipeline. It does not replace the team's main recipe or establish a better Kaggle score. The generating code for the team's latest F submissions was not available in the reviewed snapshot, so the control here is the committed three-way pipeline, refitted on identical training rows.

## Fixed recipe

| Method | Weight | What it contributes |
|---|---:|---|
| Original lexical/style reference | 0.375 | Word/character TF-IDF logistic regression plus style CatBoost; preserves the team's successful reference |
| Dual-language-model CatBoost | 0.225 | 15 predictability features each from GPT-2 and GPT-2 Large, plus their 15 differences |
| DistilRoBERTa | 0.150 | Existing fine-tuned semantic classifier |
| DeBERTa-v3-base | 0.250 | A different fine-tuned architecture, with head/tail passage coverage |

Both neural classifiers use two epochs and cleaned text. DeBERTa freezes vocabulary embeddings and uses gradient accumulation in full precision. GPT-2 Large feature extraction ran successfully on a T4 in full precision: 10,000 passages in 33.7 minutes, peak allocated GPU memory 6.32 GiB.

Component scales use quantiles fitted on 800 held-out **training** rows, pooling original, flattened, and short-line views. They are frozen at inference: a new file does not redefine ranks or ensemble weights. Quantiles are scores, not calibrated authorship probabilities.

Final fitting uses 9,200 supplied labeled rows, reserving 800 for calibration. No external labeled text, pseudo-labels, or competition test labels are used. Pretrained general-purpose model weights are downloaded publicly. The supplied unlabeled CSV is unused.

## Local evidence

| Frozen global audit, 2,000 rows | Committed recipe control | Candidate |
|---|---:|---:|
| Original text | 0.999212 | **0.999603** |
| Flattened whitespace | 0.980368 | **0.994209** |
| Synthetic short lines | 0.987352 | **0.996970** |

Both models fit the same 6,400 rows; 800 separate rows fit calibration and 800 select the recipe. The recipe was selected across the selection split plus two content-held-out development groups (conversational text and recipes). Each component was fitted without its held-out content group. Candidates had to retain the control on all nine development scenarios. Among eligible candidates, selection maximized mean native content-held-out AUC, then worst formatting AUC. The chosen recipe used 45 dual-LM features and the fixed weights above.

Original content-held-out development AUROC increased from 0.996467 to 0.997975 for conversational text and from 0.994665 to 0.997803 for recipes. These are **selection results**, not untouched test estimates. See `robust_validation/recipe.json` for every scenario, split hash, and the 243 searched combinations.

A stratified paired bootstrap of the fixed global audit (1,000 replicates) gives an original-text AUC difference of +0.000391, conditional 95% interval [+0.000016, +0.000765]. Formatting gains have wider absolute improvements. These intervals do not account for prior experimentation, source dependence, or distribution shift. This audit seed has been used in the project before; it is not a pristine independent benchmark. Do not keep tuning against these audit results.

The production refactor reproduces all three cached audit AUCs, with maximum score differences below 2.4e-9 on the local check. Normalization changes preserve all 10,000 original cleaned inputs and make all whitespace stress views identical for neural tokenization, including a literal HTML line-break edge case.

A further news-like group was evaluated after freezing the recipe: 4,830 fitting rows, 800 calibration rows, 770 held-out rows (377 human, 393 AI). Native AUC rose from 0.994465 to **0.996504**; flattened AUC from 0.952903 to 0.979597; short-line AUC from 0.930056 to 0.983828. No weights were changed after this check. See `robust_validation/frozen_news_check.json`. It is another within-dataset transfer check, not an external benchmark.

Synthetic wrapping tests do not prove performance on real human poetry or unknown generators. A high local AUC cannot guarantee private leaderboard performance, and the model has not been submitted to Kaggle. Preserve the existing pipeline as a comparison until the team evaluates the candidate.

## Command line

Use Python 3.11+ and an existing CUDA PyTorch installation (tested locally with 2.6 and on Colab with 2.14.1+cu130). Install the additional dependencies:

```bash
python -m pip install -r requirements-robust.txt
USE_TF=0 USE_FLAX=0 python train_robust.py --data data --output final_model
USE_TF=0 USE_FLAX=0 python check_robust.py --model final_model --input data/test.csv --recipe baseline
USE_TF=0 USE_FLAX=0 python predict_robust.py --model final_model --input data/test.csv --output submission_four_methods.csv --recipe baseline
```

`data/` should contain the original train/test CSVs. Fresh training takes approximately 1–2 hours on T4 including feature extraction and prediction, plus downloads. CPU/Mac device fallback is available, but full training needs sufficient RAM and can be substantially slower. Use the tested T4 route for this project; the full recipe has not been verified to fit an 8 GB Mac. Existing feature caches can be supplied with `--small-features` and `--large-features`; dataset digest, model metadata, and matrix shape are checked. Completed encoder checkpoints resume only within the same source/configuration manifest. Changing code requires a fresh output folder.

To reproduce the frozen global audit without training on its labels:

```bash
USE_TF=0 USE_FLAX=0 python train_robust.py --data data --output audit_model --validation
USE_TF=0 USE_FLAX=0 python validate_robust.py --data data --model audit_model
```

The additional frozen news check can be reproduced with `validate_content_robust.py --data data --features audit_model/small_features.npz --large-features audit_model/large_features.npz --recipe robust_validation/recipe.json --output news_check`. It retrains both neural classifiers, excluding that content cluster, and takes additional GPU time. Cached features contain no fitted authorship labels.

Do not audit a final model trained on 9,200 rows; that includes the old audit labels. Fine-tuning can vary numerically between GPU/software versions, so exact score equality is not promised for independent retraining.

## Professor's later evaluation

Keep the fitted bundle intact, including both fine-tuned neural folders, `model.joblib`, and source. Then run `predict_robust.py --model final_model --input new_test.csv --output predictions.csv`. Input needs unique `id` values and nonempty `text`; output contains `id,label` in the same order. No training, threshold selection, or recalibration is performed on the new file. GPU is recommended.

Public pretrained GPT-2 weights download automatically. For disconnected inference, cache the original model/tokenizer files under `gpt2/` and `gpt2-large/`, then pass `--offline-root`. GPT-2 Large is pinned to revision `32b71b12589c2f8d625668d2335a01cac3249519`. The fitted bundle is about 0.8 GiB before optional offline pretrained weights; share it in team storage rather than committing it to GitHub. Load only trusted joblib bundles. The notebook uses `package_robust.py` to verify the export includes model/tokenizer files and package metadata, while excluding data and feature caches.

## Completed final run

The final 9,200-row fit and separate 800-row calibration completed on Tesla T4. Both neural classifiers trained for two epochs. The fit took **33.46 minutes** using the already completed language-feature caches, with **4.62 GiB** peak allocated GPU memory during fitting. This timing excludes initial feature extraction, final checks, test inference, and archive creation.

The fitted final detector passed the full inference batch-size and row-order check: maximum score difference **4.21e-7** on eight passages. It then generated all **3,000** test predictions. Independent local verification confirmed the original test IDs and sample row order, finite scores in [0,1], and the downloaded CSV checksum. The model ZIP also passed SHA-256 and ZIP CRC checks; its serialized classifier, both fine-tuned checkpoints/tokenizers, 45-feature LM classifier, and fixed weights were verified locally.

The ZIP contains 49 files and is 860,134,885 bytes (0.80 GiB), including code and fitted models. Competition CSVs and feature caches are excluded. Its SHA-256 is `4bd236a9432666055408a15307de62a296adc867876d03a926d1a5f138c1dce8`. See `final_training_complete.json`, `final_inference_check.json`, and `local_verification.json` under `robust_validation/` for provenance.

The AUC results above belong to the separate validation fits. The final 9,200-row model has no measured Kaggle/private-test score. Prediction-file integrity checks do not measure classification accuracy.
