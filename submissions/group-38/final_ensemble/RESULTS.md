# November 9 model improvement experiment

**Team selection (November 9): the user confirmed submission_recommended.csv (including BERT), reported public AUROC approximately 0.97, and selected this ensemble as the final model. Exact leaderboard precision and private score are unknown. This supersedes the earlier 0.956 attribution.**

Evaluated file: **submission_recommended.csv**. This final version includes the fully refit compact BERT; the earlier submission_cnn_blend.csv was an intermediate candidate.

| Model | Fixed validation AUROC (2,000 passages) | Test-like quartile AUROC (500 passages) |
|---|---:|---:|
| Previous ensemble | 0.997763 | 0.994205 |
| Previous ensemble + character CNN | 0.998547 | 0.997591 |
| Final BERT/CNN/tree/previous-model ensemble | **0.999393** | **0.999478** |

Final weights: previous model 28%, character CNN 28%, enriched-feature trees 24% (equal average of XGBoost depth 3, XGBoost depth 5, CatBoost depth 6), compact BERT 20%.

BERT's own validation AUROC is 0.979705. The selected four training epochs were then repeated from the pretrained base using all 10,000 supplied labeled examples. The character CNN used six selected epochs and was also refit on all supplied labels. Final tree iteration counts were selected on validation, then fixed during full-label fitting.

The earlier user-reported public score was approximately 0.96. **The new public score is approximately 0.97 (user reported; exact precision not provided); the private score is unknown. This experiment has not established a Kaggle AUROC of 0.99.** Validation improvement cannot be directly extrapolated to hidden test performance. Multiple models and weights were selected on the same fixed split; the test-like subset is part of that split and also used for selection.

Negative results: style-kernel SVM did not outperform the previous model; neither pseudo-label weight (.2/.5) improved both checks over its parent. These models are excluded. The CatBoost depth-8 pilot was interrupted for high computation cost and has no completed score.

The auxiliary test-likeness model uses test text without authorship labels. No hidden labels or external human-vs-AI labeled dataset were accessed. External base BERT weights and the NLTK POS tagger are disclosed in code/README.md and the provenance record.

Verification: 3,000 unique test IDs, official sample order, exactly id/label columns, finite values in [0,1], 3,001 lines including header. The portable finalizer reproduced the delivered CSV byte for byte.

SHA-256: `ce5ad51331883b22add000923e606a6cee1abdd87fc5fc112045a77060b05753`.

Training and experiment records are included. Raw competition texts and trained model weight files are kept local.
