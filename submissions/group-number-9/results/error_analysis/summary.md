# Error analysis of `improve_all_blend` (report Q4)

All numbers come from `python -m experiments.error_analysis` (files in this folder) and
`results/improve_lm/`. The model is fit on the 8,000-row training partition and scored on the fixed
2,000-row validation set.

## Q4.1 Train/validation gap

| Split | AUROC |
|---|---:|
| Training, in-sample (8,000) | 1.00000 |
| Training, 5-fold out-of-fold (8,000) | 0.99952 |
| Validation (2,000) | 0.99987 |
| LODO mean over 11 topic domains (train partition) | 0.99921 |
| LODO, worst domain (cluster 8, biographies) | 0.99805 |
| Public Kaggle (900 test rows, final fit on 10,000) | 0.99134 |

The in-sample 1.0 is expected for boosted trees and extra-randomized trees, which can fit every
training row, so it says little. On unseen rows from the same distribution the model is just as
strong: out-of-fold 0.9995 and validation 0.9999 agree within noise, so there is no sign of harmful
overfitting. The gap that matters is between validation and Kaggle (about 0.0085). Plausible reasons:
(1) topic shift: in EDA, clusters 0 (fiction) and 2 (poetry) hold 620 and 486 test rows but only 164
and 25 training rows; (2) the test set contains unfamiliar generators and adversarial edits that our
labeled data does not; (3) validation was reused across many experiments, so its score is optimistic.
Validation is close to saturated, which is why we chose models mainly by LODO.

## Q4.2 Subgroups

Length bins follow the template's suggestion and split the validation set into short, typical and long
passages (word counts by whitespace).

| Subgroup | Count (AI) | AUROC | 95% CI |
|---|---:|---:|---|
| < 150 words | 381 (234) | 0.9996 | 0.9988–1.0000 |
| 150–250 words | 1,033 (448) | 0.9998 | 0.9996–1.0000 |
| > 250 words | 586 (318) | 1.0000 | 0.9999–1.0000 |
| Has a blank line | 673 (669) | 1.0000 | (only 4 human) |
| No blank line | 1,327 (331) | 0.9996 | 0.9992–0.9999 |

Short passages are hardest: they give the language-model statistics fewer tokens to average. The
blank-line split matters because EDA found that a blank line almost always means AI in the training
data. All 11 validation errors (below) have no blank line, so the model is weakest where this shortcut is
absent. By topic, cluster 6 (movie reviews) and cluster 8 (biographies) have the lowest validation
AUROC (0.9996 each), matching cluster 8 being the worst LODO domain. Differences are small and the
intervals overlap, because the validation set is nearly saturated.

## Q4.3 Errors

Errors use the final blend score, the same score submitted to Kaggle, with threshold 0.5. The score
is an average percentile rank rather than a calibrated probability, so 0.5 labels the top-ranked half
of the validation set as AI, matching its 50% AI rate (1,001 of 2,000 passages are labeled AI). This
gives 6 false positives and 5 false negatives, all listed. `P(AI)` is the probability from the blend's
gradient-boosting component, shown only to indicate how confident the error is.

| ID | Type | Score | P(AI) | Possible reason |
|---:|---|---:|---:|---|
| 988 | FP | 0.517 | 0.995 | Polished, balanced film review ("however", structured critique); low GPT-2 surprisal (2.75) |
| 7887 | FP | 0.512 | 0.993 | Greek text; English-centric features and GPT-2 byte tokens make it look predictable |
| 4078 | FP | 0.504 | 0.417 | Wikipedia disambiguation list repeating "Cassa di Risparmio ..., an Italian bank"; surprisal 2.17 |
| 2290 | FP | 0.504 | 0.427 | Encyclopedic military history in a neutral register |
| 4656 | FP | 0.502 | 0.069 | Human Wikipedia biography of an ice-hockey player; surprisal 2.25 |
| 7797 | FP | 0.500 | 0.043 | Human Wikipedia biography of a well-known politician; surprisal 2.24 |
| 783 | FN | 0.487 | 0.001 | AI recipe imitating scraped human formatting (inline ingredient list, double spaces) |
| 3168 | FN | 0.489 | 0.003 | AI math abstract full of LaTeX; markup dominates the token statistics |
| 8692 | FN | 0.490 | 0.314 | AI abstract with a "Title:/Author:" header and a Greek letter in the title |
| 6978 | FN | 0.492 | 0.031 | AI paper abstract wrapped in quotation marks, technical vocabulary |
| 8022 | FN | 0.499 | 0.084 | Fluent AI Wikipedia-style biography with invented facts the model cannot check |

## Q4.4 Patterns

Observed:

- Encyclopedic text is ambiguous in both directions. Four of six false positives are human
  Wikipedia-style passages (a disambiguation list, a military history, two biographies) and one false
  negative is an AI biography. These human passages have some of the lowest GPT-2 surprisal among the
  errors (2.17-2.25). Biographies (cluster 8) are also the worst LODO domain.
- Four of five false negatives are AI text in a structured document format: a recipe, a LaTeX
  abstract, a headed abstract and a quoted abstract.
- Every error lacks a blank line, and short passages have the lowest AUROC.
- Non-English text is unreliable: the five validation passages with more than 10% non-ASCII letters
  have AUROC 0.83 (one Greek false positive above).

Speculation (not tested): GPT-2-family models were trained on web text that includes Wikipedia, so
real encyclopedic passages may look unusually predictable, which the model reads as AI. The
language-model features also assume English prose, so markup and other languages distort them.

A fix for the first pattern: express surprisal relative to the passage's topic, for example as a
z-score within its topic cluster or as the difference from a reference model that has not seen
Wikipedia (such as our n-gram model fit on `unlabeled.csv`), so that a genre that is predictable for
everyone is not taken as evidence of AI. It can be tested with the same LODO protocol on the
encyclopedic clusters.
