# Error review (Jin)

Notes for Q4.3 / Q4.4. I read the full text of the top error cases in `results/q4_errors.json`
(8 FP + 8 FN, Final 1) and `results/q4_errors_v2.json` (8 FP + 8 FN, Final 2), all on the fixed
validation set, threshold 0.5. "Seen" = what is in the text or the numbers.
"Guess" = our explanation, not tested.

## 1. Blank-line check

Script: `src/check_blank_lines.py`.

| rule | rows | human | AI |
|---|---:|---:|---:|
| raw `\n\n` (same rule as the pipeline and `load_train`) | 3,301 | 0 | 3,301 |
| also count Windows blank lines (`\r\n\r\n` -> `\n\n`) | 3,314 | 3 | 3,311 |
| also count lines with only spaces/tabs (`\n[ \t]*\n`) | 3,322 | 17 | 3,305 |

- Seen: the report number (3,301, all AI) is right for the rule the pipeline uses.
- Seen: 17 rows write blank lines as `\r\n\r\n` (14 AI, 3 human), so the raw rule misses them.
  The 3 human ones (3919, 7376, 8838) are all recipes: an ingredient list, then a blank line
  before the steps or before "Bon Appetit!". None of them is in the validation set.
- So "blank line = AI" is almost true in train, but not fully. Human recipe writers do use blank lines.
- About 66% of AI texts have a blank line (3,311 / 5,000), and only 3 of 5,000 human texts.
- Share of texts with a blank line (`\r\n` counted): train 33.1%, unlabeled 34.4%, test 31.9%.
  The test set does not lose blank lines.
- From `final.json` / `final_v2.json`: with formatting perturbed, Final 1 drops from 0.9955 to
  0.9142 on validation; Final 2 only goes from 0.9869 to 0.9860.

Why it helps on validation but almost not on public:
- Seen: validation is cut from the same data as train, so the rule holds there too
  (641 validation texts have a blank line, all AI).
- Seen: on validation, format features add 0.0145 (0.9955 vs 0.9810 without them).
  On public they add 0.002 (0.9205 vs 0.9185).
- Seen: test has about the same share of blank lines, so the signal is not missing.
- Guess: in test, blank lines are not an AI-only mark. The 3 human recipes in train show that
  human texts from some sources use them too. We cannot check this without test labels.

## 2. Final 1: false positives (human, predicted AI)

| id | p | words | what I see | possible reason |
|---|---:|---:|---|---|
| 3050 | 0.836 | 205 | Whole text is in Greek | Guess: model is trained on English; unfamiliar words and characters |
| 4470 | 0.772 | 228 | Wikipedia-style article on Hittite food; neutral tone; "Bibliography" heading glued into a line | Guess: formal encyclopedic style looks like AI to the model |
| 6467 | 0.717 | 168 | Wikipedia-style article; section title glued to end of a sentence | Same as 4470 |
| 2158 | 0.640 | 110 | Short movie review with strong opinions | Unclear. Guess: short text (0-150 words is 66% AI in validation) |
| 483 | 0.556 | 218 | Recipe: ingredient list, then casual comments | Near 0.5, unclear |
| 5816 | 0.554 | 177 | Math paper abstract, lines hard-wrapped at ~80 chars | Guess: abstracts are in both classes (see FN 6978) |
| 2053 | 0.551 | 219 | BBC news on Bafta awards, list of winners at the end | Near 0.5, unclear |
| 2298 | 0.538 | 112 | Short personal comment, non-native English | Near 0.5, unclear. Short text |

## 3. Final 1: false negatives (AI, predicted human)

| id | p | words | what I see | possible reason |
|---|---:|---:|---|---|
| 4519 | 0.044 | 141 | Forum post; lowercase "i"; no space after periods ("point.i do"); odd content | Guess: casual forum surface looks human; model does not check meaning |
| 679 | 0.062 | 205 | Has blank lines; made-up place facts (Hay Swamp, Hay County) | Seen: blank lines did not save it. Guess: wiki style hides content errors |
| 44 | 0.082 | 242 | Has blank lines; first-person tech forum post that does not make sense | Same as 679 |
| 8444 | 0.137 | 327 | Wikipedia-style biography with accents (n with tilde etc.) and en dash | Guess: copies Wikipedia style closely |
| 6978 | 0.155 | 180 | Paper abstract wrapped in double quotes | Guess: abstracts are in both classes |
| 7294 | 0.161 | 156 | Medical forum post; tense mix-up ("came into class tomorrow"); "(not so bad)and" | Guess: casual style and typos look human |
| 714 | 0.163 | 295 | Gaming blog; one sentence repeated almost word for word; "suicide?I think" | Seen: repetition. Guess: missing spaces look human |
| 1362 | 0.188 | 233 | Wikipedia-style bio with broken encoding and a [1] footnote | Guess: looks like a scraped human page |

## 4. Patterns (Final 1)

Seen:
- 2 of 8 FN (44, 679) have blank lines, so Final 1 does not only follow the blank-line rule.
- 3 of 8 FN have missing spaces after punctuation ("point.i", "(not so bad)and", "suicide?I").
- Many FN do not make sense (made-up facts, repeated sentences, wrong tense) but read fluently.
- FP are mostly formal or unusual human texts: Wikipedia-style (2), abstract (1), news (1), Greek (1).
- 5 of 8 FP are close to 0.5 (0.53-0.64). All 8 FN are far from it (below 0.19).

Guess:
- The model learns surface style (tone, punctuation, layout), not meaning. Fluent nonsense
  passes as human, and formal human writing looks like AI.
- Missing spaces may come from line breaks removed when the data was built.

## 5. Final 2 vs Final 1

Script: `src/exp_errors.py --results results/final_v2.json --out results/q4_errors_v2.json`.

| group | n | Final 1 AUROC | Final 2 AUROC |
|---|---:|---:|---:|
| 0-150 words | 375 | 0.9939 | 0.9849 |
| 150-200 | 591 | 0.9960 | 0.9817 |
| 200-250 | 492 | 0.9930 | 0.9862 |
| 250-400 | 542 | 0.9988 | 0.9938 |
| no line breaks | 642 | 0.9928 | 0.9925 |
| single breaks only | 717 | 0.9922 | 0.9928 |
| has blank line (all AI) | 641 | - | - |

At threshold 0.5: accuracy 0.9775 vs 0.937, FP 15 vs 14, FN 30 vs 112.

| blank-line group (641, all AI) | Final 1 | Final 2 |
|---|---:|---:|
| mean p | 0.877 | 0.801 |
| predicted AI | 98.4% | 87.4% |
| errors | 10 | 81 |

Seen:
- Final 2 is lower in every length bin, but in the two groups without blank lines the two
  models are the same (0.9928 vs 0.9925, 0.9922 vs 0.9928). The validation gap comes from the
  blank-line texts.
- 81 of Final 2's 126 errors are AI texts with blank lines that it calls human.
- Error overlap: 20 shared errors (FP: 2 shared, FN: 18 shared). corr(p1, p2) = 0.961.
- Final 2 top FN: 7 of 8 have blank lines (3581, 679, 5856, 6487, 3504, 9971, 1487): recipes with
  numbered steps, Reuters-style news, a short story, paper-style text.
- Final 2 top FP: 4 of 8 are paper abstracts hard-wrapped at ~80 chars (5816, 1329, 2983, 5199),
  2 Wikipedia-style bios (3671, 7797), 1 movie review (139), 1 BBC news (9487).

Guess:
- Without the blank-line hint, the text features find many AI texts human-like, so Final 1's
  validation lead mostly measures the blank-line shortcut, not better text understanding.
- Hard-wrapped lines (single breaks) are mostly human in train (8.9% AI), yet these abstracts
  still look AI to Final 2 because of their formal academic wording.
- FP/FN counts at 0.5 also depend on calibration; the blend with the masked model may move
  probabilities. AUROC does not depend on the threshold.
