# Data diligence before scoring — November 5, 2026

The user requested a fresh start once the real CSVs were available. This implementation is based on the official competition download, not the previous synthetic fixtures. The notebook ideas are hypotheses, not demonstrated AI signatures.

## Verified source and schema

Downloaded through the user's authenticated Chrome session from the competition Data tab:
https://www.kaggle.com/competitions/cornell-tech-cs-5785-combined-2026-ai-vs-human-text/data

| File | Rows | Columns | Checks |
| --- | ---: | --- | --- |
| train.csv | 10,000 | id,text,label | 5,000/class; IDs unique; labels 0 human / 1 AI |
| unlabeled.csv | 3,000 | id,text | Unique IDs; no authorship labels |
| test.csv | 3,000 | id,text | Unique IDs; no authorship labels |
| sample_submission.csv | 3,000 | id,label | IDs/order exactly match test; dummy labels all 0.5 |
| train_starter_2000.csv | 2,000 | id,text,label | Exact subset of train, 1,000/class; not additional data |

No blank fields or exact, whitespace-normalized, or casefold-normalized duplicate passages were found within/between the main files. File hashes are recorded with the audit. Training passages contain 75–350 whitespace words, median 206, mean 208.7087. Overall human and AI means are 209.045 and 208.372 respectively; length alone is not a compelling explanation.

## Split and possible shared sources

Use the fixed stratified split with seed 20261023, yielding 8,000 training rows and 2,000 validation rows (balanced classes). Freeze ordered IDs and input-file fingerprints before fitting any classifier. Learned vocabulary, IDF, scaling and authorship-conditioned statistics use only training rows.

A full training-corpus exact Jaccard audit of lowercase word-trigram sets found 21 pairs at similarity ≥0.50 and **zero at ≥0.70**, including zero at the more conventional near-copy thresholds 0.90/0.95. The maximum was about 0.686, and several high pairs are recipe templates. Similarity alone does not establish a shared source.

Twelve groups (26 rows) share their first 80 whitespace-normalized characters. Five groups cross the fixed split; seven ≥0.50 trigram pairs cross it. There are no supplied source/domain/generator labels, so complete source isolation cannot be certified. The prescribed random split is retained, with a secondary sensitivity score excluding validation rows with these prefix/moderate-overlap indicators. Primary model selection and all required curves use the unchanged 2,000-row holdout; sensitivity scores do not select hyperparameters.

## What the training text supports testing

Training examples cover recipes, encyclopedia/news prose, scientific abstracts and informal posts. IDs 2191/6434 illustrate recipe formatting, 6689 news and 7794 a scientific abstract. AI examples include heavy repetition as well as fluent prose and informal/error-filled text. Those observations contradict a universal “AI is polished” assumption.

Training-only analysis on the same provisional/final 8,000 rows found weak differences in adjective fraction (Cohen's d about +0.14), a *lower* AI superlative fraction (d about −0.12), lower AI first-person fraction (d about −0.18), and a tiny positive-word lexicon effect (d about +0.06). These are exploratory marginal effects, not causal attribution or general rules. Sentence-length and formatting/repetition features have larger training differences. For example, repeated word-trigram fraction above 0.30 occurred in 3/4,000 human and 301/4,000 AI training passages. The full measured feature audit records definitions, directions, effect sizes and length-bin checks.

Accordingly, compare lexical baselines first and test structure features as fixed ablations. A next-POS statistical surprise/mismatch model is a small first test of the notebook's sequence idea; it is neither a grammar-error detector nor the proposed neural next-word model. Semantic claims about “unnecessary” adjectives or poetry remain unimplemented because the supplied data has no such annotations.

Discovery transparency: root's eight sampled labeled passages all belong to the frozen training partition. One collaborator's initial schema preview printed ID 3 before applying the split; that one row later falls in validation and was not used to design features or tune a model. Quantitative feature exploration uses training rows only. Dataset-wide schema/duplicate/length audits inspect metadata and text for integrity, not validation errors or test labels.

Measurement definitions: corpus lengths above are whitespace word counts. The feature audit uses a regex word tokenizer that splits hyphens and can count markup tokens; its counts and length-tertile edges differ. Sentence features use approximate punctuation/newline segments. POS ratios include punctuation tags. Repetition is the fraction of n-gram positions beyond each n-gram's first occurrence, not a proportion of documents. Formatting can encode source artifacts (for example HTML breaks), so strong classification alone does not establish a general writing-style explanation.
