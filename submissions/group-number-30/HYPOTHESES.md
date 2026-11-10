# Handwritten ideas: evidence, implementation, and limits

The notes supplied on November 5 are a collection of hypotheses, not established
facts about human or AI writing. Training-data inspection supported testing
structure and repetition, while several adjective, positivity, and pronoun
claims showed little evidence. The implemented experiments establish useful
statistical signal on this validation partition; they do not detect actual
grammatical mistakes, inappropriate wording, poetry, or authenticity.

## What the training data supports

[training_feature_audit.json](results/training_feature_audit.json) records the
exploratory feature audit on 8,000 labeled training passages, with 4,000 examples
per class. Its seed was `20261023`; its training IDs match the subsequently
persisted [split manifest](results/split.json). The JSON includes data/code
checksums, class means and sample standard deviations, Cohen's d, correlations
with length, length-tertile summaries, and example IDs. No validation outcomes
were used to compute these descriptive statistics.

Positive d below means the feature was higher on average in AI passages.
It is a descriptive effect size, not proof of cause or a validated rule.

| Handwritten idea / observable proxy | Human mean | AI mean | Cohen's d | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Adjectives / fraction of POS tokens tagged JJ/JJR/JJS | 0.07853 | 0.08355 | +0.140 | Small association; it does not measure whether adjectives fit their nouns. |
| Superlatives / JJS or RBS fraction | 0.00213 | 0.00168 | −0.120 | Lower in AI passages on average, contrary to a simple “AI uses more superlatives” rule. |
| Adjacent adjectives / adjacent JJ-family tag pairs | 0.00626 | 0.00641 | +0.012 | Essentially no mean difference in this proxy. |
| Personalization / first-person word fraction | 0.02221 | 0.01696 | −0.181 | Higher in human passages; pronoun use is not a semantic measure of personalization. |
| Positive wording / fixed positive-word lexicon fraction | 0.00088 | 0.00114 | +0.058 | Very weak mean difference; the lexicon does not measure sentiment. |
| “Rule of three” / comma-comma-and/or pattern per segment | 0.05250 | 0.05854 | +0.055 | Weak proxy; does not identify semantic three-part rhetoric. |
| Sentence structure / mean words per punctuation/newline segment | 17.356 | 23.358 | +0.287 | More promising descriptive difference, subject to segmentation and domain effects. |
| Repetition / repeated word-trigram positions | 0.02121 | 0.07633 | +0.433 | A substantial upper tail warrants testing; most AI examples are not extreme repeats. |
| Formatting / newlines per 100 regex words | 2.728 | 5.164 | +0.439 | Useful candidate signal that may reflect source formatting or lists. |

The repetition difference is concentrated in the tail: more than 30% repeated
trigram positions occurs in 3/4,000 human passages and 301/4,000 AI passages.
Training IDs `4211`, `9087`, and `3626` repeatedly reproduce a headline. These
are evidence of repetition in this dataset, not a general description of AI text.

There was one incidental exposure during initial schema inspection: a preview
printed IDs `1`, `2`, and `3` before partitioning. ID `3` belongs to validation;
it was excluded from every feature-audit statistic and reported training example.
The audit records this explicitly. No claim of a completely unseen holdout is made.

## What was implemented

All four bonus configurations used logistic regression with `C=1` and reused
the same 8,000/2,000 split. Their original `bonus_plan.json` is preserved in
[day1-evidence.zip](results/day1-evidence.zip). Bonus hyperparameters were not searched.
Each learned vectorizer, scaler, or conditional model was fitted on training
rows only. The models and features are in [models.py](models.py) and
[structure.py](structure.py).

- **Style model:** 28 surface/POS features followed by training-fitted standard
  scaling and logistic regression. These include length, lexical diversity,
  approximate sentence sizes, repetition, punctuation, HTML/newline counts, the
  fixed lexical probes, POS proportions, and POS entropy.
- **POS sequence model:** NLTK Penn Treebank tag sequences represented by
  TF-IDF tag unigrams, bigrams, and trigrams, then logistic regression. This
  directly tests whether distributions of syntactic sequences distinguish the
  supplied labels without retaining the original words in the representation.
- **Conditional POS predictability model:** Separate human and AI POS trigram
  models with additive smoothing `alpha=0.5`. Ten passage features summarize
  mean and standard deviation of next-tag surprise, conditional-mode mismatch,
  unseen-context rates, and human-minus-AI differences. Three-fold stratified
  out-of-fold features train the classifier; final conditional models then fit
  all training rows for validation/prediction. This is a small statistical
  implementation of the next-token/structure idea, not a neural word predictor.
- **Word plus style model:** The same word unigram/bigram TF-IDF representation
  as the `word_lr, C=1` baseline, concatenated with the standardized 28-feature
  style block at fixed block weights `word=1.0`, `style=0.1`, then logistic
  regression. This is the controlled feature-addition comparison.

The pretrained component is NLTK's English
`averaged_perceptron_tagger_eng`; it supplies POS annotations, not competition
authorship labels. Tokenization/tagging is local. Missing tagger data raises an
explicit error. No external LLM was used to label or predict these passages.

## Measured validation results

[bonus_comparison.json](results/bonus_comparison.json) contains metrics,
configuration and input hashes, timings, warnings, sensitivity scores, and
paired intervals. The original per-run records and predictions from `bonus/`
are retained in [day1-evidence.zip](results/day1-evidence.zip), together with
the other detailed run records and subset IDs. Original relative paths are
preserved and `MANIFEST.sha256` records file checksums. See the [README](README.md)
for extraction instructions.
The baseline and best core configuration come from
[comparison.json](results/comparison.json) and
[selected_config.json](results/selected_config.json); the original `selection.json`
is also preserved in the evidence archive.

| Model | Train AUROC | Validation AUROC | Change versus word C=1 | Paired 95% interval for change |
| --- | ---: | ---: | ---: | --- |
| Word TF-IDF LR, C=1 baseline | 0.994270 | 0.935686 | — | — |
| Style LR | 0.901183 | 0.906037 | −0.029649 | [−0.044165, −0.014655] |
| POS unigram/bigram/trigram LR | 0.971017 | 0.947155 | +0.011469 | [+0.001519, +0.021890] |
| Conditional POS predictability LR | 0.977269* | 0.926211 | −0.009475 | [−0.022030, +0.003094] |
| Word plus style LR | 0.988412 | 0.955052 | +0.019366 | [+0.010550, +0.028027] |
| Best core model: character TF-IDF LR, C=3 | 0.999148 | **0.970756** | Comparison reference only | Not estimated here |

The intervals use 1,000 paired, stratified bootstrap resamples of the fixed
validation predictions, seed `20261023`. They are exploratory and conditional
on the fitted models. They do not cover training-set variability, repeated
validation use, or selection/multiple-comparison uncertainty.

The word-plus-style comparison improved AUROC by 0.019366, approximately 1.94
percentage points. This supports testing the feature block as a complement to
word TF-IDF. It does not establish that any particular adjective, positivity,
or personalization feature caused the gain; those require feature-family
ablations. The POS sequence model also exceeded this word baseline. Style alone
was weaker, and conditional POS surprise did not show a clear advantage over
the baseline. None of these four bonus models surpassed the selected character
model, so they do not by themselves justify replacing it.

*The reported conditional-model training score uses final POS models fitted on
all training rows, whereas its classifier was fitted using out-of-fold scores.
It is optimistic resubstitution performance with a different feature-generation
regime; do not interpret its apparent train/validation gap as a clean measurement
of classifier overfitting.

Excluding the eight validation rows flagged by the fixed source-overlap audit
gave word-plus-style AUROC `0.954794`, POS-sequence `0.946978`, conditional-POS
`0.925866`, and style `0.905653`. Character C=3 remained at `0.970627`. This
limited sensitivity check leaves the ranking unchanged. It does not rule out
unobserved shared sources, prompts, domains, or paraphrases.

Recorded fit times are about 30.05 seconds for style, 1.10 for POS sequences,
3.40 for conditional POS, and 7.98 for word plus style. The runs shared a
process-local POS cache, so later runs reused earlier tagging work. These are
not comparable cold-start runtime estimates.

## Measurement and generalization limits

The feature audit counts regex words: Unicode `\w` tokens, including digits and
underscores; contractions are retained, hyphens split, and HTML tag names may
be counted. Its word range is 74–390 and pooled training tertile edges are
177 and 244. The source audit's 75–350 range uses whitespace-separated words.
These definitions must not be mixed in figures or length-bin claims.

The “sentence” features use punctuation followed by whitespace or newlines as
boundaries. Abbreviations, recipes, HTML, lists, and punctuation without spaces
can alter these counts. POS ratios divide by all Treebank tokens, including
punctuation. POS tagging itself is imperfect, especially for unusual formatting.
The conditional model includes a passage-end tag and does not reset n-gram
contexts at each sentence. For an unseen context its smoothed distribution is
uniform, while its mismatch feature falls back to the class's marginal mode.
None of this defines linguistic correctness.

Length and source confounds are visible in training. Type-token ratio correlates
`−0.464` with regex word count; its AI-minus-human effect disappears in the
longest length tertile. The newline effect is near zero in the shortest tertile
but larger in the middle and longest tertiles. HTML breaks appear in 161 human
versus seven AI training passages. Recipe markers appear in 659 human versus
774 AI passages. These are observable formatting/topic proxies, not known
source or domain labels. Length normalization and tertile comparisons do not
remove all these confounds.

The notes' semantic adjective appropriateness, poeticness, personalization, and
true grammatical-error hypotheses remain untested. A neural next-word predictor
was not trained. The tested statistical alternative is reproducible and useful
as an experiment, but should not be described as validating every handwritten
idea.

## Measured extension and Day 2 ownership

After observing the core and four bonus outcomes, one additional exploratory
comparison was declared in [extension_plan.json](results/extension_plan.json): a
fixed equal-weight probability average of character LR `C=3` and word-plus-style
LR `C=1`, versus character LR alone. The weights were not searched. The resulting
probabilities matched the arithmetic average of the independently recorded
component predictions exactly (maximum absolute difference `0.0`).

| Metric | Character LR C=3 | Equal-weight blend |
| --- | ---: | ---: |
| Training AUROC | 0.999148 | 0.998927 |
| Validation AUROC, 2,000 rows | 0.970756 | **0.978162** |
| Source-overlap sensitivity AUROC, 1,992 rows | 0.970627 | **0.978041** |
| Validation log loss, lower is better | **0.281739** | 0.297826 |
| Recorded fit seconds | 9.00 | 40.54 |

[extension_comparison.json](results/extension_comparison.json) records a
validation AUROC improvement of `+0.007406`, approximately 0.74 percentage
points, with exploratory paired 95% interval `[+0.003951, +0.011163]`. This met
the declared handoff rule: recommend the blend only if AUROC increased and the
interval's lower bound exceeded zero. The blend is therefore the **Day 1
recommendation**, with character LR C=3 retained as the simpler core fallback.
This preserves the original core selection and four-candidate bonus record.
The recommended blend's validation IDs, labels and probabilities remain directly
available in [validation.predictions.tsv](results/validation.predictions.tsv).

The result supports complementary ranking information from the two component
models on this holdout. It does not isolate which style feature matters or
establish the proposed semantic mechanism. Ranking improved, but probability
log loss worsened by about `0.016088`; better AUROC therefore does not mean
better probability estimates under every metric. Log loss alone does not
separate calibration from discrimination, so a specific calibration claim
requires a separate diagnostic. The blend also requires two models and POS
tagging, with higher recorded fit cost. AUROC is the competition metric, which
motivates the recommendation despite these tradeoffs.

This extension was chosen adaptively after the earlier validation comparisons.
Its bootstrap interval is conditional on the fitted models and does not correct
for that selection, repeated holdout use, or training variability. The promotion
rule is a practical selection heuristic, not confirmatory statistical evidence.
The source-overlap sensitivity ranking is encouraging but still does not
establish robustness to unfamiliar domains, generators, or adversarial edits.

Day 2 still owns reviewing and incorporating the controlled-improvement evidence
into Q5, checking feature-family/source robustness as appropriate, and making
the final model recommendation. These Day 1 bonus experiments supply evidence;
they do not claim that the next owner's review or the remaining assignment
sections are complete.
