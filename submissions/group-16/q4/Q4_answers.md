# Q4. Generalization and error analysis

## Q4.1

The selected word TF-IDF logistic regression model achieved training AUROC 1.0000 and validation AUROC 0.9662, a gap of 0.0338. Perfect training ranking alongside weaker validation ranking is consistent with overfitting, although held-out discrimination remains strong. High-dimensional unigrams and bigrams can capture sample-specific phrases; C=30 also applies relatively weak regularization. These are plausible explanations, not demonstrated causes. The training and validation partitions share no passages after lowercasing and whitespace normalization, but near-duplicates were not exhaustively checked. Q1 also used this holdout to compare models, so the selected model's estimate may be optimistic. This split does not establish robustness to unseen generators, domains, or adversarial edits.

## Q4.2

| Length (words) | Count | Human / AI | Validation AUROC |
|---|---:|---:|---:|
| <150 | 404 | 150 / 254 | 0.9455 |
| 150-249 | 1055 | 620 / 435 | 0.9684 |
| >=250 | 541 | 230 / 311 | 0.9792 |

We chose length because it is observable for every passage and may affect the amount of lexical evidence available. The three bins were fixed before inspecting subgroup scores; words are whitespace-separated tokens. AUROC increases from 0.9455 for short passages to 0.9684 for medium and 0.9792 for long passages, a short-to-long difference of 0.0337. This is consistent with more text providing stronger evidence, but does not establish a causal length effect: content composition and sampling variation may contribute. Both labels occur in every bin. These descriptive results identify short texts as a useful focus for further evaluation; they do not establish performance on hidden test subgroups.

## Q4.3

Threshold 0.5; five most confidently wrong examples of each type. Model probabilities have not been calibrated.

| ID | Error | P(AI) | Possible reason |
|---|---|---:|---|
| 5234 | FP | 0.9319 | Keyboard complaint: "it" + apostrophe, ", with", and "overall" push toward AI. Local phrasing may outweigh its informal style. |
| 4947 | FP | 0.9052 | Album track list: "#" and ":" each add about +0.99 log-odds toward AI. List punctuation may dominate factual details. |
| 3671 | FP | 0.8983 | Biography: "including" and related phrases contribute toward AI. Conventional explanatory wording may be mistaken for AI style. |
| 7767 | FP | 0.8829 | Botanist biography: "the university", "a member", and ", including" favor AI. Topic-related phrases may mislead the model. |
| 8238 | FP | 0.8714 | Technical abstract: "such as", ":", and "significant" favor AI. Enumerating examples may resemble learned AI phrasing. |
| 3062 | FN | 0.0107 | Personal question: curly apostrophes, "my", and "i" favor human. Informal surface cues may obscure awkward or inconsistent statements. |
| 9490 | FN | 0.0133 | Recipe repeats "buttered, buttered" and has disconnected steps. Common words and ". set" favor human; local n-grams miss coherence. |
| 1524 | FN | 0.0186 | Technical passage ends with a personal question. "of the", "state", and common words favor human, potentially masking mixed structure. |
| 4600 | FN | 0.0236 | Boxer biography includes dates and malformed "sisters,108". "at the" and "from" favor human; lexical features do not assess factual consistency. |
| 473 | FN | 0.0284 | Quoted question and answer: "have", "i", and "a job" favor human. Conversational phrasing may outweigh other signals. |

## Q4.4

Among these ten confidently wrong cases, human biographies and technical prose were flagged as AI when conventional connectors, lists, and punctuation contributed positively. Conversely, AI passages containing personal questions, recipe instructions, or biographical details received human-directed contributions from common words and conversational phrasing. The recipe contains repetition and disconnected steps that this unigram/bigram representation does not directly model as document-level coherence. These observations support hypotheses about reliance on local wording; additive feature contributions explain the fitted score, not the underlying cause of an error. The ten selected extremes cannot estimate how common each pattern is. To investigate punctuation sensitivity, we would test quote normalization and removal of standalone punctuation tokens as separate ablations, holding other settings fixed. Training-only cross-validation would compare AUROC and class-specific error rates before further holdout evaluation. These changes have not been tested; punctuation may contain useful signal, so removing it could also worsen performance.
