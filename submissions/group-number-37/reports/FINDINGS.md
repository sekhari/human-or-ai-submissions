# findings log

running notes, newest section last. every number comes from a script in `experiments/`, raw tables are next to this file.
fixed split: 8000 fit, 2000 validation, stratified, seed 3737. lodo: train without one inferred domain, score that domain.

## 1. what the data looks like (01, 02)

- adversarial validation auc: train vs test 0.74, train vs unlabeled 0.60, unlabeled vs test 0.73.
  unlabeled looks like train, so it says little about the test shift.
- one inferred domain (stories, novels, poetry) is 1084 of 3000 test rows but 143 of 10000 train rows.
- about half of that test domain looks like poetry by line length. none of the human rows in that train domain do.
  domains are kmeans clusters, not real labels, so treat these as approximate.
- a double newline appears in 3301 of 5000 ai rows and 0 of 5000 human rows.
  in test the rate is 0.32, close to train at 0.33, which fits a similar class mix but does not prove the signal holds.

## 2. the reference model (03, 05)

the first notebook's recipe: word and char tf-idf logistic regression, catboost on style features, 0.35/0.65 blend.

- validation 0.9987, lodo mean 0.9927, worst domain 0.9657.
- removing double newlines at inference: 0.9966. flattening all whitespace at inference: 0.8937.
- retraining on flat text: 0.9818.
- so it does not collapse on unseen prose domains. it still cannot be tested on human poetry locally.

## 3. does it depend on formatting? (05)

| model | val | lodo mean | lodo worst |
| --- | --- | --- | --- |
| reference style, raw text | 0.9975 | 0.9894 | 0.9504 |
| style features, cleaned text | 0.9684 | 0.9258 | 0.8311 |
| reference lexical, raw text | 0.9780 | 0.9330 | 0.8594 |
| word and char lr, cleaned text | 0.9611 | 0.9137 | 0.8363 |

cleaning normalizes whitespace, quotes, dashes and unicode. it costs about 6 points of held out domain auc on style features.
typography carries a lot of the signal. local data cannot say whether that is a trait of ai text or a quirk of how this
dataset was assembled, since train and validation share it. a lower local score for cleaned models is not proof they do worse on test.

## 4. surprisal features (04, 11)

gpt2 log probabilities on flattened text, so formatting cannot leak in.

- best single feature: std of token log prob, auc 0.81, at least 0.80 inside every inferred domain.
- the strongest features keep the same direction across all domains. that stability is why they are worth blending.

## 5. robustness, training size, unlabeled data, tuning (07, 08, 09, 10)

- attacks on the reference blend: flat whitespace 0.8937, added blank lines 0.9768, everything else at least 0.9966.
- learning curve: style features already reach 0.9905 with 500 rows and 0.9975 with 8000. cleaned lexical goes 0.8698 to 0.9611, so it still wants data.
- self-training on unlabeled or test text did not help: validation 0.9611 to about 0.9605, held out book and poetry 0.9194 to 0.9216 at best.
- grid search: about 0.001 or less between the settings near what we use, so tuning is not where the gain is.

## 6. surprisal changes the picture (05, 14, 15)

| model | train | val | lodo mean | lodo worst |
| --- | --- | --- | --- | --- |
| surprisal gbm, no formatting | 0.9856 | 0.9774 | 0.9716 | 0.9560 |
| reference blend | 1.0000 | 0.9987 | 0.9927 | 0.9657 |
| reference + 0.3 surprisal, rank blend | | 0.9990 | 0.9966 | 0.9862 |

- with the weight picked from the other domains: mean 0.9964, worst 0.9862. picked 0.3 for nine domains and 0.4 for one.
- bootstrap of the mean gain at 0.3: +0.0039, 95% interval +0.0030 to +0.0049.
- it helps most where the reference is weakest (domain 9, 0.9657 to 0.9862) and costs a little where it is near perfect (domain 7, 0.9999 to 0.9967).
- on the fixed split the best weight is 0.1 (0.9994). lodo prefers 0.3. we use 0.3 because transfer is what the test set asks for.
- limit: every domain here is prose. none of this measures human poetry.

## 7. evidence from test without labels (12)

spearman agreement between the reference and surprisal on test is lowest on the book and poetry domain: 0.59, against 0.61 to 0.81 elsewhere.
the reference is using different information where test differs most from train. this does not say which model is right.

## 8. encoder problem and what caused it (06, 13)

distilroberta on cleaned text scored 0.9462, with 27% of human training rows confidently called ai and almost no errors the other way.
typos raised auroc to 0.98. noise should not do that, so i stopped the queue and tested:

- ruled out: mps vs cpu, sdpa vs eager attention, batch size and order. all gave the same scores.
- train mode vs eval mode on the same weights (1600 fit rows, 500 val rows): human rows above zero 8.5% with dropout on, 39% with it off.
- removing dropout: human rows above zero 39% to 9%, ai rows below zero 3% to 8%, val 0.9605 to 0.9687, typo gain almost gone (+0.001).

so the model learned that clean text means ai while dropout was noising its inputs, then saw clean inputs at prediction.
the encoder is still well below surprisal on these subsets and has not been retrained on all rows.

## 9. encoder after the dropout fix (06, 18)

distilroberta on cleaned text, 2 epochs, dropout off: train 0.9983, validation 0.9898, up from 0.9462. no formatting used.
trained without the book and poetry domain it scores 0.9967 on that domain (143 rows, 41 human, no human poems).
text edits now lower its score as expected: typos 0.9773, char noise 0.9740, case noise 0.9704.

| reference | surprisal | encoder | val | held book and poetry |
| --- | --- | --- | --- | --- |
| 1.0 | 0 | 0 | 0.99868 | 0.99833 |
| 0.7 | 0.3 | 0 | 0.99900 | 1.00000 |
| 0.8 | 0.1 | 0.1 | 0.99957 | 0.99976 |
| 0.7 | 0.2 | 0.1 | 0.99952 | 1.00000 |
| 0 | 0.5 | 0.5 | 0.99353 | 0.99821 |

weights here were read off validation, so these are optimistic. experiment 19 repeats this with each domain held out.

## 10. audit of the first notebook (17)

the first notebook's numbers are accurate. its audit score reproduces exactly at 0.998243 on its own split, and its submission has
rank correlation 1.0 with a rerun of the recipe. ids do not leak labels (id auc 0.4946). near duplicates do not carry validation:
137 of 2000 audit rows have a neighbor above 0.5 cosine, nearest neighbor label agreement is 0.518, and the less similar half still
scores 0.9962. gaps are coverage, not honesty: it does not test poetry and leans on formatting.

## 11. encoder with every domain held out, and the final weights (19, 21)

each of the 10 inferred domains was held out in turn and the encoder retrained without it, about 20 minutes per domain on an m4.
reference and surprisal scores come from the same held out setup (14).

| reference | surprisal | encoder | held out mean | worst domain | test mix | fixed val |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 0 | 0 | 0.9927 | 0.9657 | 0.9945 | 0.9987 |
| 0.7 | 0.3 | 0 | 0.9966 | 0.9862 | 0.9978 | 0.9990 |
| 0.8 | 0.1 | 0.1 | 0.9965 | 0.9829 | 0.9976 | 0.9996 |
| 0.6 | 0.2 | 0.2 | 0.9977 | 0.9918 | 0.9983 | 0.9994 |
| **0.55** | **0.25** | **0.2** | **0.9977** | **0.9930** | **0.9984** | **0.9992** |
| 0.5 | 0.3 | 0.2 | 0.9977 | 0.9937 | 0.9983 | 0.9989 |
| 0.4 | 0.3 | 0.3 | 0.9973 | 0.9948 | 0.9979 | 0.9985 |

test mix weights each domain's auroc by its share of test. it is an average of within domain scores, not an estimate of the full
test auroc, and 36% of its weight rests on domain 5, which has 143 train rows and 41 humans. treat it as a direction only.

choosing weights on nine domains and scoring the tenth gives mean 0.9976, worst 0.9918, against 0.9966 and 0.9862 for the two way
blend. the rule picked 0.55/0.25/0.2 on nine folds and 0.6/0.2/0.2 on the fold holding out domain 9, so the choice is stable.
choosing by worst domain instead does worse on nearly everything (mean 0.9965), so we kept the mean rule.

bootstrap gain inside domains, 500 draws: 0.6/0.2/0.2 against the two way blend +0.0011 (ci +0.0008 to +0.0015), against the
reference +0.0050 (ci +0.0039 to +0.0063). 0.55/0.25/0.2 was not in this bootstrap, its neighbors were.

the cost: domains the reference already handles lose a little, for example domain 7 from 0.9999 to 0.9985 and domain 6 from 1.0000
to 0.9992. we accept this because test leans away from train-like text. the 0.8/0.1/0.1 blend that wins on the fixed validation
set trails the chosen blend on unseen domains (mean 0.9965, worst 0.9829), which is why validation alone did not choose the weights.

known gap: no held out score here covers human poems, because train has none in that cluster. a large part of test is poem-like, so
this remains the main unmeasured risk.

## 12. final pipeline run (train.py, predict.py)

`train.py --validate` with the default weights: train 0.9999, validation 0.9991. parts on validation: reference 0.9984, surprisal 0.9774,
encoder 0.9895. this is a real fit of all three parts on the fixed 8000 rows and matches the 0.9992 read from cached scores in 21.
`predict.py` wrote 3000 rows with matching ids and format. rank correlation with the first notebook's submission is 0.9475, and with
the two way blend 0.9896, so the encoder and surprisal do reorder test rows. not yet scored on kaggle.
