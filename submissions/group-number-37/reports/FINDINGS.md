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
