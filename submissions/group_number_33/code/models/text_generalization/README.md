This round selects `style_extra_leaf2`: 145 scalar features extracted from each
original text, followed by `ExtraTreesClassifier(n_estimators=400,
min_samples_leaf=2, max_features=0.8, max_depth=None, random_state=12, n_jobs=2)`.
The features describe length, punctuation, casing, repetition, whitespace and
layout. They use no document IDs, labels, learned vocabulary or external models.
The frozen selection contains one component with weight 1.0 and probability
aggregation. Pseudo-label experiments were evaluated but are not adopted.
Shared preprocessing and the existing `models/multinomial_nb` implementation
remain unchanged.

This method is published on `Dave-new-method-1`; merging into `main` is outside
this change. The raw data already present in the base repository is unchanged.
Experiment Python files and frozen JSON/aggregate CSV records are included;
generated models, caches and row-level prediction CSV/NPZ files remain local.

For a clean checkout, run from the repository root. **Python 3.10.8** was tested;
the dependency versions are pinned in `requirements.txt`.

```sh
python3 -m pip install -r models/text_generalization/requirements.txt
python3 -m models.text_generalization.train_final
```

No search or protocol regeneration is needed. The runner consumes the committed
`outputs/text_generalization/protocol.json`,
`outputs/text_generalization/selection/selected.json` and
`outputs/text_generalization/layout_stress/frozen_plan.json`. Preserve their
**exact bytes**: protocol hashes are checked, so even JSON reformatting can
invalidate the frozen selection. `--selection PATH` overrides the default
selection file.

After checking the frozen selection and protocol hashes, this command fits the
recipe on the original 8,000 development rows and evaluates it once on the old
2,000-row benchmark. It then refits the unchanged recipe on **all 10,000 labeled
rows** and predicts the 3,000 test rows. No pseudo-labels or unlabeled examples
enter the final fit. The old benchmark was used in earlier work, so it is not an
untouched test; its score does not alter the frozen recipe.

Each run writes `outputs/text_generalization/final/<timestamp>/`, including
`metrics.json`, `validation_predictions.csv`, `model.joblib`, frozen selection
and protocol copies, and `submission_style_extratrees_full10000.csv`. Read the
actual benchmark score from that run's `metrics.json`. The recorded
`outputs/text_generalization/final/20261106T205025Z/metrics.json` reports
**AUROC 0.995889** on the old
2,000-row benchmark after an 8,000-row fit, followed by the full 10,000-row fit
and a 3,000-row submission. All built-in checks passed. The user separately
reported a public score around **0.95**; that leaderboard result was not
authenticated or independently verified here.

The runner verifies submission schema/IDs, finite
probabilities, serialization, batched prediction parity and fit-row counts.
It creates a local CSV and does not upload to Kaggle.

Recorded development AUROCs, rounded to four decimals:

| Fixed recipe | Random | Topic held out | Layout held out |
|---|---:|---:|---:|
| **Style ExtraTrees, leaf 2 — selected** | **0.9976** | **0.9957** | **0.8513** |
| Style HGB, 31 leaves | 0.9988 | 0.9978 | 0.7988 |
| Style HGB, 15 leaves | 0.9986 | 0.9975 | 0.8112 |
| HGB31 without explicit layout features | 0.9752 | 0.9438 | 0.6868 |
| Word 1–2 grams + punctuation, LR C=4 | 0.9627 | 0.9326 | 0.4990 |
| Character 3–6 grams, LR C=8 | 0.9578 | 0.9176 | 0.5076 |

All three diagnostic partitions use only the original 8,000 development rows.
Random validation uses 6,000/2,000 rows. Topic validation uses 6,042/1,958 rows,
with two of eight unsupervised text clusters held out. Layout validation was
added after observing formatting dependence: the fixed rule selected the
largest eligible layout group, multiline text without blank paragraphs, with
5,174 fit rows and 2,826 validation rows. Layout strongly correlates with label;
blank-paragraph texts contain 20 human and 2,675 AI examples. Random and topic
validation share 494 rows, and all these results informed development choices.
They are neither independent unbiased test estimates nor guarantees of Kaggle
performance.

Selection compared 33 fixed standalone/blend scoring recipes. It maximized
mean random/topic/layout AUROC subject to random/topic mean remaining within
0.005 of the best ordinary standalone result; near ties within 0.0005 favored
fewer components, then probability averaging. ExtraTrees alone won this rule.
Exact scores, configurations and the criterion are stored in
`outputs/text_generalization/selection/selected.json`.

The source includes the word/character/style searches, style and layout
ablations, group reweighting, pseudo-label diagnostics and ensemble selection.
Run search modules in a separate working copy because their output JSON/CSV
files may be overwritten. First-time initialization commands
`python3 -m models.text_generalization.make_protocol` and
`python3 -m models.text_generalization.layout_stress --plan-only` intentionally
refuse existing plans; do not run them as clean-checkout prerequisites.
The historical layout-variant driver also refuses an existing plan, and
`layout_ablation` refuses existing results. See the
[artifact README](../../outputs/text_generalization/README.md) for a safe
layout-variant rerun that preserves recorded evidence.

Leakage/cache/alignment checks are documented in
`outputs/text_generalization/style_independent_review.txt`; those checks found
no direct implementation leakage, while formatting/source transfer remains a
material limitation.

The recorded final metrics retain historical `git_head`
`346bb6f4689bff8683d115e344ba93275afe0ce1`, the baseline HEAD at execution time.
The new method existed in the working tree then; `source_sha256` identifies the
actual Python files used. This historical HEAD is not the later publication
commit on `Dave-new-method-1`, and should not be rewritten to match it.
