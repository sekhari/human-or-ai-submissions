# Development history — Group Number 28

The source is the team working repository `lins369/kaggle_ai_vs_human`,
frozen at commit `fe4265e96a604555c89895e9bc2c3cbc6b4c48da` for this packaging pass.
All 24 original commits were retained under `submissions/group-number-28/`.
The original author/committer identities, timestamps, commit messages, and
development topology are preserved. Commit IDs change because the paths and
trees change. Some original commits retrospectively archived completed local
experiments; their original provenance and archive dates remain as recorded.

Report TeX/PDF build artifacts were excluded throughout the imported history.
Notebook cell sources were retained exactly; saved outputs and execution
counts were cleared throughout history to exclude competition passage excerpts.
The historical numeric experiment records and frozen source snapshots are
retained. Raw competition CSVs and trained model weights are supplied locally.

The additional course-packaging commits add the independent portable final
RoBERTa entrypoints, reproduction commands, team identity, and AI disclosures.
They do not represent new model-selection experiments or a new Kaggle score.

## Examples for report Q6.2

- **Character-feature ablation.** The fixed word/character SVM comparison raised development AUROC from 0.943876 (word-only) to 0.970006 (word + character). Character-only performance was close to the combination, so the evidence did not establish that the combination was superior to character features alone. [`a911a30`](https://github.com/lins369/human-or-ai-submissions/commit/a911a301d8debca272eda74bb1fefb4bba3784ec) · [`7147950`](https://github.com/lins369/human-or-ai-submissions/commit/7147950ca8b4001d62af28dd860c8ffcb337627d)

- **Learning from unlabeled data.** Self-training and vocabulary interventions used the same labeled validation protocol. The recorded differences were small and confidence intervals included zero; the team retained this negative result instead of claiming an established gain. [`0a7079f`](https://github.com/lins369/human-or-ai-submissions/commit/0a7079f3a8a5ffb8bb3c99cd7fad8d719560593f) · [`a9ec830`](https://github.com/lins369/human-or-ai-submissions/commit/a9ec8308bdb16185ee83c1fba17a2ac4b3e96997)

- **RoBERTa and public feedback.** The raw-text RoBERTa experiment produced the recorded public score 0.96970 versus the prior WC120 score 0.87399. The accompanying format-sensitivity diagnostics showed a limitation that a high clean development score alone would not reveal. [`60b7e96`](https://github.com/lins369/human-or-ai-submissions/commit/60b7e960c0313f7a977933dfd81b445327eeb2f0)

## Commit mapping

| Original commit | Course-fork commit | Original message |
| --- | --- | --- |
| `17aba0e` | [`af27f09`](https://github.com/lins369/human-or-ai-submissions/commit/af27f09232005d0e7f34f1c16cf6b8c4407abeed) | Initialize shared competition scaffold and fixed validation split |
| `4696415` | [`0cf7361`](https://github.com/lins369/human-or-ai-submissions/commit/0cf7361ad27c81ab567a8be7174c71b568d6e3db) | Archive completed Q1 baseline experiments and prediction pipeline |
| `0936f15` | [`a42f86c`](https://github.com/lins369/human-or-ai-submissions/commit/a42f86cf24a4626376a062c230b30b1639a81d60) | Archive expanded Q1 models and training-only parameter search |
| `7640f1b` | [`4aa95ef`](https://github.com/lins369/human-or-ai-submissions/commit/4aa95efe1fc40c24b8fd7c41d16462cd9385a442) | Add Q1 contract tests and recorded reproduction audits |
| `74f4859` | [`ad1f32c`](https://github.com/lins369/human-or-ai-submissions/commit/ad1f32c0bd426ed83ea034570437c2a8ea8e108f) | Document Q1 evidence history, measured conclusions and team handoff |
| `659b2ea` | [`1fae128`](https://github.com/lins369/human-or-ai-submissions/commit/1fae128c8856c1a2fd6060199fabd066d0211199) | Translate Q1 handoff and require English repository contributions |
| `2c6f596` | [`4cac8ad`](https://github.com/lins369/human-or-ai-submissions/commit/4cac8adc47f5e1f0430a3b8245cfd79aa6577e64) | Keep Q1 PR focused on code and use Overleaf for reports |
| `74d40df` | [`d1b88c1`](https://github.com/lins369/human-or-ai-submissions/commit/d1b88c1d0f5de4d3b46d5cd981276b6caa02e3cc) | Merge pull request #1 from lins369/lin/q1-experiment-evidence |
| `cccc067` | [`a911a30`](https://github.com/lins369/human-or-ai-submissions/commit/a911a301d8debca272eda74bb1fefb4bba3784ec) | Add frozen Q5 character-feature study and resumable local runner |
| `784adce` | [`7147950`](https://github.com/lins369/human-or-ai-submissions/commit/7147950ca8b4001d62af28dd860c8ffcb337627d) | Record measured Q5 feature gains and verified prediction evidence |
| `38a5c0b` | [`6ab041f`](https://github.com/lins369/human-or-ai-submissions/commit/6ab041f0832fc83df9495c17d00bb9f3bdb62d93) | Document Q5 outcomes, reproduction commands and team handoff |
| `86841b9` | [`c947821`](https://github.com/lins369/human-or-ai-submissions/commit/c947821dc33e85fda62971f4f087d44cc9ee35e9) | Merge pull request #2 from lins369/lin/q5-character-features |
| `e1b79cb` | [`2dffb3a`](https://github.com/lins369/human-or-ai-submissions/commit/2dffb3a7bbd418e967592881eff473df83f95432) | Archive verified Q5 robustness and richness experiments |
| `95fc79a` | [`5666592`](https://github.com/lins369/human-or-ai-submissions/commit/5666592a509576b9084ea9fa6482ca2a4a0c40ba) | Add public-gap audits and fold-contained crop experiments |
| `62e2030` | [`3bf8123`](https://github.com/lins369/human-or-ai-submissions/commit/3bf8123e779b44bf580fbe6924a6301bd95c2ae2) | Archive verified public-score and crop transfer evidence |
| `0b4a1a9` | [`da7b36d`](https://github.com/lins369/human-or-ai-submissions/commit/da7b36dd34f774d208332b96bc26bf793f1ec96c) | Document public feedback and adaptive crop tradeoffs |
| `782eae6` | [`60b7e96`](https://github.com/lins369/human-or-ai-submissions/commit/60b7e960c0313f7a977933dfd81b445327eeb2f0) | Add RoBERTa experiments and public score 0.96970 |
| `ac98ac1` | [`8b47b2e`](https://github.com/lins369/human-or-ai-submissions/commit/8b47b2e8cb3720da30c61001803419accbfb610a) | Merge pull request #3 from lins369/lin/q5-robustness-richness-results |
| `89555f9` | [`0a7079f`](https://github.com/lins369/human-or-ai-submissions/commit/0a7079f3a8a5ffb8bb3c99cd7fad8d719560593f) | Q3: add self-training experiment script (baseline, threshold sweep, multi-round, vocab variant) |
| `cffbe54` | [`8d36d05`](https://github.com/lins369/human-or-ai-submissions/commit/8d36d05821fba840ce2f78bca63caa63c75ae176) | Q3: results, add figure |
| `a09a989` | [`ebdb802`](https://github.com/lins369/human-or-ai-submissions/commit/ebdb80262c0ba093f3cb970675d0b431ff45c5a9) | Add formatting robustness check |
| `07a5e65` | [`a9ec830`](https://github.com/lins369/human-or-ai-submissions/commit/a9ec8308bdb16185ee83c1fba17a2ac4b3e96997) | Q3: add README with results and commands |
| `8844388` | [`9646674`](https://github.com/lins369/human-or-ai-submissions/commit/96466749be777036a1e357b4c422d330ef73ac91) | Merge pull request #4 from lins369/exp/semisupervised-q3 |
| `fe4265e` | [`d4737af`](https://github.com/lins369/human-or-ai-submissions/commit/d4737af4caee89e4c45e0fd3662f7f64e3513263) | fix: q2, q5 training |
