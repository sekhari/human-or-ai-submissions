This directory preserves the evidence for the method published on
`Dave-new-method-1`. The code and frozen JSON/aggregate CSV experiment records
are committed. Generated `model.joblib` files, caches, per-row prediction CSVs
and prediction/feature NPZs remain local and are recreated by the experiment
commands. Raw data already present in the base repository is unchanged.

| Record | Purpose |
|---|---|
| `protocol.json` | Original 8,000/2,000 membership and internal random/topic partitions |
| `layout_stress/frozen_plan.json` | Frozen layout split, counts and fixed configurations |
| `selection/selected.json` | Selected single-component ExtraTrees recipe and selection criterion |
| Per-family frozen JSON and aggregate score CSV files | Search settings, diagnostics, AUROC and timing records |
| `final/20261106T205025Z/metrics.json` | Recorded final-run metrics, versions, hashes and verification |

The three first-listed JSON files are runtime inputs to final reproduction.
Keep their exact bytes: `train_final` checks the stored hashes, including those
of the two protocol files. A clean checkout can directly run, from repository
root, using the tested Python 3.10.8 environment:

```sh
python3 -m pip install -r models/text_generalization/requirements.txt
python3 -m models.text_generalization.train_final
```

The recorded run reports **0.995889 AUROC** after an 8,000-row fit and evaluation
on the old 2,000-row benchmark. The frozen recipe was then refitted on all
10,000 true-labeled examples, without pseudo-labels, to create 3,000 test
predictions. Built-in schema/ID/probability, serialization, batch and row-count
checks passed. New runs write new timestamped final directories. The user
reported a public score around **0.95**; no authenticated leaderboard result is
recorded here.

Random, topic and the post hoc layout partitions are overlapping development
diagnostics, not unbiased independent tests. The old 2,000-row benchmark was
also used in earlier work. None of these scores guarantees Kaggle performance.

Historical metrics retain `git_head=346bb6f4689bff8683d115e344ba93275afe0ce1`, the
baseline HEAD when the experiment ran. The new code was present in the working
tree; its actual contents are identified by `source_sha256`. The later branch
publication commit is a different provenance event. Preserve the original
metadata rather than replacing that historical HEAD.

Search modules may overwrite their per-family JSON/CSV outputs; use a separate
working copy for a full search rerun. `make_protocol` and
`layout_stress --plan-only` are first-time initializers and refuse existing
plans. The saved `layout_style_variants/run.py` likewise refuses its existing
plan. `layout_ablation` refuses an existing `results.json`. None is a required
setup step for final reproduction, and no evidence needs to be removed.

To rerun the two already frozen layout variants safely, copy their plan bytes
to a fresh destination and call the reusable evaluator. This uses exactly the
recorded configurations and indices, leaving the historical directory intact:

```sh
python3 - <<'PY'
from datetime import datetime, timezone
from pathlib import Path
from models.text_generalization import layout_stress

base = Path('outputs/text_generalization')
source = base / 'layout_style_variants/frozen_plan.json'
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
destination = base / 'reruns' / f'layout_style_variants_{stamp}'
destination.mkdir(parents=True, exist_ok=False)
plan = destination / 'frozen_plan.json'
plan.write_bytes(source.read_bytes())
layout_stress.DESTINATION = destination
layout_stress.PLAN_FILE = plan
layout_stress.run()
PY
```

The evaluator writes new `results.json`, `scores.csv` and aligned prediction
NPZs to that destination. Search/selection reconstruction requires regenerating
the local prediction arrays first; the committed aggregate scores alone are
not inputs to `select_ensemble`. See the
[method README](../../models/text_generalization/README.md) for the model,
comparison table and evaluation scope.
