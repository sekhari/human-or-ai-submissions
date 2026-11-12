"""Evaluate two existing style recipes on the frozen layout stress split."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.text_generalization import layout_stress, style_search
from models.text_generalization.common import sha256

destination = Path(__file__).resolve().parent
source = ROOT / 'outputs/text_generalization/layout_stress/frozen_plan.json'
plan = json.loads(source.read_text())
names = ['style_hgb15', 'style_extra_leaf2']
configs = style_search.candidate_configs()
plan['configs'] = [dict(family='style', config=next(c for c in configs if c['name'] == name))
                   for name in names]
plan['diagnostic'] = 'Two already searched style recipes on the unchanged post hoc layout stress split'
plan['source_layout_plan'] = str(source.relative_to(ROOT))
plan['source_layout_plan_sha256'] = sha256(source)
plan['plan_created_before_these_two_model_fits'] = True
plan_path = destination / 'frozen_plan.json'
if plan_path.exists():
    raise FileExistsError('This bounded diagnostic has already been frozen')
plan_path.write_text(json.dumps(plan, indent=2))
print(json.dumps(dict(configs=plan['configs'], fit_rows=len(plan['train']),
                      validation_rows=len(plan['validation']),
                      source_layout_plan_sha256=plan['source_layout_plan_sha256']), indent=2), flush=True)
layout_stress.DESTINATION = destination
layout_stress.PLAN_FILE = plan_path
layout_stress.run()
