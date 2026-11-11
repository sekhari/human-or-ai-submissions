"""Check a private text-model handoff in a clean Python process, without fitting."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("name", choices=["10_expanded_text", "11_formatting_rule", "15_normalized_transfer", "17_final_blend"])
parser.add_argument("--mode", choices=["primary", "hard"], default="primary")
args = parser.parse_args()
if args.mode == "hard" and args.name != "11_formatting_rule":
    parser.error("The hard path is defined only for notebook 11")
root = Path(__file__).resolve().parents[1]
pointer = json.loads((root / "artifacts" / args.name / "latest.json").read_text())
split = json.loads((root / "configs/split.json").read_text())
assert pointer["split_sha256"] == split["split_sha256"]
import hashlib

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

assert digest(root / pointer["result"]) == pointer["result_sha256"]
for path, expected in pointer["artifacts"].items():
    assert digest(root / path) == expected, "Artifact changed"
truth = pd.read_csv(root / "data/raw/train.csv", dtype={"id": "string"}).set_index("id")
validation = truth.loc[split["validation_ids"]]
prediction_key = "hard_validation_predictions" if args.mode == "hard" else "validation_predictions"
model_key = "hard_model" if args.mode == "hard" else "model"
saved = pd.read_csv(root / pointer[prediction_key], dtype={"id": "string"}, float_precision="round_trip")
assert saved.id.tolist() == split["validation_ids"]
program = """
import json, sys, cloudpickle
import numpy as np
from threadpoolctl import threadpool_limits
request = json.load(sys.stdin)
with open(request['model'], 'rb') as handle:
    model = cloudpickle.load(handle)
assert list(model.classes_) == [0, 1]
with threadpool_limits(limits=1):
    predictions = np.asarray(model.predict_proba(request['texts']))
assert predictions.shape == (len(request['texts']), 2)
assert np.isfinite(predictions).all() and ((predictions >= 0) & (predictions <= 1)).all()
assert np.allclose(predictions.sum(axis=1), 1)
json.dump(predictions[:, 1].tolist(), sys.stdout)
"""
# Passage text travels over the child process's stdin and is never printed.
request = {"model": str(root / pointer[model_key]), "texts": validation.text.tolist()}
result = subprocess.run([sys.executable, "-c", program], input=json.dumps(request),
                        text=True, capture_output=True, check=True)
actual = np.asarray(json.loads(result.stdout))
difference = float(np.max(np.abs(actual - saved.p_ai.to_numpy())))
assert difference <= 1e-12, "Clean-process probabilities differ"
print(f"{args.name} {args.mode}: all {len(actual)} held-out predictions match in a clean process; maximum difference {difference:.3g}.")
print("No notebook globals, fitting, or downloads were used. Private artifact hashes passed.")
