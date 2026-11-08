"""Blend the selected text/style neural head with the original strongest booster.

    python -m experiments.deberta_hybrid_blend --stage compare  # CPU only
    python -m experiments.deberta_hybrid_blend --stage scores   # exclusive GPU
    python -m experiments.deberta_hybrid_blend --stage submit
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import roc_auc_score
import torch

from experiments.deberta_combinations import partitions
from src.blend_selection import WEIGHTS, logit_mixture, select_weights
from src.data import load_split
from src.deberta_hybrid import FusionHead, head_probabilities, prepare_inputs, train_head
from src.features import build_extended_features
from src.split import VAL_IDS_PATH

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/deberta_hybrid_blend"
CACHE = ROOT / ".cache/deberta_hybrid_blend"
HEADS = ROOT / ".cache/deberta_combinations"
RESULTS = ROOT / "results/deberta_combinations"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


def setup():
    record = {"weight_grid": WEIGHTS, "selection": "Maximum inner AUROC and conservative paired-bootstrap-one-SE weight, selected on training-only 1,600-row holdout; evaluate only these frozen choices, deduplicated.",
              "neural_model": "Previously selected hybrid text/style MLP, three seeds, frozen fine-tuned encoder; reconstruct fitting-only heads to obtain true held-out neural scores.",
              "hybrid_selection_sha256": sha(RESULTS / "hybrid_selection.json"),
              "train_csv_sha256": sha(ROOT / "Data/train.csv"), "validation_ids_sha256": sha(VAL_IDS_PATH),
              "caution": "Inner holdout previously selected the head and encoder. Fixed validation reused across project experiments. No independent public-score claim or automatic upload."}
    path = OUT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == record
    else:
        write(path, record)


def compare():
    from experiments.neural_comparison import paired_ci
    if (OUT / "evaluation.json").exists():
        return
    train, val, fitting = partitions()
    d = np.load(HEADS / "inner_features.npz")
    assert np.array_equal(d["ids"], train.id)
    choice = json.loads((RESULTS / "hybrid_selection.json").read_text())["hybrid"]
    xf, xe, _, _, _ = prepare_inputs(d["vectors"][fitting], d["style"][fitting], d["vectors"][~fitting], d["style"][~fitting], "hybrid")
    scores = []
    for seed in choice["seeds"]:
        model, _, _ = train_head(xf, d["labels"][fitting], xe, choice["configuration"], seed, choice["epochs"])
        scores.append(head_probabilities(model, xe))
    neural = np.mean(scores, axis=0)
    auc = roc_auc_score(d["labels"][~fitting], neural)
    assert abs(auc - choice["inner_auroc"]) < 1e-12, "Reconstructed heads did not reproduce the selected inner score."
    boosting = pd.read_csv(HEADS / "inner_boosting.csv")
    assert np.array_equal(boosting.id, train.loc[~fitting, "id"])
    selected = select_weights(d["labels"][~fitting], boosting.boosting.to_numpy(), logit(np.clip(neural, 1e-12, 1 - 1e-12)))
    selected["reconstructed_hybrid_inner_auroc"] = auc
    selected["protocol_sha256"] = sha(OUT / "protocol.json")
    write(OUT / "selection.json", selected)
    np.savez_compressed(CACHE / "inner_probabilities.npz", ids=boosting.id.to_numpy(), neural=neural, boosting=boosting.boosting.to_numpy())
    # No fixed-validation label is read for selection above.
    original = pd.read_csv(RESULTS / "val_predictions.csv")
    assert np.array_equal(original.id, val.id) and np.array_equal(original.label, val.label)
    b = original.boosting_reference.to_numpy()
    n = logit(np.clip(original.fusion_hybrid.to_numpy(), 1e-12, 1 - 1e-12))
    prediction = val[["id", "label"]].copy(); prediction["boosting_reference"] = b
    rows, seen = [], set()
    for name, key in [("hybrid_blend_maximum", "best_weight"), ("hybrid_blend_conservative", "conservative_weight")]:
        weight = selected[key]
        if weight in seen:
            continue
        seen.add(weight)
        p = logit_mixture(b, n, weight)
        prediction[name] = p
        auc = roc_auc_score(val.label, p); baseline = roc_auc_score(val.label, b)
        lo, hi = paired_ci(val.label.astype(int).to_numpy(), p, b)
        rows.append({"model": name, "neural_weight": weight, "validation_auroc": auc,
                     "difference_from_boosting": auc - baseline, "paired_95ci_low": lo,
                     "paired_95ci_high": hi, "beats_reference_numerically": auc > baseline})
    prediction.to_csv(OUT / "val_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "comparison.csv", index=False)
    write(OUT / "evaluation.json", {"results": rows, "selection_sha256": sha(OUT / "selection.json"),
                                    "training_rows": 8000, "validation_rows": 2000, "kaggle_uploaded": False})
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def winners():
    return [r for r in json.loads((OUT / "evaluation.json").read_text())["results"] if r["beats_reference_numerically"]]


def test_scores():
    from src.deberta import device_for, load_model, load_tokenizer
    from src.deberta_hybrid import extract
    if not winners():
        return
    test = load_split("test"); path = CACHE / "test_probabilities.npz"
    signature = {"encoder_sha256": sha(ROOT / ".cache/deberta_v3_mac_isolated/fit/final/model.safetensors"),
                 "scalers_sha256": sha(HEADS / "hybrid_scalers.joblib"),
                 "heads_sha256": {str(seed): sha(HEADS / f"hybrid_seed{seed}.pt") for seed in [42, 43, 44]},
                 "test_csv_sha256": sha(ROOT / "Data/test.csv"),
                 "extraction_source_sha256": sha(ROOT / "src/deberta_hybrid.py"),
                 "selection_sha256": sha(OUT / "selection.json")}
    if path.exists():
        assert np.array_equal(np.load(path)["ids"], test.id)
        assert json.loads((OUT / "test_provenance.json").read_text())["signature"] == signature
        return
    device = device_for("mps")
    model = load_model(42, ROOT / ".cache/deberta_v3_mac_isolated/fit/final").to(device)
    vectors, _ = extract(model, test, load_tokenizer(), device,
                         progress=lambda r: print(f"Hybrid blend test encoding: {r['rows']}/{r['total_rows']}", flush=True))
    del model
    torch.mps.empty_cache()
    style = build_extended_features(test).to_numpy(dtype=np.float32)
    normalizers = joblib.load(HEADS / "hybrid_scalers.joblib")
    t = np.clip(normalizers["text_scaler"].transform(vectors), -10, 10)
    s = np.clip(normalizers["style_scaler"].transform(np.log1p(style)), -10, 10)
    inputs = np.column_stack([t, s]).astype(np.float32)
    choice = json.loads((RESULTS / "hybrid_selection.json").read_text())["hybrid"]
    cfg = choice["configuration"]; probabilities = []
    for seed in choice["seeds"]:
        head = FusionHead(cfg["text_dim"], 58, cfg["width"], cfg["dropout"])
        head.load_state_dict(torch.load(HEADS / f"hybrid_seed{seed}.pt", weights_only=True))
        probabilities.append(head_probabilities(head, inputs))
    np.savez_compressed(path, ids=test.id.to_numpy(), probabilities=np.mean(probabilities, axis=0), vectors=vectors, style=style)
    write(OUT / "test_provenance.json", {"signature": signature, "rows": len(test), "training_rows": 8000,
                                         "encoder_precision": "fp32", "device": "mps", "head_device": "cpu"})


def submit():
    from experiments.make_submission import check_submission
    if not winners():
        return
    test = load_split("test"); d = np.load(CACHE / "test_probabilities.npz")
    assert np.array_equal(d["ids"], test.id)
    b = np.loadtxt(ROOT / "results/deberta_blend_candidate/test_components.csv", delimiter=",", skiprows=1, usecols=1)
    n = logit(np.clip(d["probabilities"], 1e-12, 1 - 1e-12))
    exports = []
    for result in winners():
        output = ROOT / "submissions" / f"deberta_{result['model']}_8000.csv"
        exports.append(str(output.relative_to(ROOT)))
        if output.exists():
            assert sha(output) == json.loads(output.with_suffix(".json").read_text())["csv_sha256"]
            continue
        frame = pd.DataFrame({"id": test.id, "label": logit_mixture(b, n, result["neural_weight"])})
        check_submission(frame, test)
        frame.to_csv(output, index=False)
        write(output.with_suffix(".json"), {"model": result["model"], "neural_weight": result["neural_weight"],
                                           "training_rows": 8000, "prediction_rows": 3000,
                                           "validation_auroc": result["validation_auroc"],
                                           "selection_sha256": sha(OUT / "selection.json"),
                                           "test_provenance_sha256": sha(OUT / "test_provenance.json"),
                                           "csv_sha256": sha(output), "uploaded_to_kaggle": False,
                                           "note": "Three-seed text/style neural head plus original booster. Frozen fine-tuned encoder; not a 10,000-row refit."})
        print(f"Wrote and checked {output}; no Kaggle upload.", flush=True)
    write(OUT / "submission_gate.json", {"generated": exports, "uploaded": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["compare", "scores", "submit"], default="compare")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    setup()
    {"compare": compare, "scores": test_scores, "submit": submit}[args.stage]()


if __name__ == "__main__":
    main()
