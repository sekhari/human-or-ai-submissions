"""RoBERTa control, agreement-pseudo-labeling and task-adaptive pretraining.

    caffeinate -i .venv/bin/python -u -m experiments.roberta_experiments

All jobs run sequentially; every GPU epoch/inference job gets a fresh process.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import roc_auc_score
import torch
from threadpoolctl import threadpool_limits

from experiments.deberta_combinations import partitions
from src.blend_selection import WEIGHTS, logit_mixture
from src.data import load_split
from src.deberta import device_for
from src.roberta_experiment import MODEL_ID, REVISION, agreement_pseudo, load_model, predict, tokenizer, train_epoch
from src.split import VAL_IDS_PATH

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/roberta_experiments"
CACHE = ROOT / ".cache/roberta_experiments"
ARMS = ["supervised", "pseudo", "tapt"]
CONFIG = {"seed": 42, "encoder_lr": 2e-5, "head_lr": 1e-4, "weight_decay": .01,
          "batch_size": 4, "gradient_accumulation": 4, "eval_batch_size": 8,
          "max_length": 512, "epochs": 2, "precision": "fp32", "pseudo_weight": .25}
MLM_CONFIG = {**CONFIG, "epochs": 1, "encoder_lr": 5e-6, "head_lr": 5e-6}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n"); temporary.replace(path)


def setup():
    OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    record = {"model_id": MODEL_ID, "revision": REVISION, "license": "MIT", "configuration": CONFIG,
              "mlm_configuration": MLM_CONFIG, "arms": ARMS, "blend_weights": WEIGHTS,
              "input_sha256": {str(p): sha(ROOT / p) for p in ["Data/train.csv", "Data/unlabeled.csv", "splits/val_ids.csv", "results/eda/cluster_assignments.csv", "results/eda/near_duplicates.csv"]},
              "source_sha256": {p: sha(ROOT / p) for p in ["src/roberta_experiment.py", "experiments/roberta_experiments.py", "src/features.py", "src/models.py", "src/blend_selection.py"]},
              "selection": "Original 6,400/1,600 inner split within 8,000 fitting rows. Each arm chooses epoch 1/2 and mixture weight by inner blend AUROC; ties prefer higher neural AUROC, earlier epoch, smaller weight. Reference mixture is existing DeBERTa/style/boost blend, reconstructed for the inner holdout. The original holdout has been used before; it is not independent.",
              "pseudo": "Teachers are supervised RoBERTa (epoch with highest inner neural AUROC) and the original OOF-feature booster, independently fit to the fitting partition. Require agreement, boosting confidence >=0.99 and neural confidence >=0.90. Targets are mean teacher probabilities. Add one pseudo microbatch at weight 0.25 per supervised optimizer update. Same real-row exposure and optimizer update count as control. No pseudo threshold tuning; quality measured on mock-unlabeled inner holdout.",
              "tapt": "One masked-language-model epoch on fitting-partition text plus allowed unlabeled.csv rows; labels discarded. 15% dynamic masking with 80/10/10 replacement, encoder and MLM head LR 5e-6. Vocabulary logits computed only at masked positions. Fresh classification head identical to the supervised control, then same supervised schedule.",
              "domain_holdout": "Remove story/poetry-like clusters 0/2 from original 8,000 training rows. Also exclude fitting near-duplicates of held-out rows using committed EDA pairs. Predict the 148 held-out labeled rows; fixed validation remains excluded. Unlabeled passages from these styles are allowed, except exact/recorded near-copies of evaluation text. Hyperparameters/durations are frozen from inner selection. This is one small, noisy topic holdout, not full leave-one-domain-out CV or an unseen-generator test.",
              "unlabeled_protection": "For every context exclude exact text and recorded near-duplicate neighbors of its heldout rows and the fixed validation rows. Test passages are not pseudo-labeled or used for teacher/gradient training. Existing unsupervised EDA clusters (which included test text, never test labels) inform the topic stress-test choice. Test predictions do not select checkpoints or methods.",
              "arm_selection": "Highest inner blend AUROC; ties prefer higher domain neural AUROC, then higher inner neural AUROC. Domain results are a training-only selection signal, not an untouched final test. Retain one preselected arm before fixed validation evaluation; refit all three frozen arm settings to report the comparison.",
              "final": "Refit all three arms on 8,000 fitting rows, using inner-selected adaptation durations and the same two-epoch LR horizon. Evaluate all frozen choices once on the reused fixed 2,000 validation rows. No sweep or arm switch based on fixed validation.",
              "export_gate": "Only preselected arm: positive neural weight, fixed-validation blend exceeds previous 0.999485, domain blend no worse than its domain boosting reference. No Kaggle upload or Git push.",
              "packages": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "numpy", "pandas", "scikit-learn"]},
              "sources": ["https://huggingface.co/FacebookAI/roberta-base", "https://aclanthology.org/2020.acl-main.740/"]}
    path = OUT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == record, "Protocol changed; preserve the run."
    else:
        write(path, record)


def raw_context(context):
    train, val, fitting = partitions()
    if context == "inner":
        fit, hold = train.loc[fitting].copy(), train.loc[~fitting].copy()
    elif context == "fit":
        fit, hold = train.copy(), val.copy()
    elif context == "domain":
        clusters = pd.read_csv(ROOT / "results/eda/cluster_assignments.csv").set_index("id")
        rare = clusters.loc[train.id, "cluster"].isin([0, 2]).to_numpy()
        fit, hold = train.loc[~rare].copy(), train.loc[rare].copy()
        pairs = pd.read_csv(ROOT / "results/eda/near_duplicates.csv")
        held = set(hold.id)
        neighbors = set(pairs.loc[pairs.id_a.isin(held), "id_b"]) | set(pairs.loc[pairs.id_b.isin(held), "id_a"])
        fit = fit.loc[~fit.id.isin(neighbors)].copy()
    else:
        raise ValueError(context)
    assert set(fit.id).isdisjoint(hold.id) and set(fit.id).isdisjoint(val.id)
    unlab = load_split("unlabeled")
    protected = pd.concat([hold, val], ignore_index=True)
    protected_ids = set(protected.id)
    pairs = pd.read_csv(ROOT / "results/eda/near_duplicates.csv")
    neighbors = set(pairs.loc[pairs.id_a.isin(protected_ids), "id_b"]) | set(pairs.loc[pairs.id_b.isin(protected_ids), "id_a"])
    unlab = unlab.loc[~unlab.id.isin(neighbors) & ~unlab.text.isin(set(protected.text))].copy()
    assert set(unlab.id).isdisjoint(protected_ids)
    return fit.reset_index(drop=True), hold.reset_index(drop=True), unlab.reset_index(drop=True)


def prepare():
    tok = tokenizer()
    for context in ["inner", "domain", "fit"]:
        fit, hold, unlab = raw_context(context)
        pd.DataFrame({"id": np.r_[fit.id, hold.id, unlab.id], "partition": ["fit"]*len(fit) + ["holdout"]*len(hold) + ["unlabeled"]*len(unlab)}).to_csv(OUT / f"{context}_ids.csv", index=False)
        counts = tok(fit.text.tolist(), truncation=False, padding=False)["input_ids"]
        write(OUT / f"{context}_data.json", {"fitting_rows": len(fit), "heldout_rows": len(hold), "unlabeled_rows": len(unlab),
                                             "excluded_unlabeled_rows": 3000-len(unlab), "fit_heldout_disjoint": True,
                                             "fraction_fitting_over_512_tokens": float(np.mean([len(x)>512 for x in counts]))})
        print(context, len(fit), len(hold), len(unlab), flush=True)
    model_snapshot = ROOT / ".cache/huggingface/models--FacebookAI--roberta-base/snapshots" / REVISION
    write(OUT / "pretrained.json", {"model_id": MODEL_ID, "revision": REVISION,
                                    "model_sha256": sha(model_snapshot / "model.safetensors"),
                                    "vocab_sha256": sha(model_snapshot / "vocab.json"), "license": "MIT"})


def boost(context):
    path = CACHE / f"{context}_boost.npz"
    fit, hold, unlab = raw_context(context)
    if path.exists():
        d = np.load(path); assert np.array_equal(d["hold_ids"], hold.id) and np.array_equal(d["unlabeled_ids"], unlab.id)
        return
    from experiments.deberta_finetune import baseline_prediction
    evaluation = pd.concat([hold, unlab], ignore_index=True)
    with threadpool_limits(limits=2):
        p = baseline_prediction(fit, evaluation)
    if context == "inner":
        old = pd.read_csv(ROOT / ".cache/deberta_combinations/inner_boosting.csv")
        assert np.array_equal(old.id, hold.id)
        np.testing.assert_allclose(p[:len(hold)], old.boosting, atol=1e-12, rtol=1e-12)
    if context == "fit":
        old = pd.read_csv(ROOT / "results/deberta_hybrid_blend/val_predictions.csv")
        assert np.array_equal(old.id, hold.id)
        np.testing.assert_allclose(p[:len(hold)], old.boosting_reference, atol=1e-12, rtol=1e-12)
    np.savez_compressed(path, hold_ids=hold.id.to_numpy(), hold=p[:len(hold)], unlabeled_ids=unlab.id.to_numpy(), unlabeled=p[len(hold):])
    write(OUT / f"{context}_boost.json", {"fitting_rows": len(fit), "heldout_rows": len(hold), "unlabeled_rows": len(unlab),
                                         "fit_ids_sha256": hashlib.sha256(fit.id.to_numpy().tobytes()).hexdigest(),
                                         "cache_sha256": sha(path)})


def train(context, arm, epoch):
    fit, _, unlab = raw_context(context)
    adapted, pseudo = None, None
    if arm == "mlm":
        frame = pd.concat([fit[["id", "text"]], unlab[["id", "text"]]], ignore_index=True)
        assert "label" not in frame
        config = MLM_CONFIG; task = "mlm"
    else:
        frame = fit[["id", "text", "label"]].copy()
        frame["soft_target"] = frame.pop("label").astype(float)
        config = CONFIG; task = "classification"
        if arm == "tapt":
            adapted = CACHE / context / "mlm/epoch_1"
        if arm == "pseudo":
            pseudo = pd.read_pickle(CACHE / f"{context}_pseudo.pkl")
    def progress(row):
        value = {"context": context, "arm": arm, **row}
        write(OUT / "progress.json", value)
        print(json.dumps(value), flush=True)
    result = train_epoch(frame, config, CACHE / context / arm, epoch, device_for("mps"), progress, task, pseudo, adapted)
    write(OUT / f"{context}_{arm}_epoch_{epoch}.json", result)


def score(context, arm, epoch):
    path = CACHE / f"{context}_{arm}_epoch_{epoch}_scores.npz"
    if path.exists():
        return
    _, hold, unlab = raw_context(context)
    # Supervised teachers predict both evaluation and U; other arms need only evaluation.
    frame = pd.concat([hold, unlab], ignore_index=True) if arm == "supervised" else hold
    device = device_for("mps")
    model = load_model(checkpoint=CACHE / context / arm / f"epoch_{epoch}").to(device)
    def progress(n, total):
        print(f"{context} {arm} epoch {epoch} predictions {n}/{total}", flush=True)
    p = predict(model, frame, tokenizer(), device, CONFIG["eval_batch_size"], progress)
    np.savez_compressed(path, hold_ids=hold.id.to_numpy(), hold=p[:len(hold)],
                        unlabeled_ids=unlab.id.to_numpy() if arm == "supervised" else np.array([], dtype=np.int64),
                        unlabeled=p[len(hold):])
    write(OUT / f"{context}_{arm}_epoch_{epoch}_scores.json", {"rows": len(frame), "checkpoint_sha256": sha(CACHE / context / arm / f"epoch_{epoch}/model.safetensors"), "scores_sha256": sha(path)})


def teacher_epoch():
    _, hold, _ = raw_context("inner")
    return max([1, 2], key=lambda e: (roc_auc_score(hold.label, np.load(CACHE / f"inner_supervised_epoch_{e}_scores.npz")["hold"]), -e))


def pseudo(context):
    path = CACHE / f"{context}_pseudo.pkl"
    if path.exists():
        return
    _, hold, unlab = raw_context(context)
    epoch = teacher_epoch()
    b = np.load(CACHE / f"{context}_boost.npz")
    n = np.load(CACHE / f"{context}_supervised_epoch_{epoch}_scores.npz")
    assert np.array_equal(b["unlabeled_ids"], unlab.id) and np.array_equal(n["unlabeled_ids"], unlab.id)
    keep, targets = agreement_pseudo(b["unlabeled"], n["unlabeled"])
    extra = unlab.loc[keep, ["id", "text"]].copy(); extra["soft_target"] = targets[keep]
    extra.to_pickle(path)
    selected = extra[["id", "soft_target"]].copy(); selected.to_csv(OUT / f"{context}_pseudo_labels.csv", index=False)
    write(OUT / f"{context}_pseudo.json", {"teacher_epoch": epoch, "available_unlabeled_rows": len(unlab), "selected_rows": len(extra),
                                           "pseudo_human_rows": int((extra.soft_target < .5).sum()), "pseudo_ai_rows": int((extra.soft_target >= .5).sum()),
                                           "mean_selected_confidence": float(np.maximum(extra.soft_target, 1-extra.soft_target).mean()) if len(extra) else None,
                                           "pseudo_data_sha256": sha(path)})
    if context == "inner":
        assert np.array_equal(b["hold_ids"], hold.id) and np.array_equal(n["hold_ids"], hold.id)
        mock_keep, mock_targets = agreement_pseudo(b["hold"], n["hold"])
        write(OUT / "mock_pseudo_quality.json", {"mock_rows": len(hold), "accepted_rows": int(mock_keep.sum()),
                                                "coverage": float(mock_keep.mean()),
                                                "accepted_hard_label_accuracy": float(((mock_targets[mock_keep] >= .5) == hold.label.to_numpy()[mock_keep]).mean()) if mock_keep.any() else None,
                                                "thresholds_fixed_in_advance": True,
                                                "caution": "This inner holdout also selects the teacher epoch; quality is conditional on that selection, not an independent precision guarantee."})
    print(f"{context} selected {len(extra)}/{len(unlab)} pseudo-labels.", flush=True)


def reference(context):
    _, hold, _ = raw_context(context)
    if context == "inner":
        d = np.load(ROOT / ".cache/deberta_hybrid_blend/inner_probabilities.npz")
        assert np.array_equal(d["ids"], hold.id)
        return logit_mixture(d["boosting"], logit(np.clip(d["neural"], 1e-12, 1-1e-12)), .5)
    if context == "fit":
        d = pd.read_csv(ROOT / "results/deberta_hybrid_blend/val_predictions.csv")
        assert np.array_equal(d.id, hold.id)
        return d.hybrid_blend_maximum.to_numpy()
    return np.load(CACHE / "domain_boost.npz")["hold"]


def select_inner():
    path = OUT / "inner_selection.json"
    if path.exists():
        return
    _, hold, _ = raw_context("inner")
    base = reference("inner")
    rows, choices = [], {}
    for arm in ARMS:
        candidates = []
        for epoch in [1, 2]:
            d = np.load(CACHE / f"inner_{arm}_epoch_{epoch}_scores.npz")
            assert np.array_equal(d["hold_ids"], hold.id)
            for weight in WEIGHTS:
                p = logit_mixture(base, logit(np.clip(d["hold"], 1e-12, 1-1e-12)), weight)
                row = {"arm": arm, "epoch": epoch, "weight": weight, "inner_blend_auroc": roc_auc_score(hold.label, p),
                       "inner_neural_auroc": roc_auc_score(hold.label, d["hold"])}
                rows.append(row); candidates.append(row)
        choices[arm] = dict(max(candidates, key=lambda r: (r["inner_blend_auroc"], r["inner_neural_auroc"], -r["epoch"], -r["weight"])))
    write(path, {"arms": choices, "teacher_epoch": teacher_epoch(), "reference_inner_auroc": roc_auc_score(hold.label, base), "protocol_sha256": sha(OUT / "protocol.json")})
    pd.DataFrame(rows).to_csv(OUT / "inner_search.csv", index=False)
    print("Inner arm selections:", json.dumps(choices), flush=True)


def select_arm():
    path = OUT / "selection.json"
    if path.exists():
        return
    inner = json.loads((OUT / "inner_selection.json").read_text())
    _, hold, _ = raw_context("domain")
    base = reference("domain")
    rows = []
    for arm, choice in inner["arms"].items():
        d = np.load(CACHE / f"domain_{arm}_epoch_{choice['epoch']}_scores.npz")
        assert np.array_equal(d["hold_ids"], hold.id)
        p = logit_mixture(base, logit(np.clip(d["hold"], 1e-12, 1-1e-12)), choice["weight"])
        rows.append({**choice, "domain_neural_auroc": roc_auc_score(hold.label, d["hold"]),
                     "domain_blend_auroc": roc_auc_score(hold.label, p), "domain_boosting_auroc": roc_auc_score(hold.label, base), "domain_rows": len(hold)})
    chosen = max(rows, key=lambda r: (r["inner_blend_auroc"], r["domain_neural_auroc"], r["inner_neural_auroc"]))
    write(path, {"chosen": chosen, "all_arms": rows, "inner_selection_sha256": sha(OUT / "inner_selection.json"),
                 "fixed_validation_used_for_selection": False})
    pd.DataFrame(rows).to_csv(OUT / "training_only_comparison.csv", index=False)
    print("Frozen final choice:", json.dumps(chosen), flush=True)


def evaluate():
    if (OUT / "evaluation.json").exists():
        return
    from experiments.neural_comparison import paired_ci
    _, val, _ = raw_context("fit")
    selection = json.loads((OUT / "selection.json").read_text())
    choices = json.loads((OUT / "inner_selection.json").read_text())["arms"]
    old = reference("fit"); y = val.label.to_numpy()
    table = val[["id", "label"]].copy(); table["previous_blend"] = old
    rows = [{"model": "previous_blend", "validation_auroc": roc_auc_score(y, old), "delta_previous_blend": 0., "paired_95ci_low": 0., "paired_95ci_high": 0.}]
    for arm in ARMS:
        choice = choices[arm]
        d = np.load(CACHE / f"fit_{arm}_epoch_{choice['epoch']}_scores.npz")
        assert np.array_equal(d["hold_ids"], val.id)
        neural = d["hold"]; p = logit_mixture(old, logit(np.clip(neural, 1e-12, 1-1e-12)), choice["weight"])
        table[f"{arm}_neural"], table[f"{arm}_blend"] = neural, p
        for kind, scores in [("neural", neural), ("blend", p)]:
            lo, hi = paired_ci(y, scores, old)
            rows.append({"model": f"{arm}_{kind}", "validation_auroc": roc_auc_score(y, scores), "delta_previous_blend": roc_auc_score(y, scores)-roc_auc_score(y, old),
                         "paired_95ci_low": lo, "paired_95ci_high": hi, "errors_at_0_5": int(((scores>=.5)!=y).sum()), "epoch": choice["epoch"], "neural_weight": choice["weight"]})
    table.to_csv(OUT / "val_predictions.csv", index=False); pd.DataFrame(rows).to_csv(OUT / "comparison.csv", index=False)
    chosen = selection["chosen"]
    selected_auc = roc_auc_score(y, table[f"{chosen['arm']}_blend"])
    eligible = bool(chosen["weight"] > 0 and selected_auc > roc_auc_score(y, old) and chosen["domain_blend_auroc"] >= chosen["domain_boosting_auroc"])
    write(OUT / "evaluation.json", {"results": rows, "chosen_arm_before_validation": chosen["arm"], "selected_blend_validation_auroc": selected_auc,
                                    "selection_sha256": sha(OUT / "selection.json"), "candidate_eligible": eligible,
                                    "training_rows": 8000, "validation_rows": 2000, "fixed_validation_reused": True, "uploaded_to_kaggle": False})
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def submit():
    report = json.loads((OUT / "evaluation.json").read_text())
    if not report["candidate_eligible"]:
        write(OUT / "submission_gate.json", {"generated": False, "uploaded": False, "reason": "Preselected arm failed at least one of the frozen candidate gates; no validation-based arm switch."})
        return
    from experiments.make_submission import check_submission
    test = load_split("test")
    choice = json.loads((OUT / "selection.json").read_text())["chosen"]
    device = device_for("mps")
    model = load_model(checkpoint=CACHE / "fit" / choice["arm"] / f"epoch_{choice['epoch']}").to(device)
    p = predict(model, test, tokenizer(), device, progress=lambda n, total: print(f"Selected test {n}/{total}", flush=True))
    old = np.load(ROOT / ".cache/deberta_hybrid_blend/test_probabilities.npz")
    assert np.array_equal(old["ids"], test.id)
    components = ROOT / "results/deberta_blend_candidate/test_components.csv"
    ids = pd.read_csv(components).id; assert np.array_equal(ids, test.id)
    boost_p = np.loadtxt(components, delimiter=",", skiprows=1, usecols=1)
    previous = logit_mixture(boost_p, logit(np.clip(old["probabilities"], 1e-12, 1-1e-12)), .5)
    scores = logit_mixture(previous, logit(np.clip(p, 1e-12, 1-1e-12)), choice["weight"])
    frame = pd.DataFrame({"id": test.id, "label": scores}); check_submission(frame, test)
    output = ROOT / "submissions" / f"roberta_{choice['arm']}_blend_8000.csv"
    frame.to_csv(output, index=False)
    np.savez_compressed(CACHE / "test_components.npz", ids=test.id.to_numpy(), previous=previous, roberta=p)
    write(output.with_suffix(".json"), {"method": choice, "validation_auroc": report["selected_blend_validation_auroc"], "training_rows": 8000,
                                         "csv_sha256": sha(output), "selection_sha256": sha(OUT / "selection.json"), "uploaded_to_kaggle": False})
    write(OUT / "submission_gate.json", {"generated": str(output.relative_to(ROOT)), "uploaded": False})


def run_job(name):
    env = dict(os.environ, TOKENIZERS_PARALLELISM="false", PYTORCH_ENABLE_MPS_FALLBACK="1")
    print(f"Starting RoBERTa job: {name}", flush=True)
    write(OUT / "status.json", {"state": "running", "job": name})
    status = subprocess.run([sys.executable, "-u", "-m", "experiments.roberta_experiments", "--job", name], cwd=ROOT, env=env).returncode
    if status:
        write(OUT / "status.json", {"state": "failed", "job": name, "exit_code": status})
        raise SystemExit(status)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--job", default="all"); args = p.parse_args()
    torch.set_num_threads(4); setup()
    if args.job == "all":
        run_job("prepare")
        for context in ["inner", "domain", "fit"]:
            run_job(f"boost:{context}")
        for epoch in [1, 2]:
            run_job(f"train:inner:supervised:{epoch}"); run_job(f"score:inner:supervised:{epoch}")
        run_job("pseudo:inner")
        run_job("train:inner:mlm:1")
        for arm in ["pseudo", "tapt"]:
            for epoch in [1, 2]:
                run_job(f"train:inner:{arm}:{epoch}"); run_job(f"score:inner:{arm}:{epoch}")
        run_job("select_inner")
        choices = json.loads((OUT / "inner_selection.json").read_text())
        for context in ["domain", "fit"]:
            control_epochs = max(choices["teacher_epoch"], choices["arms"]["supervised"]["epoch"])
            for epoch in range(1, control_epochs + 1):
                run_job(f"train:{context}:supervised:{epoch}")
            for epoch in sorted({choices["teacher_epoch"], choices["arms"]["supervised"]["epoch"]}):
                run_job(f"score:{context}:supervised:{epoch}")
            run_job(f"pseudo:{context}")
            run_job(f"train:{context}:mlm:1")
            for arm in ["pseudo", "tapt"]:
                for epoch in range(1, choices["arms"][arm]["epoch"] + 1):
                    run_job(f"train:{context}:{arm}:{epoch}")
                run_job(f"score:{context}:{arm}:{choices['arms'][arm]['epoch']}")
            if context == "domain":
                run_job("select_arm")
        run_job("evaluate"); run_job("submit")
        write(OUT / "status.json", {"state": "complete", "uploaded_to_kaggle": False})
    elif args.job in ["prepare", "select_inner", "select_arm", "evaluate", "submit"]:
        {"prepare": prepare, "select_inner": select_inner, "select_arm": select_arm, "evaluate": evaluate, "submit": submit}[args.job]()
    else:
        action, context, *rest = args.job.split(":")
        if action == "boost": boost(context)
        elif action == "pseudo": pseudo(context)
        elif action == "train": train(context, rest[0], int(rest[1]))
        elif action == "score": score(context, rest[0], int(rest[1]))
        else: raise ValueError(args.job)


if __name__ == "__main__":
    main()
