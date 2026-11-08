"""Offline, recoverable single/three-seed supervised RoBERTa + curvature run.

All GPU stages use fresh subprocesses. No model downloads or unlabeled rows.
Run --stage option1 for cached comparison, --stage run for the overnight queue.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from experiments.robust_boost import load, paired_ci, splits
from src.curvature_roberta import aligned_probability, average_probabilities, blend, select_configuration
from src.deberta import device_for
from src.features import build_extended_features
from src.roberta_experiment import REVISION, frame_hash, load_model, predict, tokenizer, train_epoch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/curvature_roberta"
CACHE = ROOT / ".cache/curvature_roberta"
PREVIOUS = ROOT / ".cache/roberta_experiments"
SEEDS = [42, 43, 44]
CONFIG = {"encoder_lr": 2e-5, "head_lr": 1e-4, "weight_decay": .01,
          "batch_size": 4, "gradient_accumulation": 4, "eval_batch_size": 8,
          "max_length": 512, "epochs": 2, "precision": "fp32", "pseudo_weight": .25}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def context_frames(context):
    train, val, full = load()
    if context == "inner":
        return splits(train)["inner"]
    if context == "domain":
        return splits(train)["rare_topics"]
    if context == "fit":
        return train, val
    if context == "full":
        return full, pd.read_csv(ROOT / "Data/test.csv")
    raise ValueError(context)


def training_frame(frame):
    f = frame[["id", "text", "label"]].copy()
    f["soft_target"] = f.pop("label").astype(float)
    return f


def checkpoint(context, seed, epoch):
    if seed == 42 and context != "full":
        return PREVIOUS / context / "supervised" / f"epoch_{epoch}"
    return CACHE / context / f"seed_{seed}" / f"epoch_{epoch}"


def baseline(context, ids):
    if context == "full":
        d = pd.read_csv(CACHE / "reference_test.csv")
        return aligned_probability(ids, d.id, d.label)
    path = {"inner": ROOT / ".cache/robust_boost/selection_inner_curv_bag.csv",
            "domain": ROOT / ".cache/robust_boost/selection_rare_topics_curv_bag.csv",
            "fit": ROOT / "results/robust_boost/validation_curv_bag.csv"}[context]
    d = pd.read_csv(path)
    return aligned_probability(ids, d.id, d.probability)


def scores(context, seed, epoch):
    _, hold = context_frames(context)
    if seed == 42 and context != "full":
        p = PREVIOUS / f"{context}_supervised_epoch_{epoch}_scores.npz"
        d = np.load(p)
        return aligned_probability(hold.id, d["hold_ids"], d["hold"])
    d = np.load(CACHE / f"scores_{context}_{seed}_{epoch}.npz")
    return aligned_probability(hold.id, d["ids"], d["probability"])


def setup():
    OUT.mkdir(exist_ok=True, parents=True)
    CACHE.mkdir(exist_ok=True, parents=True)
    sources = ["src/curvature_roberta.py", "experiments/curvature_roberta.py",
               "src/roberta_experiment.py", "src/deberta.py", "src/blend_selection.py",
               "experiments/robust_boost.py", "src/features.py", "src/models.py"]
    dependencies = ["Data/train.csv", "Data/test.csv", "splits/val_ids.csv",
                    "results/robust_boost/protocol.json", "results/eda/cluster_assignments.csv",
                    ".cache/robust_boost/selection_inner_curv_bag.csv",
                    ".cache/robust_boost/selection_rare_topics_curv_bag.csv",
                    "results/robust_boost/validation_curv_bag.csv"]
    for context in ["inner", "domain", "fit"]:
        fit, hold = context_frames(context)
        for epoch in [1, 2]:
            old_path = checkpoint(context, 42, epoch)
            complete = json.loads((old_path / "complete.json").read_text())
            assert complete["signature"]["frame"] == frame_hash(training_frame(fit))
            assert complete["signature"]["pseudo"] is None
            assert complete["signature"]["configuration"] == {"seed": 42, **CONFIG}
            scores(context, 42, epoch)
            dependencies += [str(old_path.relative_to(ROOT) / "model.safetensors"),
                str((PREVIOUS / f"{context}_supervised_epoch_{epoch}_scores.npz").relative_to(ROOT))]
        assert set(fit.id).isdisjoint(hold.id)
        pd.DataFrame({"id": np.r_[fit.id, hold.id], "partition": ["fit"] * len(fit) + ["holdout"] * len(hold)}).to_csv(OUT / f"{context}_ids.csv", index=False)
    spec = {"seeds": SEEDS, "configuration": CONFIG, "epochs_considered": [1, 2],
        "model": "FacebookAI/roberta-base", "revision": REVISION,
        "no_downloads": True, "no_additional_unlabeled_examples": True,
        "option1": "Reuse supervised seed-42 inner/domain/fit predictions, verify frame/configuration hashes, select epoch and log-odds blend weight on training holdouts.",
        "option2": "Add independent pretrained-initialized seeds 43 and 44 in each context. Average the three probabilities at a common epoch, then blend with curvature. This is a seed ensemble, not a learned stack or five-fold OOF experiment.",
        "selection": "Weights [0,.01,.025,.05,.1,.2,.35,.5]; maximize .75 inner AUROC + .25 rare-topic AUROC. Rare-topic AUROC must be >= curvature-.0005. Ties: higher inner, smaller weight, earlier epoch. Both rules registered before option1 validation.",
        "validation": "Frozen choices only, same reused fixed 2000 rows. Option2 settings never depend on option1 validation. Paired 2000-resample stratified bootstrap; no validation tuning. Earlier topic clustering used all competition text, so topic holdout is a diagnostic rather than a fully inductive unseen-domain test.",
        "export_gate": "Positive preselected neural weight, training-only composite gain >= .000001, fixed validation no more than .0002 below curvature. Same gate for both options. This is a practical guardrail, not proof of a public-score gain.",
        "final_fit": "Only eligible methods. Refit needed neural seeds on all 10000 labels from cached pretrained weights, selected epoch with unchanged two-epoch scheduler horizon. Use teammate's already-generated 10000-row curv_bag test predictions from pinned remote commit. Never evaluate full-fit models on validation.",
        "checkpointing": "Fresh process per epoch/inference. Full optimizer/scheduler/RNG recovery state at completed epochs. Retry an interrupted epoch once, then stop visibly. Remove only newly-created optimizer states after required final epoch succeeds; preserve both weights for selection. No existing checkpoints modified.",
        "time_budget_seconds": 8 * 60 * 60,
        "sources": {p: sha(ROOT / p) for p in sources},
        "inputs": {p: sha(ROOT / p) for p in dependencies},
        "versions": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "numpy", "pandas", "scikit-learn"]}}
    path = OUT / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError("Frozen protocol/source/dependency changed")
    write(path, spec)
    reference = subprocess.check_output(["git", "show", "768828d6bfe9f3594333c7033ca4db44af94bd2a:submissions/improve_curv_bag.csv"], cwd=ROOT)
    ref_path = CACHE / "reference_test.csv"
    if ref_path.exists():
        assert ref_path.read_bytes() == reference
    else:
        ref_path.write_bytes(reference)
    test = pd.read_csv(ROOT / "Data/test.csv")
    assert np.array_equal(pd.read_csv(ref_path).id, test.id)
    write(OUT / "reference_test.json", {"remote_commit": "768828d6bfe9f3594333c7033ca4db44af94bd2a",
        "path": "submissions/improve_curv_bag.csv", "sha256": sha(ref_path), "rows": len(test),
        "public_score_user_reported": .98544})


def select_option(name, seeds):
    path = OUT / f"{name}_selection.json"
    if path.exists():
        return json.loads(path.read_text())
    _, inner = context_frames("inner")
    _, domain = context_frames("domain")
    choice = select_configuration(inner.label.astype(int), baseline("inner", inner.id),
        {e: average_probabilities([scores("inner", s, e) for s in seeds]) for e in [1, 2]},
        domain.label.astype(int), baseline("domain", domain.id),
        {e: average_probabilities([scores("domain", s, e) for s in seeds]) for e in [1, 2]})
    choice.update(seeds=seeds, protocol_sha256=sha(OUT / "protocol.json"))
    write(path, choice)
    pd.DataFrame(choice["candidates"]).to_csv(OUT / f"{name}_training_search.csv", index=False)
    print(f"Frozen {name}: {json.dumps(choice['chosen'])}", flush=True)
    return choice


def evaluate_option(name):
    path = OUT / f"{name}_validation.json"
    if path.exists():
        return json.loads(path.read_text())
    choice = json.loads((OUT / f"{name}_selection.json").read_text())
    _, val = context_frames("fit")
    epoch, weight = choice["chosen"]["epoch"], choice["chosen"]["weight"]
    n = average_probabilities([scores("fit", s, epoch) for s in choice["seeds"]])
    b = baseline("fit", val.id)
    p = blend(b, n, weight)
    y = val.label.astype(int).to_numpy()
    keep = build_extended_features(val).blank_lines.eq(0).to_numpy()
    auc, ref = roc_auc_score(y, p), roc_auc_score(y, b)
    result = {"option": name, "neural_seeds": choice["seeds"], "epoch": epoch, "neural_weight": weight,
        "validation_auroc": auc, "reference_auroc": ref, "gain": auc - ref,
        "paired_95_ci": paired_ci(y, p, b), "neural_auroc": roc_auc_score(y, n),
        "no_blank_auroc": roc_auc_score(y[keep], p[keep]),
        "reference_no_blank_auroc": roc_auc_score(y[keep], b[keep]),
        "errors_at_half": int(((p >= .5) != y).sum()), "reference_errors_at_half": int(((b >= .5) != y).sum()),
        "export_eligible": bool(weight > 0 and choice["training_only_gain"] >= .000001 and auc >= ref - .0002),
        "training_rows_per_neural_member": 8000, "validation_previously_reused": True,
        "selection_sha256": sha(OUT / f"{name}_selection.json")}
    pd.DataFrame({"id": val.id, "label": y, "curvature": b, "neural": n, "blend": p}).to_csv(OUT / f"{name}_validation_predictions.csv", index=False)
    write(path, result)
    print(json.dumps(result), flush=True)
    return result


def train_worker(context, seed, epoch):
    fit, _ = context_frames(context)
    config = {"seed": seed, **CONFIG}
    def progress(row):
        value = {"stage": "training", "context": context, "seed": seed, **row}
        write(OUT / "progress.json", value)
        print(json.dumps(value), flush=True)
    result = train_epoch(training_frame(fit), config, CACHE / context / f"seed_{seed}",
                         epoch, device_for("mps"), progress)
    write(OUT / f"{context}_{seed}_epoch_{epoch}.json", result)


def score_worker(context, seed, epoch):
    path = CACHE / f"scores_{context}_{seed}_{epoch}.npz"
    if path.exists():
        scores(context, seed, epoch)
        return
    _, hold = context_frames(context)
    model_path = checkpoint(context, seed, epoch)
    device = device_for("mps")
    model = load_model(checkpoint=model_path).to(device)
    def progress(n, total):
        value = {"stage": "inference", "context": context, "seed": seed, "epoch": epoch,
                 "rows": n, "total": total}
        write(OUT / "progress.json", value)
        print(json.dumps(value), flush=True)
    p = predict(model, hold, tokenizer(), device, CONFIG["eval_batch_size"], progress)
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, ids=hold.id.to_numpy(), probability=p)
    temp.replace(path)
    write(OUT / f"scores_{context}_{seed}_{epoch}.json", {"rows": len(hold), "checkpoint_sha256": sha(model_path / "model.safetensors"), "scores_sha256": sha(path)})


def smoke():
    fit, _ = context_frames("inner")
    cfg = {"seed": 51, **CONFIG, "batch_size": 2, "gradient_accumulation": 2,
           "max_length": 128, "epochs": 1}
    sample = fit.groupby("label").head(4).reset_index(drop=True)
    value = train_epoch(training_frame(sample), cfg, CACHE / "smoke", 1, device_for("mps"),
                         lambda row: print(json.dumps(row), flush=True))
    assert value["updates"] == 2 and value["training_rows"] == 8
    write(OUT / "smoke.json", {"passed": True, "updates": value["updates"], "rows": 8,
                               "no_downloads": True, "precision": "fp32"})


def child(stage, context="inner", seed=43, epoch=1, deadline=None):
    cmd = [sys.executable, "-u", "-m", "experiments.curvature_roberta", "--stage", stage,
           "--context", context, "--seed", str(seed), "--epoch", str(epoch)]
    for attempt in [1, 2]:
        remaining = deadline - time.monotonic() if deadline else None
        if remaining is not None and remaining <= 0:
            raise TimeoutError("Overnight time budget reached; rerun to resume")
        print(f"Launching {stage} {context} seed={seed} epoch={epoch} attempt={attempt}", flush=True)
        result = subprocess.run(cmd, cwd=ROOT, timeout=remaining)
        if result.returncode == 0:
            return
        if attempt == 2:
            raise RuntimeError(f"GPU child failed twice: {stage} {context} {seed} {epoch}")


def train_and_score(context, seed, epoch, deadline):
    path = checkpoint(context, seed, epoch)
    if not (path / "complete.json").exists():
        child("train", context, seed, epoch, deadline)
    # Existing seed-42 scores already exist and remain untouched.
    if seed != 42 or context == "full":
        child("score", context, seed, epoch, deadline)


def prune_optimizer_states(context, seed, final_epoch):
    # Only our newly generated optimizer states; inference weights are preserved.
    assert not (seed == 42 and context != "full")
    assert (checkpoint(context, seed, final_epoch) / "complete.json").exists()
    for epoch in range(1, final_epoch + 1):
        (checkpoint(context, seed, epoch) / "trainer_state.pt").unlink(missing_ok=True)


def export_option(name):
    result = json.loads((OUT / f"{name}_validation.json").read_text())
    if not result["export_eligible"]:
        print(f"{name} export guardrail failed", flush=True)
        return None
    _, test = context_frames("full")
    epoch = result["epoch"]
    n = average_probabilities([scores("full", s, epoch) for s in result["neural_seeds"]])
    p = blend(baseline("full", test.id), n, result["neural_weight"])
    sample = pd.read_csv(ROOT / "Data/sample_submission.csv")
    assert len(test) == 3000 and test.id.is_unique and np.array_equal(test.id, sample.id)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    path = ROOT / "submissions" / f"curvature_roberta_{name}_10000.csv"
    pd.DataFrame({"id": test.id, "label": p}).to_csv(path, index=False)
    write(path.with_suffix(".json"), {"csv_sha256": sha(path), "training_rows_per_component": 10000,
        "test_sha256": sha(ROOT / "Data/test.csv"), "validation_from_separate_8000_row_fits": result,
        "protocol_sha256": sha(OUT / "protocol.json"), "reference_test_sha256": sha(CACHE / "reference_test.csv"),
        "neural_checkpoints": [{"seed": s, "epoch": epoch, "sha256": sha(checkpoint("full", s, epoch) / "model.safetensors")} for s in result["neural_seeds"]],
        "uploaded_to_kaggle": False, "no_new_model_downloads": True, "unlabeled_examples_added": False})
    print(f"Wrote {path}", flush=True)
    return str(path.relative_to(ROOT))


def report(status):
    rows = []
    for name in ["single", "ensemble"]:
        path = OUT / f"{name}_validation.json"
        if path.exists():
            rows.append(json.loads(path.read_text()))
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "comparison.csv", index=False)
    lines = ["# Supervised RoBERTa + curvature", "", f"Status: **{status}**.", "",
        "All pretrained weights are local. No additional unlabeled examples, Kaggle upload, Git commit or push.", "",
        "## Fixed validation", "", "| Method | AUROC | Delta vs curvature | Neural weight | Epoch |", "| --- | ---: | ---: | ---: | ---: |",
        "| Curvature/bagging reference | 0.999775 | 0 | — | — |"]
    for r in rows:
        lines.append(f"| {r['option']} | {r['validation_auroc']:.9f} | {r['gain']:+.9f} | {r['neural_weight']} | {r['epoch']} |")
    lines += ["", "Both methods choose epoch/weight on the original 1600-row inner holdout and 148-row rare-topic holdout before their fixed-validation evaluation. The second method averages independent seeds 42/43/44; it is a seed ensemble rather than an OOF-trained stacking model. The rare holdout is small and existing EDA clusters used all competition text. Fixed validation has been repeatedly reused in this project; near-perfect point scores do not establish public-score gains.", "",
        "`protocol.json` freezes both choices' rules and sources before option1 validation. `single_selection.json` / `ensemble_selection.json` contain all training-only candidates. Validation JSON files include paired bootstrap intervals and no-blank-line scores. Eligible submissions refit all neural components on 10000 labels after selection; their validation scores refer to separate 8000-row models.", "",
        "Progress: `status.json`, `progress.json`, and `run.log`. Resume with:", "", "```bash", "caffeinate -i .venv/bin/python -u -m experiments.curvature_roberta --stage run", "```", "",
        "Completed GPU epochs are reused. An interruption inside an epoch repeats that epoch from its previous optimizer/RNG checkpoint. Optimizer state is removed only after all required epochs for a newly trained model finish. Pre-existing checkpoints are preserved. Keep the Mac plugged in with its lid open for the run."]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def run():
    deadline = time.monotonic() + 8 * 60 * 60
    write(OUT / "status.json", {"status": "running", "pid": os.getpid(), "budget_hours": 8})
    try:
        select_option("single", [42])
        evaluate_option("single")
        report("single complete; ensemble running")
        if not (OUT / "smoke.json").exists():
            child("smoke", deadline=deadline)
        # Seed-42 models and predictions are reused. Each new seed starts from pretrained weights.
        for context in ["inner", "domain"]:
            for seed in [43, 44]:
                for epoch in [1, 2]:
                    train_and_score(context, seed, epoch, deadline)
                prune_optimizer_states(context, seed, 2)
        selection = select_option("ensemble", SEEDS)
        selected_epoch = selection["chosen"]["epoch"]
        for seed in [43, 44]:
            for epoch in range(1, selected_epoch + 1):
                train_and_score("fit", seed, epoch, deadline)
            prune_optimizer_states("fit", seed, selected_epoch)
        evaluate_option("ensemble")
        report("both comparisons complete; eligible full-data refits running")
        needed = {}
        for name in ["single", "ensemble"]:
            result = json.loads((OUT / f"{name}_validation.json").read_text())
            if result["export_eligible"]:
                for seed in result["neural_seeds"]:
                    needed[seed] = max(needed.get(seed, 0), result["epoch"])
        for seed, epochs in sorted(needed.items()):
            for epoch in range(1, epochs + 1):
                train_and_score("full", seed, epoch, deadline)
            prune_optimizer_states("full", seed, epochs)
        outputs = [p for name in ["single", "ensemble"] if (p := export_option(name))]
        write(OUT / "status.json", {"status": "complete", "submissions": outputs,
                                    "kaggle_uploaded": False, "no_downloads": True})
        report("complete")
    except BaseException as error:
        write(OUT / "status.json", {"status": "failed_or_interrupted", "error": repr(error),
                                    "resume": "python -m experiments.curvature_roberta --stage run"})
        report("failed or interrupted; see status.json")
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["option1", "run", "train", "score", "smoke"], required=True)
    ap.add_argument("--context", choices=["inner", "domain", "fit", "full"], default="inner")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--epoch", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(4)
    if args.stage in ["option1", "run"]:
        setup()
    if args.stage == "option1":
        select_option("single", [42])
        evaluate_option("single")
        report("single complete; ensemble not started")
    elif args.stage == "run":
        run()
    elif args.stage == "train":
        train_worker(args.context, args.seed, args.epoch)
    elif args.stage == "score":
        score_worker(args.context, args.seed, args.epoch)
    elif args.stage == "smoke":
        smoke()


if __name__ == "__main__":
    main()
