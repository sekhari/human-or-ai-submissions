"""Verify completed RoBERTa experiments and produce descriptive figures."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import roc_auc_score
import torch

from experiments.roberta_experiments import ARMS, CACHE, CONFIG, MLM_CONFIG, OUT, ROOT
from experiments.roberta_experiments import raw_context, reference, sha, write
from src.blend_selection import logit_mixture
from src.roberta_experiment import agreement_pseudo, frame_hash, load_model
from src.features import build_features


def main():
    protocol = json.loads((OUT / "protocol.json").read_text())
    inner = json.loads((OUT / "inner_selection.json").read_text())
    chosen = json.loads((OUT / "selection.json").read_text())
    evaluation = json.loads((OUT / "evaluation.json").read_text())
    for path, digest in {**protocol["input_sha256"], **protocol["source_sha256"]}.items():
        assert sha(ROOT / path) == digest
    dependencies = json.loads((OUT / "dependency_snapshot.json").read_text())
    for path, digest in dependencies["sha256"].items():
        assert sha(ROOT / path) == digest
    assert inner["protocol_sha256"] == sha(OUT / "protocol.json")
    assert chosen["inner_selection_sha256"] == sha(OUT / "inner_selection.json")
    assert evaluation["selection_sha256"] == sha(OUT / "selection.json")
    tests = json.loads((OUT / "tests.json").read_text())
    assert tests["passed"] and tests["tests_passed"] == 5
    assert tests["test_source_sha256"] == sha(ROOT / "tests/test_roberta_experiment.py")
    for name in ["benchmark.json", "mlm_smoke.json"]:
        assert json.loads((OUT / name).read_text())["passed"]
    checks, fold_details, ranking_details, layout_rows = [], [], [], []
    search = pd.read_csv(OUT / "inner_search.csv")
    for arm in ARMS:
        best = search.loc[search.arm.eq(arm)].sort_values(
            ["inner_blend_auroc", "inner_neural_auroc", "epoch", "weight"],
            ascending=[False, False, True, True]).iloc[0]
        frozen = inner["arms"][arm]
        assert int(best.epoch) == frozen["epoch"] and best.weight == frozen["weight"]
    for context in ["inner", "domain", "fit"]:
        fit, hold, unlab = raw_context(context)
        assert set(fit.id).isdisjoint(hold.id) and set(fit.id).isdisjoint(unlab.id)
        assert set(hold.text).isdisjoint(unlab.text)
        data_info = json.loads((OUT / f"{context}_data.json").read_text())
        assert (data_info["fitting_rows"], data_info["heldout_rows"], data_info["unlabeled_rows"]) == (len(fit), len(hold), len(unlab))
        boost = np.load(CACHE / f"{context}_boost.npz")
        neural = np.load(CACHE / f"{context}_supervised_epoch_{inner['teacher_epoch']}_scores.npz")
        assert np.array_equal(boost["unlabeled_ids"], unlab.id) and np.array_equal(neural["unlabeled_ids"], unlab.id)
        keep, targets = agreement_pseudo(boost["unlabeled"], neural["unlabeled"])
        pseudo = pd.read_pickle(CACHE / f"{context}_pseudo.pkl")
        assert np.array_equal(pseudo.id, unlab.loc[keep, "id"])
        np.testing.assert_allclose(pseudo.soft_target, targets[keep], atol=1e-15)
        pseudo_info = json.loads((OUT / f"{context}_pseudo.json").read_text())
        assert pseudo_info["selected_rows"] == len(pseudo) and pseudo_info["pseudo_data_sha256"] == sha(CACHE / f"{context}_pseudo.pkl")
        detail = {"context": context, "fitting_rows": len(fit), "evaluation_rows": len(hold), "pseudo_rows": len(pseudo)}
        if context in {"inner", "domain"}:
            mock_keep, mock_targets = agreement_pseudo(boost["hold"], neural["hold"])
            detail.update(mock_pseudo_accepted=int(mock_keep.sum()),
                          mock_pseudo_accuracy=float(((mock_targets[mock_keep]>=.5)==hold.label.to_numpy()[mock_keep]).mean()) if mock_keep.any() else None)
        fold_details.append(detail)
        for arm in [*ARMS, "mlm"]:
            paths = sorted((CACHE / context / arm).glob("epoch_*/complete.json"))
            assert paths
            for path in paths:
                complete = json.loads(path.read_text()); epoch = complete["epoch"]
                is_mlm = arm == "mlm"
                expected = pd.concat([fit[["id", "text"]], unlab[["id", "text"]]], ignore_index=True) if is_mlm else fit[["id", "text", "label"]].rename(columns={"label": "soft_target"})
                if not is_mlm:
                    expected["soft_target"] = expected.soft_target.astype(float)
                config = MLM_CONFIG if is_mlm else CONFIG
                assert complete["signature"]["frame"] == frame_hash(expected)
                assert complete["signature"]["configuration"] == config
                assert complete["training_rows"] == len(expected)
                expected_updates = math.ceil(math.ceil(len(expected) / config["batch_size"]) / config["gradient_accumulation"]) * epoch
                assert complete["updates"] == expected_updates
                if arm == "pseudo":
                    assert complete["signature"]["pseudo"] == frame_hash(pseudo)
                    assert complete["pseudo_batches_used"] == (expected_updates // epoch if len(pseudo) else 0)
                else:
                    assert complete["pseudo_batches_used"] == 0
                checks.append({"context": context, "arm": arm, "epoch": epoch, "updates": complete["updates"],
                               "weights_sha256": sha(path.parent / "model.safetensors")})
        for arm in ARMS:
            epoch = inner["arms"][arm]["epoch"]
            scores = np.load(CACHE / f"{context}_{arm}_epoch_{epoch}_scores.npz")
            assert np.array_equal(scores["hold_ids"], hold.id)
            assert np.isfinite(scores["hold"]).all() and ((scores["hold"]>=0) & (scores["hold"]<=1)).all()
            meta = json.loads((OUT / f"{context}_{arm}_epoch_{epoch}_scores.json").read_text())
            assert meta["checkpoint_sha256"] == sha(CACHE / context / arm / f"epoch_{epoch}/model.safetensors")
            assert meta["scores_sha256"] == sha(CACHE / f"{context}_{arm}_epoch_{epoch}_scores.npz")
            y = hold.label.to_numpy()
            ref = reference(context)
            mixed = logit_mixture(ref, logit(np.clip(scores["hold"], 1e-12, 1-1e-12)), inner["arms"][arm]["weight"])
            def reversed_pairs(p):
                return p[y==0, None] > p[y==1][None, :]
            before, after, neural_bad = reversed_pairs(ref), reversed_pairs(mixed), reversed_pairs(scores["hold"])
            ranking_details.append({"context": context, "arm": arm, "reference_reversed_pairs": int(before.sum()),
                                    "blend_reversed_pairs": int(after.sum()), "corrected_by_blend": int((before & ~after).sum()),
                                    "introduced_by_blend": int((~before & after).sum()),
                                    "reference_neural_shared_reversed_pairs": int((before & neural_bad).sum())})
            blank = build_features(hold).blank_lines.to_numpy() > 0
            for label, mask in [("no_blank_lines", ~blank), ("has_blank_lines", blank)]:
                if len(np.unique(y[mask])) < 2:
                    continue
                for name, p in [("reference", ref), ("neural", scores["hold"]), ("blend", mixed)]:
                    layout_rows.append({"context": context, "arm": arm, "subset": label, "model": name,
                                        "rows": int(mask.sum()), "human_rows": int((y[mask]==0).sum()),
                                        "ai_rows": int((y[mask]==1).sum()), "auroc": roc_auc_score(y[mask], p[mask])})
    # Verify task adaptation changes only initialization of the encoder, retaining
    # the same fresh classification head as the control.
    original = load_model(seed=CONFIG["seed"])
    head_before = {k: p.detach().clone() for k, p in original.classifier.named_parameters()}
    del original
    adapted = load_model(adapted=CACHE / "fit/mlm/epoch_1", seed=CONFIG["seed"])
    assert all(torch.equal(head_before[k], p) for k, p in adapted.classifier.named_parameters())
    del adapted
    rows = chosen["all_arms"]
    rebuilt_choice = max(rows, key=lambda r: (r["inner_blend_auroc"], r["domain_neural_auroc"], r["inner_neural_auroc"]))
    assert rebuilt_choice == chosen["chosen"]
    train, val, _ = raw_context("fit")
    table = pd.read_csv(OUT / "val_predictions.csv")
    assert np.array_equal(table.id, val.id) and np.array_equal(table.label, val.label)
    baseline = reference("fit")
    for arm in ARMS:
        cfg = inner["arms"][arm]
        cache = np.load(CACHE / f"fit_{arm}_epoch_{cfg['epoch']}_scores.npz")
        np.testing.assert_allclose(table[f"{arm}_neural"], cache["hold"], atol=1e-14)
        p = logit_mixture(baseline, logit(np.clip(cache["hold"], 1e-12, 1-1e-12)), cfg["weight"])
        np.testing.assert_allclose(table[f"{arm}_blend"], p, atol=1e-14)
        for kind in ["neural", "blend"]:
            name = f"{arm}_{kind}"; row = next(r for r in evaluation["results"] if r["model"] == name)
            assert abs(roc_auc_score(val.label, table[name])-row["validation_auroc"]) < 1e-12
    gate = json.loads((OUT / "submission_gate.json").read_text())
    if gate["generated"]:
        path = ROOT / gate["generated"]; manifest = json.loads(path.with_suffix(".json").read_text())
        assert manifest["csv_sha256"] == sha(path)
        components = np.load(CACHE / "test_components.npz")
        p = logit_mixture(components["previous"], logit(np.clip(components["roberta"], 1e-12, 1-1e-12)), chosen["chosen"]["weight"])
        actual = pd.read_csv(path); test = pd.read_csv(ROOT / "Data/test.csv")
        assert len(actual) == 3000 and actual.id.is_unique and list(actual.columns) == ["id", "label"]
        assert np.array_equal(actual.id, test.id) and np.array_equal(components["ids"], test.id)
        np.testing.assert_allclose(actual.label, p, atol=1e-12)
    # Descriptive subgroup checks; no choices change after this point.
    lengths = train.text.str.split().str.len()
    bins = np.quantile(lengths, [.25, .5, .75])
    groups = np.digitize(val.text.str.split().str.len(), bins)
    subgroup_rows = []
    for group in range(4):
        mask = groups == group
        for name in table.columns[2:]:
            subgroup_rows.append({"training_length_quartile": group+1, "model": name, "rows": int(mask.sum()), "auroc": roc_auc_score(val.label.to_numpy()[mask], table[name].to_numpy()[mask])})
    pd.DataFrame(subgroup_rows).to_csv(OUT / "length_subgroups.csv", index=False)
    write(OUT / "ranking_diagnostics.json", ranking_details)
    pd.DataFrame(layout_rows).to_csv(OUT / "layout_subgroups.csv", index=False)
    # Prespecified supplementary comparison: all three have epoch-2 scores
    # (the control's second epoch is also needed as a pseudo-label teacher).
    # This isolates equal supervised exposure from primary blend checkpoint
    # selection, which retained control epoch 1 but pseudo/TAPT epoch 2.
    from experiments.neural_comparison import paired_ci
    matched_rows = []
    for context in ["inner", "domain", "fit"]:
        _, heldout, _ = raw_context(context)
        control = np.load(CACHE / f"{context}_supervised_epoch_2_scores.npz")
        assert np.array_equal(control["hold_ids"], heldout.id)
        y = heldout.label.to_numpy()
        for arm in ARMS:
            d = np.load(CACHE / f"{context}_{arm}_epoch_2_scores.npz")
            assert np.array_equal(d["hold_ids"], heldout.id)
            auc = roc_auc_score(y, d["hold"])
            interval = (0., 0.) if arm == "supervised" else paired_ci(y, d["hold"], control["hold"])
            matched_rows.append({"context": context, "arm": arm, "epoch": 2, "evaluation_rows": len(heldout),
                                 "neural_auroc": auc, "delta_supervised_epoch2": auc-roc_auc_score(y, control["hold"]),
                                 "paired_95ci_low": interval[0], "paired_95ci_high": interval[1],
                                 "note": "Supplementary equal-epoch comparison; does not change primary checkpoint, weight or arm selection."})
    pd.DataFrame(matched_rows).to_csv(OUT / "matched_epoch_comparison.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    compare = pd.DataFrame(evaluation["results"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    labels = ["Supervised", "Pseudo-labels", "Task pretraining"]
    x = np.arange(3)
    axes[0].plot(x, [inner["arms"][a]["inner_neural_auroc"] for a in ARMS], "o-", label="Neural model")
    axes[0].plot(x, [inner["arms"][a]["inner_blend_auroc"] for a in ARMS], "o-", label="With old blend")
    axes[0].set(title="Internal holdout (tuning)", ylabel="AUROC", xticks=x, xticklabels=labels); axes[0].legend()
    by_arm = {r["arm"]: r for r in rows}
    axes[1].plot(x, [by_arm[a]["domain_neural_auroc"] for a in ARMS], "o-", label="Neural model")
    axes[1].plot(x, [by_arm[a]["domain_blend_auroc"] for a in ARMS], "o-", label="With topic-only booster")
    axes[1].axhline(rows[0]["domain_boosting_auroc"], linestyle=":", color="#888888", label="Topic-only booster")
    axes[1].set(title="Held-out story/poetry topics (n=148)", ylabel="AUROC", xticks=x, xticklabels=labels); axes[1].legend()
    for kind, marker in [("neural", "o-"), ("blend", "o-")]:
        values = [compare.set_index("model").loc[f"{a}_{kind}", "validation_auroc"] for a in ARMS]
        axes[2].plot(x, values, marker, label=kind)
    axes[2].axhline(.999485, color="#888888", linestyle=":", label="Previous blend")
    axes[2].axhline(.999884, color="#aa5555", linestyle="--", label="Teammate recorded best")
    axes[2].set(title="Fixed validation (reused)", ylabel="AUROC", xticks=x, xticklabels=labels); axes[2].legend()
    for ax in axes:
        ax.tick_params(axis="x", rotation=15); ax.grid(axis="y", alpha=.2)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    fig.savefig(OUT / "comparison.png", dpi=160); plt.close(fig)
    write(OUT / "verification.json", {"passed": True, "source_data_split_hashes_verified": True,
                                      "integration_tests_passed": 5, "mps_smoke_tests_passed": True,
                                      "teacher_agreement_and_soft_targets_reconstructed": True,
                                      "supervised_optimizer_budgets_verified": True, "matching_fresh_classification_heads": True,
                                      "selection_and_validation_blends_reconstructed": True, "contexts": fold_details,
                                      "checkpoints": checks, "candidate_generated": bool(gate["generated"]),
                                      "audit_source_sha256": sha(Path(__file__)), "kaggle_uploaded": False, "git_pushed": False})
    print("RoBERTa experiment artifact audit passed.")


if __name__ == "__main__":
    main()
