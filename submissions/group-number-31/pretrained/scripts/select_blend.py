"""Choose blends A and C by the pre-registered rule in configs/selection/blends.yaml.

  python run_all.py --profile topic_cv                                   # topic-held-out CV of the candidates
  python scripts/select_blend.py --session outputs/sessions/<topic_cv session>

A: greedy forward selection (src/ensemble.greedy_weights) on the candidates' topic-held-out out-of-fold
predictions. C: blend B + the fine-tune with the best 5-fold stratified CV (experiments/results.csv).
Only train.csv labels are used. Writes results/blend_selection/{topic_cv.csv, selection.json} and prints the
make_submission.py commands; test predictions come from the original stratified runs (--runs-dir).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ensemble import greedy_weights, rank01  # noqa: E402
from src.utils import auroc  # noqa: E402

OUT = ROOT / "results" / "blend_selection"
RESEARCH_RUNS = ROOT.parents[3] / "AI-vs-Human-Text-Cornell-Tech-Competition-" / "outputs" / "runs"


def topic_oof(session: Path, names) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    rows, cols, y = [], {}, None
    for m in names:
        run = ROOT / "outputs" / "runs" / f"{session.name}__{m}"
        o = pd.read_csv(run / "oof.csv").sort_values("id")
        met = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        assert (o["fold"] >= 0).all(), f"{m}: topic CV left rows unscored"
        cols[m] = o["pred"].to_numpy()
        y = o["label"].to_numpy() if y is None else y
        rows.append({"model": m, "topic_cv_auroc": met["cv_auc"], "fold_mean": met["cv_auc_mean_of_folds"],
                     "fold_std": met["cv_auc_std"], "fold_aucs": " ".join(f"{a:.5f}" for a in met["fold_aucs"]),
                     "run": run.name})
    return pd.DataFrame(cols), y, pd.DataFrame(rows)


def blend_auc(O: pd.DataFrame, y, weights: dict) -> float:
    w = pd.Series(weights, dtype=float)
    return auroc(y, sum(w[m] / w.sum() * rank01(O[m]) for m in w.index))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True, help="the run_all session of configs/experiments/topic_cv.yaml")
    ap.add_argument("--runs-dir", default=str(RESEARCH_RUNS), help="the original stratified runs (test predictions)")
    args = ap.parse_args()
    sel = yaml.safe_load((ROOT / "configs" / "selection" / "blends.yaml").read_text(encoding="utf-8"))
    cands = sel["candidates"]
    O, y, table = topic_oof(Path(args.session), list(cands))

    # A: greedy on topic-held-out OOF, pick counts as weights
    frac = greedy_weights(O, y, n_iter=sel["blend_a"]["n_iter"])
    a = {m: int(round(f * sel["blend_a"]["n_iter"])) for m, f in sorted(frac.items(), key=lambda kv: -kv[1])}
    # C: B + the fine-tune with the best stratified CV among the candidate runs
    log = pd.read_csv(ROOT / "experiments" / "results.csv").drop_duplicates("run_name", keep="last").set_index("run_name")
    ft_cv = {m: float(log.loc[cands[m], "cv_auc"]) for m in sel["blend_c"]["add_best_cv_of"]}
    best_ft = max(ft_cv, key=ft_cv.get)
    c = {m: 1 for m in sel["blend_c"]["base"] + [best_ft]}
    b = {m: 1 for m in sel["blend_c"]["base"]}

    table["stratified_cv_auroc"] = [float(log.loc[cands[m], "cv_auc"]) for m in table["model"]]
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "topic_cv.csv", index=False, lineterminator="\n", float_format="%.6g")
    result = {"session": Path(args.session).name, "blend_a_weights": a, "blend_c_members": list(c),
              "fine_tune_stratified_cv": ft_cv, "blend_c_added": best_ft,
              "topic_oof_auroc": {"blend_a": blend_auc(O, y, a), "blend_b": blend_auc(O, y, b),
                                  "blend_c": blend_auc(O, y, c),
                                  "best_single": float(table["topic_cv_auroc"].max())}}
    (OUT / "selection.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    runs = Path(args.runs_dir)
    for name, w in (("final_A_topic_greedy", a), ("final_C_lm4_distil" if best_ft == "ft_distilroberta" else
                                                   f"final_C_lm4_{best_ft}", c)):
        print(f"python make_submission.py --run {' '.join(str(runs / cands[m]) for m in w)} "
              f"--weights {' '.join(str(v) for v in w.values())} --name {name}")


if __name__ == "__main__":
    main()
