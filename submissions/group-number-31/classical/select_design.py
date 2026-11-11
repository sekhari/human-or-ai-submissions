"""Choose one model's design by a pre-registered rule on topic-held-out inner CV (no external data).

  python select_design.py --model tfidf_lr --jobs 10                 # CPU: 10 worker processes
  python select_design.py --model transformer_scratch --devices cuda:0 cuda:1   # one worker per GPU
  python select_design.py --model tfidf_lr --refit-check             # winner refit on 10k vs the committed Kaggle file
  python select_design.py --model tfidf_lr --write-final G           # winner refit on 10k -> final_G_tfidf_lr.csv

The candidates and the rule are in configs/selection/<model>.yaml, committed before the first run. Every
candidate is scored on the 8,000 training rows only:
  topic       train.py's topic-held-out CV (k-means k=10 on those rows, GroupKFold(5) over the clusters) - selects;
  stratified  train.py's stratified 5-fold inner CV - reported only;
  val         fit on the 8,000 rows, scored on the fixed 2,000 validation rows - reported only.
Rule: highest mean topic AUROC; every candidate within one fold s.d. of the best (the best candidate's s.d.)
ties, and the tie goes to the simplest one (the axes' option order in the yaml).

Writes results/selection/<model>/: folds.csv (one row per fit; a re-run skips finished fits), candidates.csv,
selection.json (winner, ties, and whether the winner equals the params of the committed Kaggle file).
"""
from __future__ import annotations

import argparse
import io
import itertools
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
TEAM = HERE.parent
sys.path.insert(0, str(TEAM / "common"))
sys.path.insert(0, str(HERE))
from data import SEED, load_csv, train_val_split  # noqa: E402
from metrics import summary  # noqa: E402
from validate_submission import problems  # noqa: E402

import train as T  # noqa: E402

OUT = HERE / "results" / "selection"


def candidates(sel: dict) -> list[dict]:
    """Every combination of the axes' options; `simplicity` is the options' positions (smaller = simpler)."""
    axes = sel["axes"]
    out = []
    for combo in itertools.product(*(list(enumerate(a["options"])) for a in axes)):
        params = dict(sel["base"])
        for _, opt in combo:
            params.update(opt["params"])
        out.append({"name": "+".join(o["name"] for _, o in combo), "simplicity": [i for i, _ in combo],
                    "params": params})
    return out


def same_params(a: dict, b: dict) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ------------------------------------------------------------------ workers
def worker(model: str, device: str | None, splits: dict, tasks, results) -> None:
    cfg = T.load_config(model)
    train, val = train_val_split()
    X, y = train["text"].tolist(), train["label"].to_numpy()
    unl = load_csv("unlabeled")["text"].tolist()
    for task in iter(tasks.get, None):
        t0 = time.time()
        if task["scheme"] == "val":
            tr_x, tr_y, va_x, va_y, seed = X, y, val["text"].tolist(), val["label"].to_numpy(), SEED
        else:
            tr, va = splits[task["scheme"]][task["fold"]]
            tr_x, tr_y, va_x, va_y = [X[i] for i in tr], y[tr], [X[i] for i in va], y[va]
            seed = SEED + task["fold"]  # as train.inner_cv
        m = T.build(cfg, task["params"], seed, device).fit(tr_x, tr_y, unlabeled_texts=unl)
        auc = summary(va_y, m.predict_proba(va_x))["auroc"]
        results.put({"candidate": task["candidate"], "scheme": task["scheme"], "fold": task["fold"],
                     "auroc": auc, "n": len(va_y), "hardware": T.hardware(m), "time_s": round(time.time() - t0, 1)})


def run(model: str, sel: dict, cands: list[dict], devices: list[str | None]) -> pd.DataFrame:
    out = OUT / model
    out.mkdir(parents=True, exist_ok=True)
    folds_file = out / "folds.csv"
    done = pd.read_csv(folds_file) if folds_file.exists() else pd.DataFrame(columns=["candidate", "scheme", "fold"])
    done_keys = set(zip(done["candidate"], done["scheme"], done["fold"].astype(int)))

    train, _ = train_val_split()
    X, y = train["text"].tolist(), train["label"].to_numpy()
    cfg = T.load_config(model)
    schemes = [sel["select_on"]] + list(sel.get("also_report", []))
    splits = {s: T.cv_splits(s, X, y, cfg["cv_folds"], SEED) for s in schemes if s != "val"}
    T.write_topic_folds(X, y, cfg["cv_folds"], SEED, HERE / "results" / "topic_cv")

    todo = [{"candidate": c["name"], "params": c["params"], "scheme": s, "fold": f}
            for s in schemes for c in cands for f in (range(cfg["cv_folds"]) if s != "val" else [-1])
            if (c["name"], s, f) not in done_keys]
    print(f"{model}: {len(cands)} candidates, {len(todo)} fits to run on {len(devices)} worker(s)", flush=True)
    if todo:
        ctx = mp.get_context("spawn")
        tasks, results = ctx.Queue(), ctx.Queue()
        for t in todo:
            tasks.put(t)
        for _ in devices:
            tasks.put(None)
        procs = [ctx.Process(target=worker, args=(model, d, splits, tasks, results)) for d in devices]
        for p in procs:
            p.start()
        write_header = not folds_file.exists()
        with open(folds_file, "a", encoding="utf-8", newline="") as fh:
            for i in range(len(todo)):
                r = results.get()
                pd.DataFrame([r]).to_csv(fh, header=write_header, index=False, lineterminator="\n",
                                         float_format="%.6f")
                write_header = False
                fh.flush()
                print(f"  [{i + 1}/{len(todo)}] {r['candidate']} {r['scheme']} fold {r['fold']}: "
                      f"AUROC {r['auroc']:.4f} ({r['time_s']:.0f}s)", flush=True)
        for p in procs:
            p.join()
    return pd.read_csv(folds_file)


def decide(model: str, sel: dict, cands: list[dict], folds: pd.DataFrame) -> dict:
    rows = []
    for i, c in enumerate(cands):
        row = {"candidate": c["name"], "simplicity": "-".join(map(str, c["simplicity"])), "order": i}
        for s in [sel["select_on"]] + list(sel.get("also_report", [])):
            a = folds[(folds["candidate"] == c["name"]) & (folds["scheme"] == s)].sort_values("fold")["auroc"]
            if s == "val":
                row["val_auroc"] = float(a.iloc[0])
            else:
                row[f"{s}_mean"], row[f"{s}_std"] = float(np.mean(a)), float(np.std(a))
                row[f"{s}_folds"] = " ".join(f"{v:.5f}" for v in a)
        row["params"] = json.dumps(c["params"], sort_keys=True)
        rows.append(row)
    df = pd.DataFrame(rows)
    key = sel["select_on"]
    best = df.loc[df[f"{key}_mean"].idxmax()]
    df["tied"] = df[f"{key}_mean"] >= best[f"{key}_mean"] - best[f"{key}_std"]
    winner = df[df["tied"]].sort_values("order").iloc[0]  # candidates() yields the simplest combination first
    df["selected"] = df["candidate"] == winner["candidate"]
    df.drop(columns="order").to_csv(OUT / model / "candidates.csv", index=False, lineterminator="\n",
                                    float_format="%.6f")

    current = json.loads((HERE / "results" / model / "metrics.json").read_text(encoding="utf-8"))["params"]
    win_params = json.loads(winner["params"])
    result = {
        "model": model, "rule": f"max mean {key} inner-CV AUROC; ties within 1 fold s.d. of the best -> simplest",
        "best": best["candidate"], "best_mean": float(best[f"{key}_mean"]), "best_std": float(best[f"{key}_std"]),
        "tied": df.loc[df["tied"], "candidate"].tolist(), "winner": winner["candidate"], "winner_params": win_params,
        "current_params": current, "winner_equals_current": same_params(win_params, current),
        "n_candidates": len(df),
    }
    (OUT / model / "selection.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{model}: best {result['best']} ({result['best_mean']:.5f} +/- {result['best_std']:.5f}); "
          f"{len(result['tied'])} tied; winner {result['winner']}; equals the committed file's params: "
          f"{result['winner_equals_current']}")
    return result


def refit(model: str, letter: str | None, device: str | None) -> None:
    """Refit the winner on all 10,000 rows. Without a letter, compare with the committed file (check only)."""
    sel = json.loads((OUT / model / "selection.json").read_text(encoding="utf-8"))
    cfg = T.load_config(model)
    train, test = load_csv("train"), load_csv("test")
    unl = load_csv("unlabeled")["text"].tolist()
    m = T.build(cfg, sel["winner_params"], SEED, device).fit(train["text"].tolist(), train["label"].to_numpy(),
                                                            unlabeled_texts=unl)
    sub = pd.DataFrame({"id": test["id"], "label": np.clip(m.predict_proba(test["text"].tolist()), 0, 1)})
    bad = problems(sub, load_csv("sample_submission"))
    if bad:
        raise SystemExit("invalid submission: " + "; ".join(bad))
    if letter:
        path = T.SUBS / f"final_{letter}_{model}.csv"
        if path.exists():
            raise SystemExit(f"{path.name} exists; pick another letter")
        sub.to_csv(path, index=False, lineterminator="\n", float_format="%.6f")
        print(f"wrote {path.relative_to(TEAM)} (winner {sel['winner']}; not uploaded)")
        return
    committed = sorted(T.SUBS.glob(f"final_{T.LETTERS[model]}_{model}.csv"))[0]
    old = pd.read_csv(committed)
    new = pd.read_csv(io.StringIO(sub.to_csv(index=False, float_format="%.6f")))  # the file's rounding
    diff = float(np.abs(old["label"].to_numpy() - new["label"].to_numpy()).max())
    res = {"winner": sel["winner"], "committed_file": committed.name, "ids_identical": bool((old["id"] == new["id"]).all()),
           "max_abs_diff": diff, "identical": bool(diff == 0 and (old["id"] == new["id"]).all()),
           "hardware": T.hardware(m)}
    (OUT / model / "refit_check.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"{model}: refit of {sel['winner']} vs {committed.name}: max |diff| {diff:.6g} -> "
          f"{'identical' if res['identical'] else 'DIFFERENT'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--jobs", type=int, default=1, help="CPU worker processes")
    ap.add_argument("--devices", nargs="*", default=None, help="torch models: one worker per device")
    ap.add_argument("--refit-check", action="store_true", help="refit the winner on 10k, compare with the committed file")
    ap.add_argument("--write-final", metavar="LETTER", default=None, help="refit the winner, write final_<LETTER>_<model>.csv")
    args = ap.parse_args()
    if args.refit_check or args.write_final:
        return refit(args.model, args.write_final, (args.devices or [None])[0])
    sel = yaml.safe_load((HERE / "configs" / "selection" / f"{args.model}.yaml").read_text(encoding="utf-8"))
    cands = candidates(sel)
    devices = args.devices or [None] * args.jobs
    folds = run(args.model, sel, cands, devices)
    decide(args.model, sel, cands, folds)


if __name__ == "__main__":
    main()
