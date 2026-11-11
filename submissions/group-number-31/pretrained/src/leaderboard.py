"""Regenerates the 'Results' table in AGENTS.md from experiments/results.csv.

The table lives between two marker comments in AGENTS.md so that it can be
rewritten automatically without touching the rest of the file. Writes are
guarded by a file lock because run_all.py runs several jobs in parallel."""
from __future__ import annotations

import pandas as pd

from .config import ROOT
from .utils import file_lock

START, END = "<!-- LEADERBOARD:START -->", "<!-- LEADERBOARD:END -->"


def _fmt(x, nd=4):
    v = pd.to_numeric(x, errors="coerce")
    return "" if pd.isna(v) else f"{v:.{nd}f}"


def render_table(results: pd.DataFrame, top_k: int = 12) -> str:
    if results.empty:
        return "_No runs recorded yet._"
    df = results.copy()
    for c in ["config", "tpr_at_1fpr", "device", "train_time_s", "kaggle_public", "notes", "semi_supervised"]:
        if c not in df:
            df[c] = ""
    df["cv_auc"] = pd.to_numeric(df["cv_auc"], errors="coerce")
    df["kaggle_public"] = pd.to_numeric(df["kaggle_public"], errors="coerce")
    df["config"] = df["config"].where(df["config"].notna() & (df["config"] != ""), df["model"])
    lines = []

    k = df.dropna(subset=["kaggle_public"]).sort_values("kaggle_public", ascending=False)
    if len(k):
        b = k.iloc[0]
        lines.append(f"**Best Kaggle public score:** {b.kaggle_public:.4f} — `{b.run_name}` ({b.config})\n")
    else:
        lines.append("**Best Kaggle public score:** none recorded yet\n")

    lines.append("| rank | run | config | CV scheme | CV AUROC | TPR@1%FPR | Kaggle public | train_frac | "
                 "semi-sup | device | time (s) | author | notes |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    top = df.sort_values("cv_auc", ascending=False).head(top_k)
    for i, (_, r) in enumerate(top.iterrows(), 1):
        std = pd.to_numeric(r.get("cv_auc_std"), errors="coerce")
        cv = f"{r.cv_auc:.4f}" + ("" if pd.isna(std) or std == 0 else f" ± {std:.4f}")
        cell = lambda v: "" if pd.isna(v) else str(v)  # noqa: E731
        lines.append(f"| {i} | `{r.run_name}` | {r.config} | {cell(r.cv_scheme)} | {cv} | {_fmt(r.tpr_at_1fpr, 3)} | "
                     f"{_fmt(r.kaggle_public)} | {cell(r.train_frac)} | {cell(r.semi_supervised)} | "
                     f"{cell(r.device)} | {_fmt(r.train_time_s, 0)} | {cell(r.author)} | {cell(r.notes)} |")
    lines.append(f"\n_Top {top_k} of {len(df)} logged runs by CV AUROC; full log in `experiments/results.csv`. "
                 "CV scores from different schemes are not directly comparable (`cluster` CV holds out whole "
                 "topics and is harder than `stratified`)._")
    return "\n".join(lines)


def update_agents_md(results_file) -> None:
    path = ROOT / "AGENTS.md"
    if not path.exists() or not results_file.exists():
        return
    with file_lock(path):
        text = path.read_text(encoding="utf-8")
        if START not in text or END not in text:
            return
        table = render_table(pd.read_csv(results_file))
        head, rest = text.split(START, 1)
        _, tail = rest.split(END, 1)
        path.write_text(f"{head}{START}\n{table}\n{END}{tail}", encoding="utf-8")
