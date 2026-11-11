"""Bring the Overleaf report up to date in one command: regenerate every table, inline them, compile, check.

  python reports/refresh.py              # run from the team folder, in the cs-aml env
  python reports/refresh.py --no-pdf     # skip the compile (no tectonic installed)

Steps, in order (each is an ordinary script you can also run alone):
  1. classical/report_tables.py, pretrained/scripts/report_tables.py, common/compare_tracks.py,
     reports/build_tables.py   -> the generated tables (from results files only; nothing typed by hand)
  2. reports/inline_tables.py  -> copies them into reports/overleaf/kaggle_report.tex
  3. checks on that file: pure ASCII (Overleaf's default pdfLaTeX), no \\input / \\includegraphics (the folder
     must be self-contained), and lists the [TODO]s left
  4. tectonic compile of a copy of reports/overleaf/ alone -> reports/kaggle_report_preview.pdf

Afterwards upload the files in reports/overleaf/ to Overleaf (replace kaggle_report.tex there).
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TEAM = Path(__file__).resolve().parent.parent
OVERLEAF = TEAM / "reports" / "overleaf"
REPORT = OVERLEAF / "kaggle_report.tex"
PREVIEW = TEAM / "reports" / "kaggle_report_preview.pdf"
STEPS = [  # (working dir, script)
    (TEAM / "classical", "report_tables.py"),
    (TEAM, "pretrained/scripts/report_tables.py"),
    (TEAM, "common/compare_tracks.py"),
    (TEAM, "reports/build_tables.py"),
    (TEAM, "reports/inline_tables.py"),
]
TECTONIC = [Path(sys.prefix).parent / "tex" / "Library" / "bin" / "tectonic.exe"]  # the team's `tex` conda env


def run(cwd: Path, script: str) -> None:
    print(f"+ {script}", flush=True)
    r = subprocess.run([sys.executable, script], cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0:
        sys.exit(f"{script} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")


def check() -> None:
    text = REPORT.read_text(encoding="utf-8")
    bad = sorted({ch for ch in text if ord(ch) > 127})
    if bad:
        sys.exit(f"non-ASCII characters in {REPORT.name} (pdfLaTeX may fail): {bad}")
    if re.search(r"\\(input|include|includegraphics|bibliography)\b", text):
        sys.exit(f"{REPORT.name} pulls in another file; the Overleaf folder must stay self-contained")
    others = [p.name for p in OVERLEAF.iterdir() if p.name != REPORT.name]
    if others:
        print(f"note: reports/overleaf/ also holds {others}; Overleaf will get them too")
    todos = re.findall(r"\[TODO[^\]]*\]", text)
    print(f"ok: ASCII, self-contained; {len(todos)} TODO(s) left: {sorted(set(todos))}")


def compile_pdf() -> None:
    exe = next((str(p) for p in TECTONIC if p.exists()), shutil.which("tectonic"))
    if not exe:
        print("tectonic not found: skipped the compile (Overleaf will compile it)")
        return
    with tempfile.TemporaryDirectory() as tmp:  # compile the folder alone, as Overleaf sees it
        shutil.copytree(OVERLEAF, tmp, dirs_exist_ok=True)
        r = subprocess.run([exe, REPORT.name], cwd=tmp, capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        log = r.stdout + r.stderr
        if r.returncode != 0:
            sys.exit("compile failed:\n" + log[-3000:])
        warnings = [ln for ln in log.splitlines() if re.search(r"Overfull|undefined|Citation", ln)]
        shutil.copyfile(Path(tmp) / REPORT.with_suffix(".pdf").name, PREVIEW)
    print(f"compiled -> {PREVIEW.relative_to(TEAM)}" + (f"; {len(warnings)} layout warning(s):" if warnings else ""))
    for w in warnings[:10]:
        print("  " + w)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-pdf", action="store_true", help="skip the tectonic compile")
    args = ap.parse_args()
    for cwd, script in STEPS:
        run(cwd, script)
    check()
    if not args.no_pdf:
        compile_pdf()
    print("upload the files in reports/overleaf/ to Overleaf")


if __name__ == "__main__":
    main()
