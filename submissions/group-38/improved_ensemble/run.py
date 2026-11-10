"""Reproduce Group 38's improved ensemble and Kaggle prediction CSV."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Competition ZIP downloaded from Kaggle")
    parser.add_argument("--output", default="results", help="Local output directory")
    args = parser.parse_args()
    data = Path(args.data).expanduser().resolve()
    if not data.is_file():
        parser.error(f"Data ZIP not found: {data}")
    output = Path(args.output).expanduser().resolve()
    env = os.environ.copy()
    env["COMPETITION_ZIP"] = str(data)
    env["MODEL_OUTPUT_DIR"] = str(output)
    here = Path(__file__).resolve().parent
    for script in ["lexical.py", "style.py", "style_mlp.py", "finalize.py"]:
        subprocess.run([sys.executable, str(here / script)], env=env, check=True)
    print(f"Kaggle CSV: {output / 'deliverables' / 'submission_improved.csv'}")


if __name__ == "__main__":
    main()
