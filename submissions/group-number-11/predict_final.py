"""Stable CLI for the frozen final prediction pipeline."""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    source = Path(__file__).resolve().parent / "src"
    sys.path.insert(0, str(source))
    runpy.run_path(str(source / "predict_final.py"), run_name="__main__")
