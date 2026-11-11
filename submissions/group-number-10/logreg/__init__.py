"""Logistic regression track (owner: Joshua).

    features.py   text cleaning + the four feature blocks (words, chars, style, lm)
    model.py      BlockLogReg (one logreg on glued blocks) and StackedLogReg (one logreg per block + meta logreg)
    ablation.py   which blocks help?            python -m logreg.ablation
    stack.py      stacking + test-like check     python -m logreg.stack

Everything this track writes goes under results/logreg/.
"""

from src.data import RESULTS_DIR as _RESULTS_ROOT

RESULTS_DIR = _RESULTS_ROOT / "logreg"
SEED = 42
