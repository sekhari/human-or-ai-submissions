"""reuse cached surprisal rows when the text matches, compute anything missing."""
import numpy as np
import pandas as pd

from .data import OUT, load_all
from .surprisal import NAMES, surprisal_matrix


def get_surprisal(texts, tag="gpt2"):
    texts = list(texts)
    out = np.zeros((len(texts), len(NAMES)), dtype=np.float32)
    missing = list(range(len(texts)))
    path = OUT / f"surprisal_{tag}.npy"
    if path.exists():
        tr, un, te = load_all()
        lookup = {t: i for i, t in enumerate(pd.concat([tr.text, un.text, te.text], ignore_index=True))}
        cache = np.load(path)
        missing = []
        for i, t in enumerate(texts):
            if t in lookup:
                out[i] = cache[lookup[t]]
            else:
                missing.append(i)
    if missing:
        out[missing] = surprisal_matrix([texts[i] for i in missing], tag)
    return out
