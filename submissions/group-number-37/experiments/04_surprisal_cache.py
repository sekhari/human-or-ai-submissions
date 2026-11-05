"""cache gpt2 surprisal features for every passage. labels are not used here."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from src.data import OUT, load_all
from src.surprisal import surprisal_matrix

model = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
OUT.mkdir(exist_ok=True)
tr, un, te = load_all()
texts = pd.concat([tr.text, un.text, te.text], ignore_index=True).tolist()
tag = model.replace("/", "_")
np.save(OUT / f"surprisal_{tag}.npy", surprisal_matrix(texts, model, log_every=50, checkpoint=OUT / f"surprisal_{tag}_ckpt.npz"))
