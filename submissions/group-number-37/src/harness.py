"""shared setup for experiments: data, domains, split, and result saving."""
from pathlib import Path

import numpy as np
import pandas as pd

from .data import OUT, fixed_split, load_all
from .domains import infer_domains

REPORTS = Path(__file__).resolve().parents[1] / "reports"


class Ctx:
    def __init__(self):
        self.tr, self.un, self.te = load_all()
        self.texts = np.array(pd.concat([self.tr.text, self.un.text, self.te.text], ignore_index=True), dtype=object)
        self.y = self.tr.label.values
        self.n = len(self.tr)
        self.un_i = np.arange(self.n, self.n + len(self.un))
        self.te_i = np.arange(self.n + len(self.un), len(self.texts))
        self.dom, self.tops = infer_domains(self.texts)
        self.big = int(np.bincount(self.dom[self.te_i]).argmax())
        self.fit_i, self.val_i = fixed_split(self.tr)

    def surprisal(self, tag="gpt2"):
        return np.load(OUT / f"surprisal_{tag}.npy")

    def domain_z(self, s):
        """standardize features inside each inferred domain over all rows. no labels used."""
        z = np.array(s, dtype=np.float64)
        for d in np.unique(self.dom):
            m = self.dom == d
            z[m] = (s[m] - s[m].mean(0)) / (s[m].std(0) + 1e-6)
        return z


def upsert(name, row, key="model"):
    """write or replace one row in a report csv, so experiments can be rerun piece by piece."""
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / name
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(df):
        df = df[df[key] != row[key]]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.round(4).to_csv(path, index=False)
    return df
