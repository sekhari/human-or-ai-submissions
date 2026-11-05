"""Create the fixed stratified 2,000-row validation set, keeping near-duplicate groups intact."""
import argparse
import numpy as np, pandas as pd
from common import SEED, VAL_IDS
from neardup import groups

ap = argparse.ArgumentParser(); ap.add_argument("--train", default="data/train.csv")
ap.add_argument("--threshold", type=float, default=0.7); a = ap.parse_args()
df = pd.read_csv(a.train)
df["g"], _ = groups(df["text"].tolist(), a.threshold)
rng = np.random.default_rng(SEED)
gs = df.groupby("g").agg(n=("id", "size"), y=("label", "nunique"), label=("label", "first")).reset_index()
print("groups with >1 row:", (gs.n > 1).sum(), "rows in them:", gs.n[gs.n > 1].sum(), "mixed-label:", (gs.y > 1).sum())
gs = gs[gs.y == 1].iloc[rng.permutation((gs.y == 1).sum())]   # mixed-label groups stay in training
chosen, need = [], {0: 1000, 1: 1000}
for g, n, lab in zip(gs.g, gs.n, gs.label):
    if n <= need[lab]:
        chosen.append(g); need[lab] -= n
assert need == {0: 0, 1: 0}, need
val = df[df.g.isin(chosen)]
VAL_IDS.write_text("\n".join(map(str, sorted(val.id))) + "\n")
print("val", len(val), val.label.value_counts().to_dict(), "groups crossing split:",
      len(set(val.g) & set(df[~df.g.isin(chosen)].g)))
