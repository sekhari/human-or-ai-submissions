"""Label-free check: do test passages show character-level attack traces that training lacks?

Uses only text from train (split by label), unlabeled, and test. Never uses test labels.
"""
import re, unicodedata, json
from collections import Counter
import numpy as np, pandas as pd

d = pd.read_csv("data/train.csv"); u = pd.read_csv("data/unlabeled.csv"); e = pd.read_csv("data/test.csv")
sets = {"train_human": d.text[d.label == 0], "train_ai": d.text[d.label == 1], "unlabeled": u.text, "test": e.text}
vocab = Counter(w for t in d.text for w in re.findall(r"[a-z]+", t.lower()))
common = {w for w, c in vocab.items() if c >= 5}

def script(ch):
    try: n = unicodedata.name(ch)
    except ValueError: return "?"
    return n.split()[0]

def feats(t):
    words = re.findall(r"\w+", t)
    mixed = sum(1 for w in words if any(c.isalpha() and c.isascii() for c in w) and any(script(c) in ("CYRILLIC", "GREEK") for c in w))
    alpha = [w.lower() for w in re.findall(r"[A-Za-z]+", t)]
    return {
        "zero_width_or_format": any(unicodedata.category(c) == "Cf" for c in t),
        "mixed_script_word": mixed > 0,
        "odd_space": bool(re.search(r"[  -   　]", t)),
        "case_swap_word": len(re.findall(r"\b[a-z]+[A-Z][a-zA-Z]*\b", t)) >= 3,
        "multi_space": "  " in t,
        "curly_quote": bool(re.search("[‘’“”]", t)),
        "oov_rate": np.mean([w not in common for w in alpha]) if alpha else 0.0,
        "article_rate": np.mean([w in ("the", "a", "an") for w in alpha]) if alpha else 0.0,
    }

rows = {}
for k, s in sets.items():
    f = pd.DataFrame([feats(t) for t in s])
    rows[k] = {c: round(float(f[c].mean()), 4) for c in f.columns}
    rows[k]["oov_p90"] = round(float(f.oov_rate.quantile(.9)), 4)
r = pd.DataFrame(rows); print(r.to_string())
# Examples of the rarest traces in test, truncated, for a human sanity check.
for name, pat in [("format chars", lambda t: any(unicodedata.category(c) == "Cf" for c in t)),
                  ("mixed script", lambda t: feats(t)["mixed_script_word"]),
                  ("case swap", lambda t: feats(t)["case_swap_word"])]:
    ex = e.text[e.text.map(pat)].head(2).tolist()
    print("\n==", name, len(e.text[e.text.map(pat)]))
    for x in ex: print(repr(x[:220]))
json.dump(rows, open("results/attack_traces.json", "w"), indent=1)
