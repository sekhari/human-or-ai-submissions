"""Text distortion: keep frequent words and punctuation, mask other words by length.

Masking content words removes most topic information, so the model must rely on
function words, word lengths and punctuation patterns that may transfer to new domains.
"""
import re
from collections import Counter
from features import normalize

_TOK = re.compile(r"[A-Za-z]+|\d+|[^\w\s]|\w+")

def top_words(texts, k):
    df = Counter()
    for t in texts:
        df.update(set(w.lower() for w in re.findall(r"[A-Za-z]+", t)))
    ranked = sorted(df.items(), key=lambda x: (-x[1], x[0]))  # break ties alphabetically for reproducibility
    return {w for w, _ in (ranked if k is None else ranked[:k])}

def distort(t, keep):
    out = []
    for tok in _TOK.findall(normalize(t)):
        lw = tok.lower()
        if lw in keep: out.append(lw)
        elif tok.isdigit(): out.append("#")
        elif tok[0].isalpha() or tok[0].isalnum(): out.append("*" * min(len(tok), 12))
        else: out.append(tok)
    return " ".join(out)
