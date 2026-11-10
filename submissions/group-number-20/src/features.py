"""Text normalization and stylometric features."""
import re
import numpy as np

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "})

def normalize(t):
    """Remove formatting signals that may not carry over to the test set."""
    return re.sub(r"\s+", " ", t.translate(_QUOTES)).strip()

def perturb_format(t):
    """Simulate a formatting shift on evaluation text: curly quotes and flattened line breaks."""
    t = re.sub(r"\s*\n\s*", " ", t)
    t = re.sub(r'"([^"]*)"', "“\\1”", t)
    return t.replace("'", "’").strip()

NAMES = ["n_chars", "n_words", "n_sents", "words_per_sent", "sd_sent_len", "word_len", "type_token",
         "hapax", "stop_rate", "comma", "semicolon", "colon", "paren", "quote", "digit", "upper",
         "newline", "bullet", "first_person", "contraction", "ends_punct"]
_STOP = set("the of and to a in that is was it for on with as be by at this are from or an".split())
_FP = {"i", "me", "my", "we", "our", "us"}

def stylometric(t):
    w = re.findall(r"[A-Za-z']+", t); lw = [x.lower() for x in w]; n = max(len(w), 1); c = max(len(t), 1)
    s = [x for x in re.split(r"[.!?]+\s", t) if x.strip()]; sl = [len(x.split()) for x in s] or [0]
    vals, cnt = np.unique(lw or [""], return_counts=True)
    return [len(t), len(w), len(s), np.mean(sl), np.std(sl), np.mean([len(x) for x in w] or [0]), len(vals) / n,
            (cnt == 1).sum() / n, sum(x in _STOP for x in lw) / n, t.count(",") / c, t.count(";") / c, t.count(":") / c,
            t.count("(") / c, (t.count('"') + t.count("“")) / c, sum(x.isdigit() for x in t) / c,
            sum(x.isupper() for x in t) / c, t.count("\n") / c, len(re.findall(r"(?m)^\s*(?:[-*>]|\d+\.)", t)) / c,
            sum(x in _FP for x in lw) / n, sum("'" in x for x in w) / n, float(t.strip()[-1:] in tuple('.!?"'))]

def stylometric_matrix(texts):
    return np.array([stylometric(t) for t in texts], dtype=float)
