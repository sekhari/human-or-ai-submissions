"""Label-free stylometry additions for Yuchen's CatBoost experiments."""
from collections import Counter
import math
import re

import numpy as np
import pandas as pd

from style_features import style_features

WORDS = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
FUNCTION_GROUPS = {
    "first_person": "i me my mine we us our ours",
    "second_person": "you your yours",
    "third_person": "he him his she her hers they them their theirs",
    "articles": "a an the",
    "prepositions": "in on at by for from with into through during before after of to",
    "conjunctions": "and but or nor yet so because although while whereas",
    "modals": "can could may might must shall should will would",
    "negation": "no not never neither nobody nothing nowhere",
    "hedges": "perhaps possibly likely generally usually often sometimes relatively",
    "transitions": "however therefore moreover furthermore consequently additionally thus hence",
}


def extra_one(text):
    tokens = WORDS.findall(text)
    lower = [t.lower() for t in tokens]
    n = max(len(lower), 1)
    counts = Counter(lower)
    chars = Counter(text)
    sentences = [s for s in re.split(r"[.!?]+\s*", text) if s.strip()]
    lines = [s.strip() for s in text.splitlines() if s.strip()]
    f = {}
    f["lex_entropy"] = -sum((c/n)*math.log(c/n) for c in counts.values())
    f["char_entropy"] = -sum((c/max(len(text),1))*math.log(c/max(len(text),1)) for c in chars.values())
    f["top_word_share"] = max(counts.values(), default=0)/n
    f["top5_word_share"] = sum(c for _, c in counts.most_common(5))/n
    f["short_word_share"] = sum(len(w) <= 3 for w in lower)/n
    f["long_word_share"] = sum(len(w) >= 8 for w in lower)/n
    f["contraction_share"] = sum("'" in w for w in lower)/n
    f["allcaps_word_share"] = sum(w.isupper() and len(w)>1 for w in tokens)/n
    f["titlecase_word_share"] = sum(w.istitle() for w in tokens)/n
    f["repeated_word_adjacency"] = sum(a==b for a,b in zip(lower,lower[1:]))/max(n-1,1)
    for size in [2,3,4]:
        grams = [tuple(lower[i:i+size]) for i in range(max(len(lower)-size+1,0))]
        f[f"repeat_{size}gram_share"] = 1-len(set(grams))/len(grams) if grams else 0.0
    for name, vocab in FUNCTION_GROUPS.items():
        f[f"func_{name}"] = sum(counts[w] for w in vocab.split())/n
    for name, pattern in {
        "ellipsis": r"\.{3}|…", "curly_quote": r"[“”‘’]", "url": r"https?://|www\.",
        "repeated_punct": r"[!?]{2,}", "colon_line_end": r":\s*$",
        "bracket_reference": r"\[\d+\]", "numbered_line": r"^\s*\d+[.)]\s",
    }.items():
        f[f"pattern_{name}"] = len(re.findall(pattern,text,re.MULTILINE))/n
    f["sentence_initial_lower_share"] = sum(s.strip()[:1].islower() for s in sentences)/max(len(sentences),1)
    f["line_initial_upper_share"] = sum(s[:1].isupper() for s in lines)/max(len(lines),1)
    f["line_end_punct_share"] = sum(s[-1:] in '.!?,;:' for s in lines)/max(len(lines),1)
    f["unique_line_share"] = len(set(lines))/max(len(lines),1)
    f["unique_sentence_share"] = len(set(sentences))/max(len(sentences),1)
    f["parenthesis_imbalance"] = abs(text.count('(')-text.count(')'))/n
    f["quote_odd"] = float(text.count('"')%2)
    return f


def features(texts):
    """Return existing 53 features and an extended, deterministic feature table."""
    texts = list(texts)
    base = style_features(texts).astype(float)
    extended = pd.concat([base, pd.DataFrame([extra_one(t) for t in texts])], axis=1)
    if not np.isfinite(extended.to_numpy()).all():
        raise ValueError('Nonfinite features')
    return base, extended
