"""Text distortion (Stamatatos 2017, "Authorship attribution using text distortion").

Every word that is not among the k most frequent words of the training texts is masked: "*" if it starts
lower-case, "#" if it starts upper-case. Function words, punctuation, digits, spacing and line breaks are kept.
What is left is the passage's grammatical and layout skeleton without its topic, so a model trained on it
leans less on vocabulary that does not carry over to unseen domains (test has far more verse than train).
The word list is learned from the training texts passed to fit(), never from validation or test text.
"""
from __future__ import annotations

import re
from collections import Counter

WORD = re.compile(r"[A-Za-z]+")


class Distorter:
    def __init__(self, k: int = 300):
        self.k = k
        self.keep: set[str] = set()

    def fit(self, texts) -> "Distorter":
        cnt = Counter(w.lower() for t in texts for w in WORD.findall(t))
        self.keep = {w for w, _ in cnt.most_common(self.k)}
        return self

    def _mask(self, m: re.Match) -> str:
        w = m.group(0)
        return w if w.lower() in self.keep else ("#" if w[0].isupper() else "*")

    def transform(self, texts) -> list[str]:
        return [WORD.sub(self._mask, t) for t in texts]
