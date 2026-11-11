"""Synthetic adversarial edits used to stress-test models (Q5).

The competition says the test set contains "adversarially modified AI-written passages" but
gives no details, so we simulate common, published evasion edits (RAID-style): homoglyph
swaps, zero-width characters, typos, random upper-casing, article deletion and extra
whitespace. Each perturbed passage receives ONE attack chosen at random (seeded).
"""

from __future__ import annotations

import random
import re

CYR = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "y": "у", "x": "х", "i": "і"}
ATTACKS = ["homoglyph", "zero_width", "typo", "upper_lower", "article_deletion", "whitespace"]


def _words_apply(text, rate, fn, rng):
    toks = re.split(r"(\s+)", text)
    return "".join(fn(t) if t.strip() and rng.random() < rate else t for t in toks)


def attack(text: str, kind: str, rng: random.Random, rate: float = 0.15) -> str:
    if kind == "homoglyph":
        return "".join(CYR[c] if c in CYR and rng.random() < rate else c for c in text)
    if kind == "zero_width":
        return _words_apply(text, rate, lambda w: w[: len(w) // 2] + "​" + w[len(w) // 2:], rng)
    if kind == "typo":
        def typo(w):
            if len(w) < 4:
                return w
            i = rng.randrange(1, len(w) - 2)
            return w[:i] + w[i + 1] + w[i] + w[i + 2:]
        return _words_apply(text, rate, typo, rng)
    if kind == "upper_lower":
        return _words_apply(text, rate, lambda w: "".join(c.upper() if rng.random() < 0.5 else c for c in w), rng)
    if kind == "article_deletion":
        return re.sub(r"\b(the|a|an|The|A|An) ", lambda m: "" if rng.random() < 0.7 else m.group(0), text)
    if kind == "whitespace":
        return _words_apply(text, rate, lambda w: w + " ", rng)
    raise ValueError(kind)


def perturb_texts(texts, seed: int = 0, kinds=ATTACKS, rate: float = 0.15):
    """Return (perturbed_texts, attack_names): one random attack per text."""
    rng = random.Random(seed)
    out, names = [], []
    for t in texts:
        k = rng.choice(list(kinds))
        out.append(attack(t, k, rng, rate))
        names.append(k)
    return out, names
