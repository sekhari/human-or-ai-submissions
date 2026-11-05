"""label-preserving text edits. used to stress test models and as training augmentation.

these are written from scratch from general knowledge of how detectors get evaded. no external
labeled data is involved, only edits applied to the passages we already have labels for.
"""
import random
import re

HOMOGLYPHS = {"a": "а", "e": "е", "o": "о", "c": "с", "p": "р", "i": "і", "x": "х", "y": "у"}
ZERO_WIDTH = "​"
SPELLING = {"color": "colour", "center": "centre", "organize": "organise", "realize": "realise", "favorite": "favourite",
            "behavior": "behaviour", "analyze": "analyse", "theater": "theatre", "defense": "defence", "labor": "labour"}
TYPOS = {"the": "teh", "and": "adn", "receive": "recieve", "until": "untill", "their": "thier", "which": "wich",
         "because": "becuase", "people": "poeple", "definitely": "definately", "patience": "patients"}
SYNONYMS = {"important": "crucial", "big": "large", "help": "assist", "use": "employ", "show": "demonstrate", "many": "numerous",
            "start": "begin", "buy": "purchase", "end": "finish", "good": "fine", "hard": "difficult", "often": "frequently",
            "however": "nevertheless", "also": "additionally", "get": "obtain", "need": "require", "small": "little",
            "shows": "demonstrates", "uses": "employs", "helps": "assists"}


def _swap_chars(t, table, rate, r):
    return "".join(table[c] if c in table and r.random() < rate else c for c in t)


def _swap_words(t, table, rate, r):
    def sub(m):
        w = m.group(0)
        n = table.get(w.lower())
        if n is None or r.random() > rate:
            return w
        return n.capitalize() if w[0].isupper() else n
    return re.sub(r"[A-Za-z]+", sub, t)


def flatten_ws(t, r):
    return re.sub(r"\s+", " ", t).strip()


def drop_blank_lines(t, r):
    return re.sub(r"\n\s*\n", "\n", t)


def add_blank_lines(t, r):
    return t.replace(". ", ".\n\n") if r.random() < .5 else t.replace("\n", "\n\n")


def extra_spaces(t, r):
    return re.sub(r" ", lambda m: "  " if r.random() < .15 else " ", t)


def homoglyph(t, r):
    return _swap_chars(t, HOMOGLYPHS, .08, r)


def zero_width(t, r):
    return re.sub(r" ", lambda m: ZERO_WIDTH + " " if r.random() < .3 else " ", t)


def drop_articles(t, r):
    return re.sub(r"\b(the|a|an|The|A|An)\s", lambda m: "" if r.random() < .7 else m.group(0), t)


def strip_digits(t, r):
    return re.sub(r"\d", "", t)


def typos(t, r):
    return _swap_words(t, TYPOS, .5, r)


def char_noise(t, r):
    def sub(m):
        w = m.group(0)
        if len(w) < 5 or r.random() > .05:
            return w
        i = r.randrange(1, len(w) - 2)
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    return re.sub(r"[A-Za-z]+", sub, t)


def spelling(t, r):
    return _swap_words(t, SPELLING, 1., r)


def synonyms(t, r):
    return _swap_words(t, SYNONYMS, .6, r)


def case_noise(t, r):
    return re.sub(r"[A-Za-z]+", lambda m: m.group(0).swapcase() if r.random() < .05 else m.group(0), t)


ATTACKS = {f.__name__: f for f in (flatten_ws, drop_blank_lines, add_blank_lines, extra_spaces, homoglyph, zero_width,
                                   drop_articles, strip_digits, typos, char_noise, spelling, synonyms, case_noise)}


def apply(name, texts, seed=0):
    r = random.Random(seed)
    return [ATTACKS[name](t, r) for t in texts]


def augment_random(texts, seed=0, k=2):
    """apply k random edits to each text."""
    r = random.Random(seed)
    names = sorted(ATTACKS)
    out = []
    for t in texts:
        for n in r.sample(names, k):
            t = ATTACKS[n](t, r)
        out.append(t)
    return out
