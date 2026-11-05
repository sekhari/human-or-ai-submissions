"""undo evasion tricks before modeling so edits cannot flip the token stream."""
import re
import unicodedata

CONFUSABLE = str.maketrans({"а": "a", "е": "e", "о": "o", "с": "c", "р": "p", "і": "i", "х": "x", "у": "y", "ѕ": "s",
                            "ԁ": "d", "ɡ": "g", "һ": "h", "ј": "j", "ո": "n", "ν": "v", "ο": "o", "ı": "i",
                            "’": "'", "‘": "'", "“": '"', "”": '"', "—": "-", "–": "-", "…": "...", "\xa0": " "})
ZERO = re.compile("[​-‏⁠﻿­]")
LATIN_MIX = re.compile(r"\S+")


def _fix_token(tok):
    has_latin = any("a" <= c.lower() <= "z" for c in tok)
    return tok.translate(CONFUSABLE) if has_latin else tok


def clean(text):
    """nfkc, strip zero width, fix mixed script tokens, flatten all whitespace."""
    t = ZERO.sub("", unicodedata.normalize("NFKC", text))
    t = LATIN_MIX.sub(lambda m: _fix_token(m.group(0)), t)
    t = t.replace("<br />", " ").replace("<br/>", " ")
    return re.sub(r"\s+", " ", t).strip()
