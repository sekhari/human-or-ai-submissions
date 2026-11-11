"""Genre-robust features added in Yuchen's v2 (shift-aware) experiments.

Motivation: test is ~36% poetry/fiction (topic cluster 4) versus 1.5% of train.
Blank lines identify most AI texts, so the leaderboard AUC is decided by how
well the model ranks *non-blank-line* texts in an unseen genre. These features
aim at signals that do not depend on topic words:

- word rarity from a general-English unigram model (`wordfreq` Zipf scores):
  LLM text prefers frequent words and has lower rarity variance;
- compressibility (zlib / lzma ratios): a cheap predictability proxy;
- poem-structure cues: end rhyme rate, line-initial capitals, enjambment,
  archaic forms, and layout-free punctuation habits.

All features are label-free and computed per text.
"""
from __future__ import annotations

import lzma
import re
import zlib
from collections import Counter

import numpy as np
import pandas as pd
from wordfreq import zipf_frequency

WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
ARCHAIC = set("thee thou thy thine hath doth saith ye o'er e'er ere wherefore whence thence "
              "hither thither art shalt wilt canst dost hast oft nay yea tis 'tis".split())
LLM_WORDS = set("realm tapestry whisper whispers whispering embrace embraces symphony vibrant "
                "testament delve delves navigate journey unwavering profound intricate "
                "beacon resilience serene ethereal tranquil cascade cascading dance dances "
                "eternal boundless solace hues amidst weave woven".split())

_zipf_cache: dict[str, float] = {}


def _zipf(w: str) -> float:
    v = _zipf_cache.get(w)
    if v is None:
        v = zipf_frequency(w, "en")
        _zipf_cache[w] = v
    return v


def _rhyme_key(word: str) -> str:
    w = word.lower()
    m = re.search(r"[aeiouy]+[^aeiouy]*$", w)
    return m.group(0) if m else w[-2:]


def v2_one(text: str) -> dict[str, float]:
    tokens = WORD.findall(text)
    lower = [t.lower() for t in tokens]
    n = max(len(lower), 1)
    z = np.array([_zipf(w) for w in lower]) if lower else np.zeros(1)
    content = z[z < 5.5] if (z < 5.5).any() else np.zeros(1)
    f: dict[str, float] = {
        "zipf_mean": z.mean(),
        "zipf_std": z.std(),
        "zipf_p10": np.percentile(z, 10),
        "zipf_p25": np.percentile(z, 25),
        "zipf_content_mean": content.mean(),
        "zipf_content_std": content.std(),
        "oov_share": float((z == 0).mean()),
        "rare_share": float((z < 3).mean()),
        "very_rare_share": float((z < 2).mean()),
        "archaic_share": sum(w in ARCHAIC for w in lower) / n,
        "llm_word_share": sum(w in LLM_WORDS for w in lower) / n,
    }
    raw = text.encode("utf-8", "ignore")
    flat = re.sub(r"\s+", " ", text).strip().encode("utf-8", "ignore")
    L = max(len(flat), 1)
    f["zlib_ratio"] = len(zlib.compress(flat, 9)) / L
    f["lzma_ratio"] = len(lzma.compress(flat)) / L
    words_b = " ".join(lower).encode()
    f["zlib_words_ratio"] = len(zlib.compress(words_b, 9)) / max(len(words_b), 1)
    f["raw_minus_flat_len"] = (len(raw) - len(flat)) / max(len(raw), 1)

    # Line / poem structure (computed on non-empty lines, so layout-independent of blank lines)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    nl = max(len(lines), 1)
    last_words = [WORD.findall(l)[-1].lower() for l in lines if WORD.findall(l)]
    keys = [_rhyme_key(w) for w in last_words]
    adj = sum(a == b and wa != wb for a, b, wa, wb in zip(keys, keys[1:], last_words, last_words[1:]))
    alt = sum(a == b and wa != wb for a, b, wa, wb in zip(keys, keys[2:], last_words, last_words[2:]))
    f["rhyme_adjacent"] = adj / max(len(keys) - 1, 1)
    f["rhyme_alternate"] = alt / max(len(keys) - 2, 1)
    kc = Counter(keys)
    f["rhyme_key_concentration"] = (sum(c * c for c in kc.values()) / max(len(keys), 1) ** 2) if keys else 0.0
    f["line_end_comma_share"] = sum(l.endswith(",") for l in lines) / nl
    f["line_end_period_share"] = sum(l[-1:] in ".!?" for l in lines) / nl
    f["line_end_none_share"] = sum(l[-1:].isalnum() for l in lines) / nl
    f["line_start_lower_share"] = sum(l[:1].islower() for l in lines) / nl
    f["line_start_and_share"] = sum(l.lower().startswith(("and ", "but ", "or ")) for l in lines) / nl
    lw = [len(WORD.findall(l)) for l in lines]
    f["line_words_mean"] = float(np.mean(lw)) if lw else 0.0
    f["line_words_cv"] = float(np.std(lw) / np.mean(lw)) if lw and np.mean(lw) > 0 else 0.0
    f["is_versey"] = float(len(lines) >= 6 and np.median([len(l) for l in lines]) < 70)

    # punctuation habits that survive layout normalisation
    nc = max(len(text), 1)
    f["dash_any_rate"] = len(re.findall(r"--|—|–| - ", text)) / nc
    f["exclam_q_rate"] = len(re.findall(r"[!?]", text)) / nc
    f["semicolon_colon_rate"] = len(re.findall(r"[;:]", text)) / nc
    f["space_before_punct"] = len(re.findall(r" [,.;:!?]", text)) / nc
    f["no_space_after_punct"] = len(re.findall(r"[a-z][,.;:!?][A-Za-z]", text)) / nc
    f["typo_lower_i"] = len(re.findall(r"\bi\b", text)) / n
    f["digit_token_share"] = len(re.findall(r"\b\d+\b", text)) / n
    sent = [s for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text)) if s.strip()]
    sl = [len(WORD.findall(s)) for s in sent] or [0]
    f["flat_sent_len_mean"] = float(np.mean(sl))
    f["flat_sent_len_cv"] = float(np.std(sl) / np.mean(sl)) if np.mean(sl) > 0 else 0.0
    f["truncated_end"] = float(text.rstrip()[-1:].isalpha())
    return f


def v2_features(texts) -> pd.DataFrame:
    df = pd.DataFrame([v2_one(t) for t in texts]).astype(float)
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError("non-finite v2 features")
    return df
