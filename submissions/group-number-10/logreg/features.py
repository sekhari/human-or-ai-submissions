"""Text cleaning and feature blocks for the logistic regression track.

Every block has the same shape:
    block.fit_transform(texts, vocab_texts) -> sparse matrix for the rows the model trains on
    block.transform(texts)                  -> sparse matrix for any other rows
Anything learned (tf-idf vocabulary, mean/std for z-scoring) comes from fit_transform's inputs.
vocab_texts is extra unlabeled text (unlabeled.csv + test.csv) the tf-idf vocabulary is built from;
no labels are involved, but it does mean the vocabulary has seen the test passages.
"""

import gzip
import hashlib
import re
import unicodedata

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from logreg import RESULTS_DIR


ZERO_WIDTH = ['​', '‌', '‍', '⁠', '﻿']

# Cyrillic/Greek letters that look like Latin ones (RAID's homoglyph attack)
HOMOGLYPHS = str.maketrans('аеорсухіјԁԛѕΑΒΕΗΙΚΜΝΟΡΤΧΥΖ',
                           'aeopcyxijdqsABEHIKMNOPTXYZ')

# curly quotes / long dashes -> plain ascii, so "don’t" and "don't" are the same word
# (test has 1.5-3x more curly quotes and 3.6x more em dashes than train)
PUNCT_MAP = str.maketrans({'’': "'", '‘': "'", '“': '"', '”': '"', '—': '-', '–': '-'})


def clean_text(text):
    text = unicodedata.normalize('NFKC', text)   # fancy unicode -> plain forms
    for z in ZERO_WIDTH:
        text = text.replace(z, '')
    text = text.translate(HOMOGLYPHS).translate(PUNCT_MAP)
    # tf-idf throws away whitespace, so turn each newline into a visible token
    return text.replace('\n', ' NEWLINE ')


class TfidfBlock:
    def __init__(self, **tfidf_args):
        self.tfidf_args = tfidf_args

    def fit_transform(self, texts, vocab_texts=()):
        self.vec = TfidfVectorizer(**self.tfidf_args)
        self.vec.fit([clean_text(t) for t in [*texts, *vocab_texts]])
        return self.transform(texts)

    def transform(self, texts):
        return self.vec.transform([clean_text(t) for t in texts])


class DenseBlock:
    """A handful of numeric features per text, z-scored with the fit rows' mean/std."""

    def __init__(self, featurize):
        self.featurize = featurize   # list of texts -> (n, d) array

    def fit_transform(self, texts, vocab_texts=()):
        X = self.featurize(texts)
        self.mean, self.std = X.mean(axis=0), X.std(axis=0) + 1e-9
        return sp.csr_matrix((X - self.mean) / self.std)

    def transform(self, texts):
        return sp.csr_matrix((self.featurize(texts) - self.mean) / self.std)


# --- style block --------------------------------------------------------------

def style_features(text):
    words = re.findall(r"[A-Za-z']+", text)
    lower = [w.lower() for w in words]
    n_words = max(len(words), 1)
    n_chars = max(len(text), 1)

    sentences = [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s]
    sent_lens = np.array([len(s.split()) for s in sentences]) if sentences else np.array([0])

    word_counts = pd.Series(lower, dtype=object).value_counts()
    trigrams = pd.Series(list(zip(lower, lower[1:], lower[2:])), dtype=object).value_counts()
    raw_bytes = text.encode('utf-8')

    return {
        # --- length / structure ---
        'log_length':          np.log1p(len(text)),
        'newline_rate':        text.count('\n') / n_chars,
        'ends_mid_sentence':   float(not text.rstrip().endswith(('.', '!', '?'))),
        'sent_len_mean':       sent_lens.mean(),
        'sent_len_std':        sent_lens.std(),
        'sent_len_cv':         sent_lens.std() / (sent_lens.mean() + 1e-9),   # "burstiness"

        # --- vocabulary / repetition ---
        'vocab_diversity':     len(word_counts) / n_words,
        'hapax_rate':          (word_counts == 1).sum() / n_words,            # words used once
        'repeated_trigrams':   (trigrams[trigrams > 1] - 1).sum() / n_words,  # looping text
        'gzip_ratio':          len(gzip.compress(raw_bytes)) / max(len(raw_bytes), 1),

        # --- punctuation / casing ---
        'punct_rate':          sum(c in ',;:' for c in text) / n_words,
        'upper_rate':          sum(c.isupper() for c in text) / n_chars,
        'contraction_rate':    len(re.findall(r"\w+['’]\w+", text)) / n_words,
        'curly_quote_rate':    sum(c in '’‘“”' for c in text) / n_chars,     # kept as a signal here
        'em_dash_rate':        sum(c in '—–' for c in text) / n_words,

        # --- attack traces ---
        'non_ascii_rate':      sum(ord(c) > 127 for c in text) / n_chars,
        'zero_width_count':    sum(text.count(z) for z in ZERO_WIDTH),
        'homoglyph_count':     sum(a != b for a, b in zip(text, text.translate(HOMOGLYPHS))),
        'double_space_rate':   text.count('  ') / n_chars,
        'mid_word_caps_rate':  len(re.findall(r'[a-z][A-Z]', text)) / n_words,
    }


def style_matrix(texts):
    return pd.DataFrame([style_features(t) for t in texts]).to_numpy(dtype=float)


# --- language-model block ------------------------------------------------------
# How "predictable" each text is to GPT-2. AI text tends to sit close to what a
# model would have written. Slow the first time (~10-20 min), so it's cached,
# keyed by a hash of the text so any caller (train, val, test, new text) can use it.

LM_CACHE = RESULTS_DIR / 'lm_cache_gpt2.csv'
LM_MAX_TOKENS = 256
LM_COLUMNS = ['lm_logp_mean', 'lm_logp_std', 'lm_logrank', 'lm_top1', 'lm_top10', 'lm_entropy']


def lm_available():
    try:
        import torch, transformers  # noqa: F401
        return True
    except ImportError:
        return False


def text_key(text):
    return hashlib.sha1(text.encode('utf-8')).hexdigest()


def score_with_gpt2(texts):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.utils import get_device

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained('gpt2').to(device).eval()

    rows = []
    with torch.no_grad():
        for i, text in enumerate(texts):
            ids = tokenizer(text, return_tensors='pt', truncation=True,
                            max_length=LM_MAX_TOKENS).input_ids.to(device)
            if ids.shape[1] < 3:
                rows.append([0.0] * len(LM_COLUMNS))
                continue

            log_probs = model(ids).logits[0, :-1].log_softmax(-1)   # model's guess for each next token
            actual = ids[0, 1:]                                      # the token that actually came next
            token_logp = log_probs[torch.arange(len(actual)), actual]
            rank = (log_probs > token_logp[:, None]).sum(-1) + 1     # 1 = model's top guess
            entropy = -(log_probs.exp() * log_probs).sum(-1)

            rows.append([
                token_logp.mean().item(),                  # avg log-prob
                token_logp.std().item(),                   # spread of log-prob
                rank.float().log().mean().item(),          # avg log-rank
                (rank == 1).float().mean().item(),         # share of top-1 guesses
                (rank <= 10).float().mean().item(),        # share of top-10 guesses
                entropy.mean().item(),                     # avg model uncertainty
            ])
            if (i + 1) % 500 == 0:
                print(f'    gpt2 scored {i + 1}/{len(texts)}')
    return rows


def lm_matrix(texts):
    """GPT-2 features for texts; only texts missing from the cache get scored."""
    if LM_CACHE.exists():
        cache = pd.read_csv(LM_CACHE)
    else:
        cache = pd.DataFrame(columns=['key'] + LM_COLUMNS)

    keys = [text_key(t) for t in texts]
    known = set(cache['key'])
    missing = {k: t for k, t in zip(keys, texts) if k not in known}
    if missing:
        print(f'  scoring {len(missing)} texts with gpt2...')
        new = pd.DataFrame(score_with_gpt2(list(missing.values())), columns=LM_COLUMNS)
        new.insert(0, 'key', list(missing.keys()))
        cache = pd.concat([cache, new], ignore_index=True)
        LM_CACHE.parent.mkdir(parents=True, exist_ok=True)
        cache.to_csv(LM_CACHE, index=False)

    return cache.set_index('key').loc[keys, LM_COLUMNS].to_numpy(dtype=float)


# =============================================================================
# Registry
# =============================================================================

BLOCKS = {
    # word + word-pair counts; token_pattern keeps 1-letter words like "I" and "a"
    'words': lambda: TfidfBlock(ngram_range=(1, 2), min_df=3, sublinear_tf=True,
                                lowercase=False, token_pattern=r'(?u)\b\w+\b'),
    # 3-5 character chunks inside words: spelling, suffixes, punctuation habits
    'chars': lambda: TfidfBlock(analyzer='char_wb', ngram_range=(3, 5), min_df=5,
                                sublinear_tf=True, lowercase=False),
    'style': lambda: DenseBlock(style_matrix),
    'lm':    lambda: DenseBlock(lm_matrix),
}


def available_blocks():
    return [n for n in BLOCKS if n != 'lm' or lm_available()]
