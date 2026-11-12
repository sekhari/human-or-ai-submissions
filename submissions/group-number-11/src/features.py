"""Text-only representations; IDs are never predictive inputs."""
import re
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def normalize(text):
    return str(text).replace('\\n', '\n').replace('\\t', '\t')


def vectorizer(kind):
    if kind == 'word':
        return TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2),
            min_df=2, max_df=.995, sublinear_tf=True, max_features=180000,
            strip_accents='unicode', dtype=np.float32)
    return TfidfVectorizer(preprocessor=normalize, analyzer='char',
        ngram_range=(2, 5), min_df=3, sublinear_tf=True,
        max_features=180000, dtype=np.float32)


FUNCTION_WORDS = ('the a an and or but if while because although however therefore '
    'thus moreover furthermore in on at to for of with by from as is are was were '
    'be been being have has had do does did will would can could should may might '
    'must i you he she it we they my your his her its our their this that these '
    'those which who what when where why how not no so very just also than then '
    'there here some any all each every both such into about over under through').split()


def style_feature_names():
    return (['log_characters','log_words','type_token_ratio','hapax_ratio',
        'word_length_mean','word_length_std','word_length_median',
        'sentence_words_mean','sentence_words_std','sentence_words_max','sentences_per_word',
        'initial_uppercase_ratio','all_uppercase_ratio','digit_ratio','non_ascii_ratio',
        'lines_per_word','blank_lines_per_word','long_word_ratio','short_word_ratio','repeated_type_ratio']
        + [f'character_{repr(p)}' for p in string.punctuation + '“”‘’–—\n\t']
        + [f'function_{w}' for w in FUNCTION_WORDS]
        + ['contractions','parentheticals','bracketed_spans','urls','four_digit_numbers','list_start','multiple_spaces'])


def style_features(texts):
    rows = []
    for raw in texts:
        t = normalize(raw)
        words = re.findall(r"\b\w+(?:['’-]\w+)*\b", t)
        low = [w.lower() for w in words]
        n, c = max(len(words), 1), max(len(t), 1)
        lengths = np.array([len(w) for w in words] or [0])
        sentences = [s.split() for s in re.split(r'[.!?]+\s*', t) if s.strip()]
        sl = np.array([len(s) for s in sentences] or [0])
        counts = {w: low.count(w) for w in set(low)}
        row = [np.log1p(c), np.log1p(n), len(set(low))/n,
            sum(v == 1 for v in counts.values())/n,
            lengths.mean(), lengths.std(), np.median(lengths),
            sl.mean(), sl.std(), sl.max(), len(sentences)/n,
            sum(w[0].isupper() for w in words)/n,
            sum(w.isupper() for w in words)/n,
            sum(x.isdigit() for x in t)/c,
            sum(ord(x)>127 for x in t)/c,
            len(t.splitlines())/n, t.count('\n\n')/n,
            sum(len(w)>8 for w in words)/n,
            sum(len(w)<4 for w in words)/n,
            sum(v>1 for v in counts.values())/max(len(counts),1)]
        row += [t.count(p)/c for p in string.punctuation + '“”‘’–—\n\t']
        row += [counts.get(w,0)/n for w in FUNCTION_WORDS]
        row += [len(re.findall(p,t))/n for p in
            [r'\b\w+[’\']\w+\b',r'\([^)]*\)',r'\[[^]]*\]',
             r'https?://',r'\b\d{4}\b',r'^\s*[-*\d]+[.)]?\s',r'  +']]
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)
