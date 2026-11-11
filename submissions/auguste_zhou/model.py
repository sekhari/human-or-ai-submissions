"""Shared TF-IDF training and batched prediction."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import re


def extract_style_features(text):
    """Return the original 27 stylometric features in their trained order."""
    words = re.findall(r"\b[\w']+\b", text.lower())
    word_count = len(words)

    if words:
        word_lengths = np.array([len(word) for word in words])
    else:
        word_lengths = np.array([0])

    sentences = [
        sentence
        for sentence in re.split(r"[.!?]+", text)
        if sentence.strip()
    ]

    sentence_count = max(1, len(sentences))
    sentence_lengths = [
        len(re.findall(r"\b[\w']+\b", sentence))
        for sentence in sentences
    ]

    safe_word_count = max(1, word_count)
    safe_character_count = max(1, len(text))
    unique_word_count = len(set(words))

    bigrams = list(zip(words, words[1:]))
    trigrams = list(zip(words, words[1:], words[2:]))

    paragraphs = [
        paragraph
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]
    paragraph_count = len(paragraphs)

    # Text length and sentence structure (7 features)
    text_structure = [
        np.log1p(word_count),
        np.log1p(safe_character_count),
        np.mean(word_lengths),
        np.std(word_lengths),
        np.mean(sentence_lengths),
        np.std(sentence_lengths),
        sentence_count / safe_word_count,
    ]

    # Vocabulary diversity (4 features)
    vocabulary = [
        unique_word_count / safe_word_count,
        unique_word_count / np.sqrt(safe_word_count),
        sum(len(word) <= 3 for word in words) / safe_word_count,
        sum(len(word) >= 8 for word in words) / safe_word_count,
    ]

    # Repeated phrases (2 features)
    repetition = [
        (len(bigrams) - len(set(bigrams))) / max(1, len(bigrams)),
        (len(trigrams) - len(set(trigrams))) / max(1, len(trigrams)),
    ]

    # Punctuation ratios per word (9 features)
    punctuation = [
        text.count(",") / safe_word_count,
        text.count(";") / safe_word_count,
        text.count(":") / safe_word_count,
        text.count("?") / safe_word_count,
        text.count("!") / safe_word_count,
        text.count("(") / safe_word_count,
        text.count('"') / safe_word_count,
        text.count("'") / safe_word_count,
        text.count("-") / safe_word_count,
    ]

    # Paragraph and whitespace structure (2 features)
    paragraphs_and_lines = [
        text.count("\n") / safe_word_count,
        paragraph_count / safe_word_count,
    ]

    # Character composition (3 features)
    character_composition = [
        sum(character.isdigit() for character in text) / safe_character_count,
        sum(character.isupper() for character in text) / safe_character_count,
        sum(character.isspace() for character in text) / safe_character_count,
    ]

    return (
        text_structure
        + vocabulary
        + repetition
        + punctuation
        + paragraphs_and_lines
        + character_composition
    )


def fit_model(texts, labels):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=26000,
                          sublinear_tf=True, strip_accents='unicode', dtype=np.float32)
    char = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=3,
                          max_features=35000, sublinear_tf=True, dtype=np.float32)
    style = np.asarray([extract_style_features(text) for text in texts], dtype=np.float32)
    scaler = StandardScaler().fit(style)
    style = np.clip(scaler.transform(style), -5, 5).astype(np.float32)
    features = hstack([word.fit_transform(texts), char.fit_transform(texts),
                       csr_matrix(style * 0.1)], format='csr')
    classifier = SGDClassifier(loss='log_loss', penalty=None, max_iter=5,
                               tol=None, random_state=42, average=False)
    classifier.fit(features, np.asarray(labels).astype(int))
    return {'word': word, 'char': char, 'scaler': scaler, 'classifier': classifier}


def predict_probabilities(bundle, texts):
    probabilities = np.empty(len(texts))
    for start in range(0, len(texts), 100):
        batch = texts.iloc[start:start + 100]
        parts = [bundle['word'].transform(batch), bundle['char'].transform(batch)]
        if 'scaler' in bundle:
            style = np.asarray([extract_style_features(text) for text in batch], dtype=np.float32)
            style = np.clip(bundle['scaler'].transform(style), -5, 5).astype(np.float32)
            parts.append(csr_matrix(style * 0.1))
        features = hstack(parts, format='csr')
        probabilities[start:start + 100] = bundle['classifier'].predict_proba(features)[:, 1]
    return probabilities
