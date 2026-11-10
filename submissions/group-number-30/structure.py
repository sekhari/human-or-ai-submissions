"""Data-grounded, exploratory structure features for the supplied passages.

Training-only inspection found formatting and a tail of extreme repetition more
promising than adjective or positivity counts. Those latter features remain
explicit weak hypothesis probes, not semantic or universal authorship rules.
NLTK supplies pretrained POS annotations, never authorship labels. No API calls.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted

WORD_RE = re.compile(r"\b\w+(?:['’]\w+)*\b", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|[\r\n]+")
FIRST_PERSON = frozenset("i me my mine we us our ours".split())
SECOND_PERSON = frozenset("you your yours".split())
HEDGES = frozenset("perhaps possibly probably seem seems might may arguably generally typically".split())
POSITIVE = frozenset("amazing beautiful delightful excellent fantastic great happy incredible lovely remarkable wonderful".split())


def _texts(X):
    if isinstance(X, (str, bytes)):
        raise ValueError("Expected a sequence of text documents, not a single string.")
    values = list(X)
    if any(not isinstance(text, str) for text in values):
        raise ValueError("Documents must be strings; missing text must be handled explicitly.")
    return values


@lru_cache(maxsize=1)
def _tagger():
    from nltk.tag import PerceptronTagger
    try:
        return PerceptronTagger(lang="eng")
    except LookupError as exc:
        raise RuntimeError(
            "Missing NLTK POS resource. Run python -m nltk.downloader "
            "averaged_perceptron_tagger_eng (with the documented NLTK_DATA path). "
            "POS extraction has no silent fallback."
        ) from exc


@lru_cache(maxsize=20000)
def pos_tags(text):
    """Process-local, label-independent tag cache; no cache enters saved models.

    Treebank word tokenization requires no extra tokenizer model. Sentence
    boundaries use punctuation/newline regexes and are approximate, especially
    for abbreviations, recipes, malformed HTML, and short list entries.
    """
    from nltk.tokenize import TreebankWordTokenizer
    tokenizer, tagger = TreebankWordTokenizer(), _tagger()
    tags = []
    for sentence in SENTENCE_RE.split(text):
        tags.extend(tag for _, tag in tagger.tag(tokenizer.tokenize(sentence)))
    return tuple(tags)


class POSTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        _texts(X)
        _tagger()
        self.fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self, "fitted_")
        return np.asarray([" ".join(pos_tags(text)) for text in _texts(X)], dtype=object)


class StyleFeatures(TransformerMixin, BaseEstimator):
    """Surface/POS statistics, including measured weak handwritten hypotheses.

    ``use_pos=False`` explicitly selects a surface-only ablation. Word/character
    normalization is fixed, learned scaling belongs in a training-only Pipeline.
    HTML and newline features can be source/domain shortcuts: their validation
    utility does not establish robustness to a changed distribution.
    """
    SURFACE_NAMES = (
        "log_words", "log_characters", "mean_word_characters", "type_token_ratio",
        "mean_sentence_words", "sentence_length_cv", "repeated_bigram_ratio",
        "repeated_trigram_ratio", "newline_per_100_words", "html_break_per_100_words",
        "digit_character_ratio", "comma_character_ratio", "semicolon_character_ratio",
        "exclamation_character_ratio", "question_character_ratio", "dash_character_ratio",
        "first_person_word_ratio", "second_person_word_ratio", "hedge_word_ratio",
        "positive_lexicon_word_ratio", "serial_list_pattern_per_sentence",
    )
    POS_NAMES = (
        "adjective_pos_ratio", "adverb_pos_ratio", "superlative_pos_ratio",
        "proper_noun_pos_ratio", "pronoun_pos_ratio", "pos_entropy",
        "adjacent_adjectives_pos_ratio",
    )

    def __init__(self, use_pos=True):
        self.use_pos = use_pos

    def fit(self, X, y=None):
        _texts(X)
        if self.use_pos:
            _tagger()
        self.feature_names_ = np.asarray(self.SURFACE_NAMES + (self.POS_NAMES if self.use_pos else ()), dtype=object)
        return self

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_")
        return self.feature_names_.copy()

    def transform(self, X):
        check_is_fitted(self, "feature_names_")
        return np.asarray([self._row(text) for text in _texts(X)], dtype=float).reshape(-1, len(self.feature_names_))

    def _row(self, text):
        words = WORD_RE.findall(text.lower())
        n, chars = len(words), len(text)
        wd, cd = max(n, 1), max(chars, 1)
        sentence_sizes = [len(WORD_RE.findall(s)) for s in SENTENCE_RE.split(text) if WORD_RE.search(s)]
        sentence_mean = float(np.mean(sentence_sizes)) if sentence_sizes else 0.0
        sentence_cv = float(np.std(sentence_sizes)) / max(sentence_mean, 1.0) if sentence_sizes else 0.0

        def repeated(order):
            grams = Counter(tuple(words[i:i + order]) for i in range(max(n - order + 1, 0)))
            return sum(v - 1 for v in grams.values()) / max(n - order + 1, 1)

        row = [
            np.log1p(n), np.log1p(chars), sum(map(len, words)) / wd,
            len(set(words)) / wd, sentence_mean, sentence_cv, repeated(2), repeated(3),
            text.count("\n") * 100 / wd, len(re.findall(r"<br\s*/?>", text, re.I)) * 100 / wd,
            sum(c.isdigit() for c in text) / cd, text.count(",") / cd,
            text.count(";") / cd, text.count("!") / cd, text.count("?") / cd,
            sum(text.count(dash) for dash in ("-", "–", "—")) / cd,
            sum(w in FIRST_PERSON for w in words) / wd,
            sum(w in SECOND_PERSON for w in words) / wd,
            sum(w in HEDGES for w in words) / wd,
            sum(w in POSITIVE for w in words) / wd,
            len(re.findall(r"[^,;.!?]+,\s*[^,;.!?]+,\s*(?:and|or)\b", text, re.I)) / max(len(sentence_sizes), 1),
        ]
        if self.use_pos:
            tags = pos_tags(text)
            td = max(len(tags), 1)
            probabilities = np.asarray(list(Counter(tags).values()), dtype=float) / td
            row.extend([
                sum(t.startswith("JJ") for t in tags) / td,
                sum(t.startswith("RB") for t in tags) / td,
                sum(t in ("JJS", "RBS") for t in tags) / td,
                sum(t in ("NNP", "NNPS") for t in tags) / td,
                sum(t in ("PRP", "PRP$", "WP", "WP$") for t in tags) / td,
                -float(np.sum(probabilities * np.log(probabilities))),
                sum(a.startswith("JJ") and b.startswith("JJ") for a, b in zip(tags, tags[1:])) / max(len(tags) - 1, 1),
            ])
        return row


class ClassConditionalPOSSurprisal(TransformerMixin, BaseEstimator):
    """A testable POS predictability hypothesis, not a grammatical-error count.

    Two add-alpha smoothed POS n-gram models learn from supplied training labels.
    fit_transform uses stratified out-of-fold scores to train a downstream
    classifier, then fits final language models on all training rows. transform
    never changes their state. In-sample scores later produced by transform can
    still be optimistic; use held-out comparisons to assess utility.
    """
    FEATURE_NAMES = (
        "human_mean_surprise", "ai_mean_surprise", "human_minus_ai_mean_surprise",
        "human_std_surprise", "ai_std_surprise", "human_mode_mismatch",
        "ai_mode_mismatch", "human_minus_ai_mode_mismatch",
        "human_unseen_context_ratio", "ai_unseen_context_ratio",
    )

    def __init__(self, order=2, alpha=0.5, cv=3, random_state=20261023):
        self.order = order
        self.alpha = alpha
        self.cv = cv
        self.random_state = random_state

    def _validate(self, X, y):
        texts, labels = _texts(X), np.asarray(y)
        if labels.ndim != 1 or len(texts) != len(labels) or set(labels.tolist()) != {0, 1}:
            raise ValueError("Expected matching documents and binary 0/1 labels, with both classes.")
        if self.order not in (2, 3) or not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("order must be 2 or 3 and alpha finite and positive.")
        return texts, labels

    def _events(self, tags):
        sequence = ["<BOS>"] * (self.order - 1)
        sequence += [t if t in self.vocabulary_ else "<UNK>" for t in tags] + ["<EOS>"]
        for i in range(self.order - 1, len(sequence)):
            yield tuple(sequence[i - self.order + 1:i]), sequence[i]

    def fit(self, X, y):
        texts, labels = self._validate(X, y)
        sequences = [pos_tags(t) for t in texts]
        self.vocabulary_ = frozenset(t for tags in sequences for t in tags) | {"<UNK>", "<EOS>"}
        self.models_, self.totals_, self.modes_, self.defaults_ = {}, {}, {}, {}
        for label in (0, 1):
            counts, marginal = defaultdict(Counter), Counter()
            for tags, target in zip(sequences, labels):
                if target == label:
                    for context, token in self._events(tags):
                        counts[context][token] += 1
                        marginal[token] += 1
            self.models_[label] = dict(counts)
            self.totals_[label] = {k: sum(v.values()) for k, v in counts.items()}
            self.modes_[label] = {k: min(v, key=lambda t: (-v[t], t)) for k, v in counts.items()}
            self.defaults_[label] = min(marginal, key=lambda t: (-marginal[t], t))
        self.n_training_samples_ = len(texts)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        if fit_params:
            raise ValueError("Unexpected fit parameters.")
        texts, labels = self._validate(X, y)
        if not isinstance(self.cv, int) or self.cv < 2:
            raise ValueError("cv must be an integer >= 2.")
        folds = min(self.cv, int(min(np.sum(labels == 0), np.sum(labels == 1))))
        if folds < 2:
            raise ValueError("Cross-fitting requires at least two training documents per class.")
        values = np.empty((len(texts), len(self.FEATURE_NAMES)))
        split = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.random_state)
        for train, holdout in split.split(texts, labels):
            model = type(self)(self.order, self.alpha, self.cv, self.random_state)
            model.fit([texts[i] for i in train], labels[train])
            values[holdout] = model.transform([texts[i] for i in holdout])
        self.fit(texts, labels)
        self.cross_fit_folds_ = folds
        return values

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "models_")
        return np.asarray(self.FEATURE_NAMES, dtype=object)

    def transform(self, X):
        check_is_fitted(self, "models_")
        rows = []
        for text in _texts(X):
            events, stats = list(self._events(pos_tags(text))), []
            for label in (0, 1):
                surprises, mismatches, unseen = [], 0, 0
                for context, tag in events:
                    counts = self.models_[label].get(context)
                    observed = counts.get(tag, 0) if counts is not None else 0
                    total = self.totals_[label].get(context, 0)
                    probability = (observed + self.alpha) / (total + self.alpha * len(self.vocabulary_))
                    surprises.append(-np.log(probability))
                    mismatches += tag != self.modes_[label].get(context, self.defaults_[label])
                    unseen += counts is None
                stats.append((np.mean(surprises), np.std(surprises), mismatches / len(events), unseen / len(events)))
            h, a = stats
            rows.append([h[0], a[0], h[0] - a[0], h[1], a[1], h[2], a[2], h[2] - a[2], h[3], a[3]])
        return np.asarray(rows, dtype=float).reshape(-1, len(self.FEATURE_NAMES))
