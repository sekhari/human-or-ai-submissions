"""Q1: model comparison and hyperparameter selection.

What this script answers (report section 3, Q1)
-----------------------------------------------
Q1.1/Q1.2  Compare 5 models that use genuinely different text representations.
Q1.3       Pick each model's hyperparameters with a protocol that does not touch validation.
Q1.4       Pick one model to carry forward (that decision is made from the output CSVs).

Evaluation protocol
-------------------
1. common.get_split() gives a fixed, stratified 2,000-row validation set. It is NEVER
   used to choose hyperparameters; it is scored exactly once per final model.
2. Hyperparameters are chosen by 5-fold stratified cross-validation AUROC on the
   8,000 training rows only.
3. Each vectorizer sits INSIDE a sklearn Pipeline. That means inside every CV fold
   the TF-IDF vocabulary and IDF weights are learned from that fold's training part
   only. Fitting TF-IDF on all 8,000 rows first, then cross-validating, would leak
   document-frequency information from the held-out fold and inflate CV scores.
4. The best configuration is refit on all 8,000 rows, then scored on validation.

Why AUROC
---------
It is the Kaggle metric. AUROC depends only on how the scores RANK the passages,
not on whether they are calibrated probabilities, which is why an SVM's
decision_function is a valid score here (see score() below).

Two text views (--text_view)
----------------------------
raw  : text exactly as given.
norm : all whitespace collapsed (common.normalize_ws). This removes the paragraph-
       break artifact. A model whose AUROC drops a lot under "norm" was relying on
       formatting rather than wording.

Usage
-----
python q1_model_comparison.py --text_view raw     # full grids, ~20 min on 2 cores
python q1_model_comparison.py --text_view norm
python q1_model_comparison.py --quick             # tiny grids: a 2-minute smoke test
python q1_model_comparison.py --only A_word_tfidf_lr B_char_tfidf_svm   # run a subset

Outputs (in results/, suffixed _raw or _norm)
---------------------------------------------
q1_model_comparison_*.csv : one row per model: best params, CV / train / val AUROC, times.
q1_grid_search_*.csv      : every grid point's CV mean and std (the Q1.3 table).
val_scores_<model>_*.csv  : per-passage validation scores (reused for error analysis / ensembling).
"""
import argparse
import json
import os
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from common import SEED, get_split, normalize_ws

OUT_DIR = os.path.join(os.path.dirname(__file__), "results")


# =============================================================================
# Stylometric features (used by model D)
# =============================================================================
class StyleFeatures(BaseEstimator, TransformerMixin):
    """Turns each passage into 25 numbers describing HOW it is written, not WHAT it is about.

    Motivation: TF-IDF models mostly learn topic words. The test set contains domains
    and generators not seen in training, so topic words may not transfer. Writing-style
    signals (sentence rhythm, vocabulary richness, repetition, punctuation habits) are
    more likely to.

    Subclassing BaseEstimator + TransformerMixin makes this a regular sklearn transformer,
    so it can sit in a Pipeline and be cross-validated like any vectorizer. It has nothing
    to learn (every feature is computed per passage), so fit() does nothing.

    Every count is divided by the passage's word or character count. Passages range from
    75 to 350 words, and raw counts would mostly encode length.
    """

    # Sentence splitter: end punctuation followed by whitespace/closing quote/bracket,
    # OR any newline (list items and recipe lines count as their own "sentences").
    _sent = re.compile(r"[.!?]+[\s\"')\]]+|\n+")
    # A "word" = letters and apostrophes (so "don't" is one word, numbers are ignored).
    _word = re.compile(r"[A-Za-z']+")

    def fit(self, X, y=None):
        return self  # stateless: nothing to learn

    def _one(self, t):
        """Compute the feature vector for a single passage t."""
        words = self._word.findall(t)
        lw = [w.lower() for w in words]
        n = max(len(words), 1)    # word count; max(...,1) avoids division by zero
        nc = max(len(t), 1)       # character count
        sents = [s for s in self._sent.split(t) if s.strip()]
        # Words per sentence, one entry per sentence.
        slen = np.array([len(self._word.findall(s)) for s in sents]) if sents else np.array([0])
        cnt = Counter(lw)         # word -> frequency
        bigrams = list(zip(lw, lw[1:]))
        trigrams = list(zip(lw, lw[1:], lw[2:]))
        return [
            # --- length and sentence structure ---
            np.log1p(len(words)),                             # passage length (log: diminishing effect)
            np.mean([len(w) for w in words]) if words else 0, # mean word length
            len(sents),                                       # number of sentences
            slen.mean(), slen.std(), slen.max(),              # sentence-length mean / spread / max
            slen.std() / (slen.mean() + 1e-6),                # "burstiness": humans vary sentence length
                                                              # more than LLMs (coefficient of variation)
            # --- vocabulary richness ---
            len(cnt) / n,                                     # type-token ratio: distinct words / words
            sum(1 for c in cnt.values() if c == 1) / n,       # hapax ratio: words used exactly once
            sum(1 for w in lw if w in ENGLISH_STOP_WORDS) / n,# function-word rate ("the", "of", ...)
            # --- repetition ---
            1 - len(set(bigrams)) / max(len(bigrams), 1),     # share of repeated bigrams
            1 - len(set(trigrams)) / max(len(trigrams), 1),   # share of repeated trigrams: catches the
                                                              # degenerate loops some generators produce
            # --- punctuation habits (per word) ---
            t.count(",") / n, t.count(";") / n, t.count(":") / n,
            (t.count("-") + t.count("–") + t.count("—")) / n,   # hyphen, en dash, em dash
            (t.count('"') + t.count("“") + t.count("”")) / n,   # straight and curly double quotes
            t.count("'") / n, t.count("(") / n,
            t.count("!") / n, t.count("?") / n,
            # --- character-level (per character) ---
            sum(ch.isdigit() for ch in t) / nc,               # digit rate
            sum(ch.isupper() for ch in t) / nc,               # capital-letter rate
            t.count("\n") / n,                                # newline rate. NOTE: this one carries the
                                                              # formatting leak; dropping it alone costs
                                                              # 0.025 AUROC on raw text
            sum(1 for ch in t if ord(ch) > 127) / nc,         # non-ASCII rate (smart quotes, accents,
                                                              # and homoglyph-style adversarial edits)
        ]

    def transform(self, X):
        # Output is a dense (n_passages, 25) float matrix.
        return np.array([self._one(t) for t in X], dtype=np.float64)


# =============================================================================
# The five candidate models
# =============================================================================
def build_models(quick=False):
    """Return {name: (description, pipeline, param_grid)}.

    Model names start with a letter (A-E) so the result tables sort in a stable order.

    Grid parameter keys use sklearn's "<step>__<param>" convention:
    "vec__min_df" means the min_df argument of the Pipeline step named "vec".

    quick=True swaps every grid for one or two values, for a fast smoke test.
    """
    # g(full, small): pick the full grid normally, the small one under --quick.
    g = (lambda full, small: small if quick else full)

    # Word-tokenizer settings shared by models A, C and E.
    #   token_pattern: a token is either a run of word characters OR a single
    #   punctuation mark. sklearn's default pattern throws punctuation away, and
    #   punctuation use is one of the clearer human-vs-AI differences.
    #   sublinear_tf: use 1 + log(tf) so a word repeated 10 times is not 10x as important.
    word = dict(lowercase=True, sublinear_tf=True, strip_accents=None, token_pattern=r"(?u)\b\w+\b|[^\w\s]")

    return {
        # A: the course baseline family, tuned. Linear models on sparse TF-IDF are strong
        # for text classification and fast. C = inverse regularization strength
        # (larger = weaker L2 penalty). liblinear is a good solver for small sparse data.
        "A_word_tfidf_lr": (
            "Word TF-IDF (unigram/bigram, punctuation kept as tokens) + logistic regression",
            Pipeline([("vec", TfidfVectorizer(**word)),
                      ("clf", LogisticRegression(max_iter=3000, solver="liblinear"))]),
            {"vec__ngram_range": g([(1, 1), (1, 2)], [(1, 2)]),   # unigrams only vs. + bigrams
             "vec__min_df": g([1, 2, 5], [2]),                    # drop words seen in < k documents
             "clf__C": g([0.3, 1, 3, 10, 30], [1, 10])},
        ),

        # B: character n-grams. "char_wb" builds n-grams only inside word boundaries
        # (each word padded with spaces). This captures sub-word patterns: suffixes,
        # spelling, punctuation attached to words. It is also more robust to typos and
        # tokenization tricks than whole words. Case is kept (lowercase=False) because
        # capitalization is itself a style signal. max_features caps memory.
        # LinearSVC has no predict_proba; score() uses its decision_function instead.
        "B_char_tfidf_svm": (
            "Char_wb TF-IDF n-grams + linear SVM (decision_function as score)",
            Pipeline([("vec", TfidfVectorizer(analyzer="char_wb", lowercase=False, sublinear_tf=True,
                                              min_df=3, max_features=300_000)),
                      ("clf", LinearSVC(dual="auto", max_iter=5000))]),
            {"vec__ngram_range": g([(2, 4), (3, 5), (1, 5)], [(3, 5)]),
             "clf__C": g([0.03, 0.1, 0.3, 1], [0.1, 1])},
        ),

        # C: generative baseline. Multinomial NB models word COUNTS, so it uses
        # CountVectorizer instead of TF-IDF (that is why sublinear_tf is filtered out of
        # the shared settings). alpha = additive smoothing for words unseen in a class.
        # Expected to be the weakest model; it is here as a cheap, different reference point.
        "C_word_counts_nb": (
            "Word count unigram+bigram + multinomial naive Bayes",
            Pipeline([("vec", CountVectorizer(ngram_range=(1, 2), min_df=2, **{k: v for k, v in word.items() if k != "sublinear_tf"})),
                      ("clf", MultinomialNB())]),
            {"clf__alpha": g([0.01, 0.03, 0.1, 0.3, 1.0], [0.1, 1.0])},
        ),

        # D: dense hand-crafted features + gradient-boosted trees. Trees can model
        # non-linear interactions between the 25 features (e.g. "short sentences AND
        # low vocabulary richness") that a linear model cannot. HistGradientBoosting is
        # sklearn's fast LightGBM-style implementation. early_stopping is off so that
        # max_iter means exactly what the grid says.
        "D_style_hgb": (
            "25 hand-crafted stylometric features + histogram gradient boosting",
            Pipeline([("feat", StyleFeatures()),
                      ("clf", HistGradientBoostingClassifier(random_state=SEED, early_stopping=False))]),
            {"clf__learning_rate": g([0.03, 0.1], [0.1]),
             "clf__max_iter": g([200, 500], [200]),       # number of boosting rounds (trees)
             "clf__max_leaf_nodes": g([15, 31], [31])},   # tree complexity
        ),

        # E: word and character features side by side. FeatureUnion runs both
        # vectorizers on the same text and concatenates their sparse matrices
        # column-wise; one logistic regression then weighs both views jointly.
        # The vectorizer settings are fixed at sensible values (not tuned) to keep
        # the grid small; only C is searched.
        "E_word+char_lr": (
            "Word (1-2) + char_wb (3-5) TF-IDF concatenated + logistic regression",
            Pipeline([("vec", FeatureUnion([
                          ("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2, **word)),
                          ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=False,
                                                sublinear_tf=True, min_df=3, max_features=300_000))])),
                      ("clf", LogisticRegression(max_iter=3000, solver="liblinear"))]),
            {"clf__C": g([1, 3, 10, 30], [10])},
        ),
    }


def score(model, X):
    """Return one real-valued score per passage, higher = more likely AI.

    AUROC only needs a ranking, so any monotone score works: predict_proba for models
    that have it, the SVM's signed distance to the hyperplane (decision_function)
    otherwise. Note: for the Kaggle submission file, SVM scores would have to be
    mapped to [0, 1] (e.g. calibration or a sigmoid); AUROC is unchanged by that.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]   # column 1 = P(label == 1) = P(AI)
    return model.decision_function(X)


def main():
    # ---------------------------------------------------------------- arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny grids for a fast smoke test")
    ap.add_argument("--only", nargs="*", help="subset of model names")
    ap.add_argument("--n_jobs", type=int, default=-1, help="parallel CV jobs; -1 = all cores")
    ap.add_argument("--text_view", choices=["raw", "norm"], default="raw",
                    help="raw = text as given; norm = all whitespace collapsed to single spaces")
    args = ap.parse_args()

    # ---------------------------------------------------------------- data
    tr, va = get_split()
    # .values gives numpy arrays; sklearn vectorizers accept any iterable of strings.
    Xtr, ytr, Xva, yva = tr["text"].values, tr["label"].values, va["text"].values, va["label"].values
    if args.text_view == "norm":
        # Normalize BOTH sides: a model trained on normalized text must be scored on it too.
        Xtr, Xva = normalize_ws(Xtr), normalize_ws(Xva)
    print(f"[{args.text_view}] train {len(tr)} (AI={ytr.mean():.2f}) | val {len(va)} (AI={yva.mean():.2f})")

    # Same 5 folds for every model and every grid point, so CV scores are comparable.
    # Stratified = each fold keeps the 50/50 class balance.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    rows, grids = [], []   # rows -> summary table, grids -> every grid point (Q1.3)
    for name, (desc, pipe, grid) in build_models(args.quick).items():
        if args.only and name not in args.only:
            continue
        print(f"\n=== {name}: {desc}")

        # ------------------------------------------------ hyperparameter search
        # GridSearchCV tries every combination in `grid`, scores each by mean AUROC
        # over the 5 folds (training data only), and with refit=True refits the
        # winner on all 8,000 rows into gs.best_estimator_.
        t0 = time.time()
        gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=args.n_jobs, refit=True)
        gs.fit(Xtr, ytr)
        search_time = time.time() - t0

        # ------------------------------------------------ timed refit
        # GridSearchCV already refit the model, but its timing is mixed into the whole
        # search. Refitting once more, alone, gives a clean "train time" for the
        # report table. The result is identical (same data, same seed).
        best = gs.best_estimator_
        t0 = time.time()
        best.fit(Xtr, ytr)
        fit_time = time.time() - t0

        # ------------------------------------------------ evaluation
        # Train AUROC is reported only to show over/underfitting (a gap between train
        # and val/CV). Validation is the honest estimate.
        tr_auc = roc_auc_score(ytr, score(best, Xtr))
        va_auc = roc_auc_score(yva, score(best, Xva))
        print(f"best {gs.best_params_}  cv={gs.best_score_:.4f}  train={tr_auc:.4f}  val={va_auc:.4f}  fit={fit_time:.1f}s")

        # ------------------------------------------------ record results
        # json.dumps makes the params dict a readable string; default=str handles
        # tuples like ngram_range=(1, 2).
        rows.append(dict(model=name, representation=desc, best_params=json.dumps(gs.best_params_, default=str),
                         cv_auroc=gs.best_score_, train_auroc=tr_auc, val_auroc=va_auc,
                         fit_seconds=fit_time, search_seconds=search_time))
        # Every grid point with its CV mean and std across folds. The std shows
        # whether the "best" setting is meaningfully better than its neighbors.
        cvr = pd.DataFrame(gs.cv_results_)
        for _, r in cvr.iterrows():
            grids.append(dict(model=name, params=json.dumps(r["params"], default=str),
                              cv_mean=r["mean_test_score"], cv_std=r["std_test_score"], rank=r["rank_test_score"]))
        # Per-passage validation scores, saved so later questions (error analysis, Q4;
        # ensembling, Q5) can reuse them without retraining.
        pd.DataFrame({"id": va["id"], "label": yva, "score": score(best, Xva)}).to_csv(
            os.path.join(OUT_DIR, f"val_scores_{name}_{args.text_view}.csv"), index=False)

    # ---------------------------------------------------------------- write tables
    res = pd.DataFrame(rows).sort_values("val_auroc", ascending=False)
    res.to_csv(os.path.join(OUT_DIR, f"q1_model_comparison_{args.text_view}.csv"), index=False)
    pd.DataFrame(grids).to_csv(os.path.join(OUT_DIR, f"q1_grid_search_{args.text_view}.csv"), index=False)
    print("\n", res[["model", "cv_auroc", "train_auroc", "val_auroc", "fit_seconds"]].to_string(index=False))


if __name__ == "__main__":
    main()
