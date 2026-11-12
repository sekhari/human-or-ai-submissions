import sys
from pathlib import Path
from time import perf_counter
from time import perf_counter

from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from scipy.sparse import hstack, csr_matrix

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from sklearn.feature_extraction.text import TfidfVectorizer

from preprocess import dave_145_stylization, preprocess_data, tfidf_data, tfidf_data_char, tfidf_data_combo, word2vec, sentence_embeddings
from scorer import create_kaggle_submission, evaluate_model


def tfidf_combo_minlm(X_train, X_validation, X_test):

    # ============================================================
    # WORD TF-IDF
    # ============================================================

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        max_features=40000,
        min_df=3,
        max_df=0.95,
        sublinear_tf=True
    )

    X_train_word = word_vectorizer.fit_transform(X_train)
    X_validation_word = word_vectorizer.transform(X_validation)
    X_test_word = word_vectorizer.transform(X_test)


    # ============================================================
    # CHARACTER TF-IDF
    # ============================================================

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=40000,
        min_df=3,
        max_df=0.95,
        sublinear_tf=True
    )

    X_train_char = char_vectorizer.fit_transform(X_train)
    X_validation_char = char_vectorizer.transform(X_validation)
    X_test_char = char_vectorizer.transform(X_test)


    # ============================================================
    # MINILM EMBEDDINGS
    # ============================================================

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    X_train_embedding = embedding_model.encode(
        X_train.tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32
    )

    X_validation_embedding = embedding_model.encode(
        X_validation.tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32
    )

    X_test_embedding = embedding_model.encode(
        X_test.tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32
    )

    # Convert embeddings to sparse matrices
    X_train_embedding = csr_matrix(X_train_embedding)
    X_validation_embedding = csr_matrix(X_validation_embedding)
    X_test_embedding = csr_matrix(X_test_embedding)


    # ============================================================
    # COMBINE FEATURES
    # ============================================================

    WORD_WEIGHT = 1.0
    CHAR_WEIGHT = 1.0
    MINILM_WEIGHT = 0.5

    X_train_final = hstack([
        WORD_WEIGHT * X_train_word,
        CHAR_WEIGHT * X_train_char,
        MINILM_WEIGHT * X_train_embedding
    ], format="csr")

    X_validation_final = hstack([
        WORD_WEIGHT * X_validation_word,
        CHAR_WEIGHT * X_validation_char,
        MINILM_WEIGHT * X_validation_embedding
    ], format="csr")

    X_test_final = hstack([
        WORD_WEIGHT * X_test_word,
        CHAR_WEIGHT * X_test_char,
        MINILM_WEIGHT * X_test_embedding
    ], format="csr")

    return X_train_final, X_validation_final, X_test_final

def logistic_regression(X_train, y_train):
    print("\nL2 Logistic Regression")

    model_l2 = LogisticRegression(
            C=10,
            solver="liblinear",
            max_iter=1500,
        )
    model_l2.fit(X_train, y_train)

    return model_l2

X_train, y_train, X_validation, y_validation, X_test = preprocess_data(featurization=dave_145_stylization)
start = perf_counter()
model = logistic_regression(X_train, y_train)
print(f"Model training: {perf_counter() - start:.2f} seconds")

evaluate_model(model, X_validation, y_validation)
# create_kaggle_submission(model, X_test, "submission_logistic_regression_tfidfcombo_minlm.csv")