
import os
import json
import hashlib
import warnings
import argparse

import numpy as np
import pandas as pd

from pathlib import Path
from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    log_loss
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument("--train", default="data/train.csv")
parser.add_argument("--text-col", default="text")
parser.add_argument("--label-col", default="label")
parser.add_argument("--validation-size", type=int, default=2000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--skip-minilm", action="store_true")
parser.add_argument(
    "--output-dir",
    default="experiment_results"
)

args = parser.parse_args()

OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

RESULTS = []

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading data...")

df = pd.read_csv(args.train)

X = df[args.text_col].fillna("").astype(str)
y = df[args.label_col].astype(int)

if set(y.unique()) != {0, 1}:
    raise ValueError("Labels must be 0 (human) and 1 (AI).")

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=args.validation_size,
    stratify=y,
    random_state=args.seed
)

X_train_text = X_train.tolist()
X_val_text = X_val.tolist()

y_train = y_train.to_numpy()
y_val = y_val.to_numpy()

np.savez(
    OUTPUT_DIR / "lr_split_indices.npz",
    train_indices=X_train.index.to_numpy(),
    val_indices=X_val.index.to_numpy()
)

print("Training examples:", len(X_train))
print("Validation examples:", len(X_val))


# ============================================================
# RESULT SAVING
# ============================================================

def save_result(
    experiment_name,
    model_name,
    probabilities,
    model_params,
    feature_config
):
    """
    Save model performance and the complete feature configuration.

    Every experiment gets its own copy of the configuration
    used to construct its feature matrix.
    """

    predictions = (probabilities >= 0.5).astype(int)

    word = feature_config.get("word_tfidf", {})
    char = feature_config.get("char_tfidf", {})
    weights = feature_config.get("feature_weights", {})

    row = {
        # Experiment identification
        "Experiment": experiment_name,
        "Model": model_name,

        # Performance metrics
        "AUROC": roc_auc_score(y_val, probabilities),
        "Accuracy": accuracy_score(y_val, predictions),
        "F1": f1_score(y_val, predictions),
        "Log_Loss": log_loss(y_val, probabilities),

        # Full configurations stored as JSON
        "Model_Parameters": json.dumps(
            model_params,
            default=str
        ),

        "Feature_Configuration": json.dumps(
            feature_config,
            default=str
        ),

        # Word TF-IDF
        "Word_Analyzer": word.get("analyzer"),
        "Word_Ngrams": str(word.get("ngram_range")),
        "Word_Max_Features": word.get("max_features"),
        "Word_Min_DF": word.get("min_df"),
        "Word_Max_DF": word.get("max_df"),
        "Word_Sublinear_TF": word.get("sublinear_tf"),

        # Character TF-IDF
        "Character_Analyzer": char.get("analyzer"),
        "Character_Ngrams": str(char.get("ngram_range")),
        "Character_Max_Features": char.get("max_features"),
        "Character_Min_DF": char.get("min_df"),
        "Character_Max_DF": char.get("max_df"),
        "Character_Sublinear_TF": char.get("sublinear_tf"),

        # Feature weights
        "Word_Weight": weights.get("word"),
        "Character_Weight": weights.get("char"),
        "MiniLM_Weight": weights.get("minilm"),

        # Embedding model
        "MiniLM_Model": feature_config.get("minilm_model"),

        # Logistic Regression parameters
        "C": model_params.get("C"),
        "Solver": model_params.get("solver"),
        "Max_Iterations": model_params.get("max_iter"),

        # Additional model parameters
        "NB_SVM": model_params.get("nb_svm"),
        "NB_SVM_Alpha": model_params.get("nb_svm_alpha")
    }

    RESULTS.append(row)

    print(
        f"{experiment_name}: "
        f"AUROC={row['AUROC']:.5f}"
    )

    return row["AUROC"]


# ============================================================
# LOGISTIC REGRESSION EVALUATION
# ============================================================

def evaluate_lr(
    Xtr,
    Xva,
    C,
    feature_config,
    experiment_name
):

    model = LogisticRegression(
        C=C,
        max_iter=1500,
        solver="liblinear",
        random_state=args.seed
    )

    model.fit(Xtr, y_train)

    probabilities = model.predict_proba(Xva)[:, 1]

    return save_result(
        experiment_name=experiment_name,
        model_name="LogisticRegression",
        probabilities=probabilities,
        model_params={
            "C": C,
            "solver": "liblinear",
            "max_iter": 1500,
            "random_state": args.seed
        },
        feature_config=feature_config
    )


# ============================================================
# EXPERIMENT 1: WORD TF-IDF
# ============================================================

print("\n========== WORD TF-IDF EXPERIMENTS ==========")

word_configs = []

for ngram in [(1, 1), (1, 2), (1, 3)]:
    for features in [10000, 20000, 40000]:
        for min_df in [2, 3]:
            for sublinear in [True, False]:

                word_configs.append({
                    "analyzer": "word",
                    "ngram_range": ngram,
                    "max_features": features,
                    "min_df": min_df,
                    "max_df": 0.95,
                    "sublinear_tf": sublinear
                })


best_word_auc = -1
best_word_config = None

for i, config in enumerate(word_configs, start=1):

    print(f"\nWord configuration {i}/{len(word_configs)}")

    vectorizer = TfidfVectorizer(**config)

    Xtr = vectorizer.fit_transform(X_train_text)
    Xva = vectorizer.transform(X_val_text)

    feature_config = {
        "word_tfidf": config,
        "char_tfidf": {},
        "feature_weights": {
            "word": 1.0,
            "char": 0.0,
            "minilm": 0.0
        },
        "minilm_model": None
    }

    auc = evaluate_lr(
        Xtr,
        Xva,
        C=1.0,
        feature_config=feature_config,
        experiment_name=f"Word TF-IDF {i}"
    )

    if auc > best_word_auc:
        best_word_auc = auc
        best_word_config = config


print("\nBest word TF-IDF configuration:")
print(best_word_config)
print("AUROC:", best_word_auc)


# ============================================================
# EXPERIMENT 2: CHARACTER TF-IDF
# ============================================================

print("\n========== CHARACTER TF-IDF EXPERIMENTS ==========")

char_configs = []

for ngram in [(2, 4), (3, 5), (3, 6), (4, 6)]:
    for features in [20000, 40000]:
        for min_df in [2, 3]:
            for sublinear in [True, False]:

                char_configs.append({
                    "analyzer": "char_wb",
                    "ngram_range": ngram,
                    "max_features": features,
                    "min_df": min_df,
                    "max_df": 0.95,
                    "sublinear_tf": sublinear
                })


best_char_auc = -1
best_char_config = None

for i, config in enumerate(char_configs, start=1):

    print(f"\nCharacter configuration {i}/{len(char_configs)}")

    vectorizer = TfidfVectorizer(**config)

    Xtr = vectorizer.fit_transform(X_train_text)
    Xva = vectorizer.transform(X_val_text)

    feature_config = {
        "word_tfidf": {},
        "char_tfidf": config,
        "feature_weights": {
            "word": 0.0,
            "char": 1.0,
            "minilm": 0.0
        },
        "minilm_model": None
    }

    auc = evaluate_lr(
        Xtr,
        Xva,
        C=1.0,
        feature_config=feature_config,
        experiment_name=f"Character TF-IDF {i}"
    )

    if auc > best_char_auc:
        best_char_auc = auc
        best_char_config = config


print("\nBest character TF-IDF configuration:")
print(best_char_config)
print("AUROC:", best_char_auc)


# ============================================================
# REBUILD BEST WORD AND CHARACTER MATRICES
# ============================================================

print("\nRebuilding best TF-IDF matrices...")

word_vectorizer = TfidfVectorizer(**best_word_config)
char_vectorizer = TfidfVectorizer(**best_char_config)

X_train_word = word_vectorizer.fit_transform(X_train_text)
X_val_word = word_vectorizer.transform(X_val_text)

X_train_char = char_vectorizer.fit_transform(X_train_text)
X_val_char = char_vectorizer.transform(X_val_text)


# ============================================================
# EXPERIMENT 3: WORD + CHARACTER TF-IDF
# ============================================================

print("\n========== WORD + CHARACTER EXPERIMENTS ==========")

weight_combinations = [
    (1.0, 0.25),
    (1.0, 0.5),
    (1.0, 1.0),
    (0.5, 1.0),
    (0.25, 1.0)
]

best_wc_auc = -1
best_wc_weights = None

for word_weight, char_weight in weight_combinations:

    print(
        f"\nTesting word weight={word_weight}, "
        f"character weight={char_weight}"
    )

    Xtr = hstack([
        word_weight * X_train_word,
        char_weight * X_train_char
    ], format="csr")

    Xva = hstack([
        word_weight * X_val_word,
        char_weight * X_val_char
    ], format="csr")

    feature_config = {
        "word_tfidf": best_word_config,
        "char_tfidf": best_char_config,
        "feature_weights": {
            "word": word_weight,
            "char": char_weight,
            "minilm": 0.0
        },
        "minilm_model": None
    }

    auc = evaluate_lr(
        Xtr,
        Xva,
        C=1.0,
        feature_config=feature_config,
        experiment_name=(
            f"Word + Character "
            f"W={word_weight} C={char_weight}"
        )
    )

    if auc > best_wc_auc:
        best_wc_auc = auc
        best_wc_weights = (word_weight, char_weight)

        X_train_wc = Xtr
        X_val_wc = Xva


print("\nBest word/character weights:", best_wc_weights)
print("Best combined AUROC:", best_wc_auc)


# ============================================================
# INITIALIZE SELECTED FEATURE REPRESENTATION
# ============================================================

X_train_final = X_train_wc
X_val_final = X_val_wc

best_minilm_weight = 0.0

best_feature_config = {
    "word_tfidf": best_word_config,
    "char_tfidf": best_char_config,
    "feature_weights": {
        "word": best_wc_weights[0],
        "char": best_wc_weights[1],
        "minilm": 0.0
    },
    "minilm_model": None
}


# ============================================================
# EXPERIMENT 4: MINILM EMBEDDINGS
# ============================================================

if not args.skip_minilm:

    print("\n========== MINILM EXPERIMENTS ==========")

    from sentence_transformers import SentenceTransformer

    # Fingerprint the data split and model to avoid stale caches.
    digest = hashlib.sha256()

    digest.update("\n".join(X_train_text).encode())
    digest.update("\n".join(X_val_text).encode())
    digest.update(EMBEDDING_MODEL.encode())

    fingerprint = digest.hexdigest()[:12]

    train_cache = (
        OUTPUT_DIR / f"lr_minilm_{fingerprint}_train.npy"
    )

    val_cache = (
        OUTPUT_DIR / f"lr_minilm_{fingerprint}_val.npy"
    )

    if train_cache.exists() and val_cache.exists():

        print("Loading cached MiniLM embeddings...")

        E_train = np.load(train_cache)
        E_val = np.load(val_cache)

    else:

        print("Generating MiniLM embeddings...")

        embedder = SentenceTransformer(EMBEDDING_MODEL)

        E_train = embedder.encode(
            X_train_text,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32
        )

        E_val = embedder.encode(
            X_val_text,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32
        )

        np.save(train_cache, E_train)
        np.save(val_cache, E_val)

    E_train = csr_matrix(E_train)
    E_val = csr_matrix(E_val)

    # --------------------------------------------------------
    # MiniLM by itself
    # --------------------------------------------------------

    minilm_only_config = {
        "word_tfidf": {},
        "char_tfidf": {},
        "feature_weights": {
            "word": 0.0,
            "char": 0.0,
            "minilm": 1.0
        },
        "minilm_model": EMBEDDING_MODEL
    }

    evaluate_lr(
        E_train,
        E_val,
        C=1.0,
        feature_config=minilm_only_config,
        experiment_name="MiniLM Alone"
    )

    # --------------------------------------------------------
    # Word + Character + MiniLM
    # --------------------------------------------------------

    best_embedding_auc = -1
    best_embedding_weight = None

    for embedding_weight in [0.1, 0.25, 0.5, 1.0, 2.0]:

        print(
            f"\nTesting MiniLM weight={embedding_weight}"
        )

        Xtr = hstack([
            X_train_wc,
            embedding_weight * E_train
        ], format="csr")

        Xva = hstack([
            X_val_wc,
            embedding_weight * E_val
        ], format="csr")

        feature_config = {
            "word_tfidf": best_word_config,
            "char_tfidf": best_char_config,
            "feature_weights": {
                "word": best_wc_weights[0],
                "char": best_wc_weights[1],
                "minilm": embedding_weight
            },
            "minilm_model": EMBEDDING_MODEL
        }

        auc = evaluate_lr(
            Xtr,
            Xva,
            C=1.0,
            feature_config=feature_config,
            experiment_name=(
                f"Word + Character + MiniLM "
                f"Weight={embedding_weight}"
            )
        )

        if auc > best_embedding_auc:
            best_embedding_auc = auc
            best_embedding_weight = embedding_weight

            best_embedding_train = Xtr
            best_embedding_val = Xva

    # Only use MiniLM in the final representation if it improves
    # on the best word + character representation.

    if best_embedding_auc > best_wc_auc:

        print("\nMiniLM improves validation AUROC.")

        X_train_final = best_embedding_train
        X_val_final = best_embedding_val

        best_minilm_weight = best_embedding_weight

        best_feature_config = {
            "word_tfidf": best_word_config,
            "char_tfidf": best_char_config,
            "feature_weights": {
                "word": best_wc_weights[0],
                "char": best_wc_weights[1],
                "minilm": best_minilm_weight
            },
            "minilm_model": EMBEDDING_MODEL
        }

    else:

        print("\nMiniLM did not improve validation AUROC.")
        print("Using word + character TF-IDF only.")


# ============================================================
# EXPERIMENT 5: LOGISTIC REGRESSION C TUNING
# ============================================================

print("\n========== LOGISTIC REGRESSION C TUNING ==========")

best_c = None
best_c_auc = -1

C_values = [
    0.001,
    0.01,
    0.1,
    1,
    10,
    100
]

for C in C_values:

    print(f"\nTesting Logistic Regression C={C}")

    model = LogisticRegression(
        C=C,
        max_iter=1500,
        solver="liblinear",
        random_state=args.seed
    )

    model.fit(X_train_final, y_train)

    probabilities = model.predict_proba(X_val_final)[:, 1]

    # IMPORTANT:
    # This is the exact feature configuration used to build
    # X_train_final and X_val_final.
    #
    # It is saved separately for every C value.

    auc = save_result(
        experiment_name=f"Logistic Regression C={C}",
        model_name="LogisticRegression",
        probabilities=probabilities,
        model_params={
            "C": C,
            "solver": "liblinear",
            "max_iter": 1500,
            "random_state": args.seed
        },
        feature_config=best_feature_config
    )

    if auc > best_c_auc:

        best_c_auc = auc
        best_c = C


print("\nBest Logistic Regression C:", best_c)
print("Best C AUROC:", best_c_auc)


# ============================================================
# EXPERIMENT 6: OPTIONAL NB-SVM
# Uses sparse word + character TF-IDF features.
# ============================================================

print("\n========== NB-SVM EXPERIMENT ==========")

def fit_nb_svm(
    Xtr,
    ytr,
    Xva,
    alpha=1.0,
    C=1.0
):
    """
    NB-SVM-style feature reweighting.

    Computes a log-count ratio from binary feature occurrences,
    transforms the sparse feature matrix, and fits Logistic
    Regression.
    """

    X_binary = Xtr.copy()
    X_binary.data = np.ones_like(X_binary.data)

    positive = X_binary[ytr == 1].sum(axis=0).A1
    negative = X_binary[ytr == 0].sum(axis=0).A1

    positive_total = positive.sum()
    negative_total = negative.sum()

    r = np.log(
        (positive + alpha) /
        (positive_total + alpha * Xtr.shape[1])
    ) - np.log(
        (negative + alpha) /
        (negative_total + alpha * Xtr.shape[1])
    )

    Xtr_weighted = Xtr.multiply(r).tocsr()
    Xva_weighted = Xva.multiply(r).tocsr()

    model = LogisticRegression(
        C=C,
        max_iter=1500,
        solver="liblinear",
        random_state=args.seed
    )

    model.fit(Xtr_weighted, ytr)

    probabilities = model.predict_proba(Xva_weighted)[:, 1]

    return probabilities


# NB-SVM is evaluated on word + character TF-IDF, not MiniLM.
# This ensures the feature configuration is explicit.

for alpha in [0.1, 1.0, 10.0]:

    probabilities = fit_nb_svm(
        X_train_wc,
        y_train,
        X_val_wc,
        alpha=alpha,
        C=1.0
    )

    nb_svm_config = {
        "word_tfidf": best_word_config,
        "char_tfidf": best_char_config,
        "feature_weights": {
            "word": best_wc_weights[0],
            "char": best_wc_weights[1],
            "minilm": 0.0
        },
        "minilm_model": None
    }

    save_result(
        experiment_name=f"NB-SVM alpha={alpha}",
        model_name="NB-SVM LogisticRegression",
        probabilities=probabilities,
        model_params={
            "C": 1.0,
            "solver": "liblinear",
            "max_iter": 1500,
            "nb_svm": True,
            "nb_svm_alpha": alpha,
            "random_state": args.seed
        },
        feature_config=nb_svm_config
    )


# ============================================================
# SAVE ALL EXPERIMENT RESULTS
# ============================================================

print("\n========== SAVING RESULTS ==========")

results_df = pd.DataFrame(RESULTS)

results_df = results_df.sort_values(
    by="AUROC",
    ascending=False
)

results_file = OUTPUT_DIR / "experiment_results.csv"

results_df.to_csv(
    results_file,
    index=False
)


# ============================================================
# SAVE BEST CONFIGURATIONS
# ============================================================

best_configurations = {
    "best_word_tfidf": best_word_config,
    "best_word_tfidf_auroc": best_word_auc,

    "best_character_tfidf": best_char_config,
    "best_character_tfidf_auroc": best_char_auc,

    "best_word_character_weights": {
        "word": best_wc_weights[0],
        "character": best_wc_weights[1]
    },
    "best_word_character_auroc": best_wc_auc,

    "selected_feature_configuration": best_feature_config,

    "best_logistic_regression_C": best_c,
    "best_logistic_regression_auroc": best_c_auc
}

with open(
    OUTPUT_DIR / "best_configurations.json",
    "w"
) as f:
    json.dump(
        best_configurations,
        f,
        indent=2,
        default=str
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n========== TOP EXPERIMENTS ==========")

display_columns = [
    "Experiment",
    "Model",
    "AUROC",
    "Accuracy",
    "F1",
    "C",
    "Word_Ngrams",
    "Character_Ngrams",
    "Word_Weight",
    "Character_Weight",
    "MiniLM_Weight"
]

print(
    results_df[display_columns]
    .head(25)
    .to_string(index=False)
)

print("\nResults saved to:", results_file)

print(
    "Best configurations saved to:",
    OUTPUT_DIR / "best_configurations.json"
)

print(
    "Validation split saved to:",
    OUTPUT_DIR / "lr_split_indices.npz"
)

print("\nAll experiments completed.")
