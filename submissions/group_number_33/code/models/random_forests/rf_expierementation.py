
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
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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
    default="rf_experiment_results"
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
    raise ValueError("Labels must be 0 (human), 1 (AI).")

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=args.validation_size,
    stratify=y,
    random_state=args.seed
)

y_train = y_train.to_numpy()
y_val = y_val.to_numpy()

X_train_text = X_train.tolist()
X_val_text = X_val.tolist()

np.savez(
    OUTPUT_DIR / "rf_split_indices.npz",
    train_indices=X_train.index.to_numpy(),
    val_indices=X_val.index.to_numpy()
)

print("Training:", len(X_train))
print("Validation:", len(X_val))


# ============================================================
# EXPERIMENT RESULT STORAGE
# ============================================================

def save_result(
    name,
    model_type,
    probabilities,
    model_params,
    feature_metadata
):

    predictions = (probabilities >= 0.5).astype(int)

    row = {
        "Experiment": name,
        "Model": model_type,
        "AUROC": roc_auc_score(y_val, probabilities),
        "Accuracy": accuracy_score(y_val, predictions),
        "F1": f1_score(y_val, predictions),
        "Log_Loss": log_loss(y_val, probabilities),

        # Complete configurations
        "Model_Parameters": json.dumps(model_params),
        "Feature_Configuration": json.dumps(feature_metadata),

        # Word TF-IDF
        "Word_Ngrams": str(
            feature_metadata.get("word_tfidf", {}).get("ngram_range")
        ),
        "Word_Max_Features": feature_metadata.get(
            "word_tfidf", {}
        ).get("max_features"),
        "Word_Min_DF": feature_metadata.get(
            "word_tfidf", {}
        ).get("min_df"),
        "Word_Max_DF": feature_metadata.get(
            "word_tfidf", {}
        ).get("max_df"),
        "Word_Sublinear_TF": feature_metadata.get(
            "word_tfidf", {}
        ).get("sublinear_tf"),

        # Character TF-IDF
        "Character_Ngrams": str(
            feature_metadata.get("char_tfidf", {}).get("ngram_range")
        ),
        "Character_Max_Features": feature_metadata.get(
            "char_tfidf", {}
        ).get("max_features"),
        "Character_Min_DF": feature_metadata.get(
            "char_tfidf", {}
        ).get("min_df"),
        "Character_Max_DF": feature_metadata.get(
            "char_tfidf", {}
        ).get("max_df"),
        "Character_Sublinear_TF": feature_metadata.get(
            "char_tfidf", {}
        ).get("sublinear_tf"),

        # Feature weights
        "Word_Weight": feature_metadata.get(
            "feature_weights", {}
        ).get("word"),
        "Character_Weight": feature_metadata.get(
            "feature_weights", {}
        ).get("char"),
        "MiniLM_Weight": feature_metadata.get(
            "feature_weights", {}
        ).get("minilm"),

        # MiniLM
        "MiniLM_Model": feature_metadata.get("minilm_model"),

        # SVD
        "SVD_Components": feature_metadata.get("svd_components"),
        "SVD_Explained_Variance": feature_metadata.get(
            "svd_explained_variance"
        ),

        # Classifier parameters
        "C": model_params.get("C"),
        "N_Estimators": model_params.get("n_estimators"),
        "Max_Depth": model_params.get("max_depth"),
        "Max_Features": model_params.get("max_features"),
        "Min_Samples_Leaf": model_params.get("min_samples_leaf")
    }

    RESULTS.append(row)

    print(
        f"{name[:35]:35s} "
        f"AUROC={row['AUROC']:.5f}"
    )

    return row["AUROC"]


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_lr(Xtr, Xva, C, metadata, name="Logistic Regression"):

    model = LogisticRegression(
        C=C,
        max_iter=1500,
        solver="liblinear",
        random_state=args.seed
    )

    model.fit(Xtr, y_train)

    probabilities = model.predict_proba(Xva)[:, 1]

    return save_result(
        name,
        "LogisticRegression",
        probabilities,
        {
            "C": C,
            "solver": "liblinear",
            "max_iter": 1500
        },
        metadata
    )


def evaluate_rf(Xtr, Xva, params, metadata, name):

    model = RandomForestClassifier(
        **params,
        random_state=args.seed,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(Xtr, y_train)

    probabilities = model.predict_proba(Xva)[:, 1]

    full_params = {
        **params,
        "random_state": args.seed,
        "n_jobs": -1,
        "class_weight": "balanced"
    }

    return save_result(
        name,
        "RandomForestClassifier",
        probabilities,
        full_params,
        metadata
    )


# ============================================================
# EXPERIMENT 1: WORD TF-IDF
# ============================================================

print("\n========== WORD TF-IDF ==========")

word_configs = []

for ngram in [(1, 1), (1, 2), (1, 3)]:
    for features in [10000, 20000]:
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

for i, config in enumerate(word_configs):

    vectorizer = TfidfVectorizer(**config)

    Xtr = vectorizer.fit_transform(X_train_text)
    Xva = vectorizer.transform(X_val_text)

    metadata = {
        "word_tfidf": config,
        "char_tfidf": {},
        "feature_weights": {
            "word": 1,
            "char": 0,
            "minilm": 0
        }
    }

    auc = evaluate_lr(
        Xtr, Xva, 1.0, metadata,
        f"Word TF-IDF {i+1}"
    )

    if auc > best_word_auc:
        best_word_auc = auc
        best_word_config = config

print("Best word configuration:", best_word_config)


# ============================================================
# EXPERIMENT 2: CHARACTER TF-IDF
# ============================================================

print("\n========== CHARACTER TF-IDF ==========")

char_configs = []

for ngram in [(2, 4), (3, 5), (3, 6), (4, 6)]:
    for features in [15000, 30000]:
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

for i, config in enumerate(char_configs):

    vectorizer = TfidfVectorizer(**config)

    Xtr = vectorizer.fit_transform(X_train_text)
    Xva = vectorizer.transform(X_val_text)

    metadata = {
        "word_tfidf": {},
        "char_tfidf": config,
        "feature_weights": {
            "word": 0,
            "char": 1,
            "minilm": 0
        }
    }

    auc = evaluate_lr(
        Xtr, Xva, 1.0, metadata,
        f"Character TF-IDF {i+1}"
    )

    if auc > best_char_auc:
        best_char_auc = auc
        best_char_config = config

print("Best character configuration:", best_char_config)


# ============================================================
# REBUILD BEST TF-IDF MATRICES
# ============================================================

word_vectorizer = TfidfVectorizer(**best_word_config)
char_vectorizer = TfidfVectorizer(**best_char_config)

X_train_word = word_vectorizer.fit_transform(X_train_text)
X_val_word = word_vectorizer.transform(X_val_text)

X_train_char = char_vectorizer.fit_transform(X_train_text)
X_val_char = char_vectorizer.transform(X_val_text)


# ============================================================
# EXPERIMENT 3: WORD + CHARACTER WEIGHTS
# ============================================================

print("\n========== WORD + CHARACTER ==========")

best_wc_auc = -1
best_wc_weights = None

for ww, cw in [
    (1.0, 0.25),
    (1.0, 0.5),
    (1.0, 1.0),
    (0.5, 1.0),
    (0.25, 1.0)
]:

    Xtr = hstack([
        ww * X_train_word,
        cw * X_train_char
    ], format="csr")

    Xva = hstack([
        ww * X_val_word,
        cw * X_val_char
    ], format="csr")

    metadata = {
        "word_tfidf": best_word_config,
        "char_tfidf": best_char_config,
        "feature_weights": {
            "word": ww,
            "char": cw,
            "minilm": 0
        },
        "minilm_model": None
    }

    auc = evaluate_lr(
        Xtr, Xva, 1.0, metadata,
        "Word + Character"
    )

    if auc > best_wc_auc:
        best_wc_auc = auc
        best_wc_weights = (ww, cw)
        X_train_wc = Xtr
        X_val_wc = Xva

print("Best word/character weights:", best_wc_weights)


# ============================================================
# EXPERIMENT 4: MINILM EMBEDDINGS
# ============================================================

E_train = None
E_val = None

best_embedding_weight = 0.0

X_train_final = X_train_wc
X_val_final = X_val_wc

best_feature_metadata = {
    "word_tfidf": best_word_config,
    "char_tfidf": best_char_config,
    "feature_weights": {
        "word": best_wc_weights[0],
        "char": best_wc_weights[1],
        "minilm": 0
    },
    "minilm_model": None
}

if not args.skip_minilm:

    print("\n========== MINILM ==========")

    from sentence_transformers import SentenceTransformer

    # Cache fingerprint prevents accidental reuse across different splits/data.
    digest = hashlib.sha256()
    digest.update("\n".join(X_train_text).encode())
    digest.update("\n".join(X_val_text).encode())
    digest.update(EMBEDDING_MODEL.encode())
    fingerprint = digest.hexdigest()[:12]

    train_cache = OUTPUT_DIR / f"rf_minilm_{fingerprint}_train.npy"
    val_cache = OUTPUT_DIR / f"rf_minilm_{fingerprint}_val.npy"

    if train_cache.exists() and val_cache.exists():

        E_train = np.load(train_cache)
        E_val = np.load(val_cache)

    else:

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

    # MiniLM alone
    metadata = {
        "word_tfidf": {},
        "char_tfidf": {},
        "feature_weights": {
            "word": 0,
            "char": 0,
            "minilm": 1
        },
        "minilm_model": EMBEDDING_MODEL
    }

    evaluate_lr(
        E_train, E_val, 1.0, metadata,
        "MiniLM alone"
    )

    # Test combinations
    best_emb_auc = -1
    best_emb_weight = None

    for weight in [0.1, 0.25, 0.5, 1.0, 2.0]:

        Xtr = hstack([
            X_train_wc,
            weight * E_train
        ], format="csr")

        Xva = hstack([
            X_val_wc,
            weight * E_val
        ], format="csr")

        metadata = {
            "word_tfidf": best_word_config,
            "char_tfidf": best_char_config,
            "feature_weights": {
                "word": best_wc_weights[0],
                "char": best_wc_weights[1],
                "minilm": weight
            },
            "minilm_model": EMBEDDING_MODEL
        }

        auc = evaluate_lr(
            Xtr, Xva, 1.0, metadata,
            "Word + Char + MiniLM"
        )

        if auc > best_emb_auc:
            best_emb_auc = auc
            best_emb_weight = weight
            best_emb_train = Xtr
            best_emb_val = Xva

    if best_emb_auc > best_wc_auc:

        best_embedding_weight = best_emb_weight
        X_train_final = best_emb_train
        X_val_final = best_emb_val

        best_feature_metadata = {
            "word_tfidf": best_word_config,
            "char_tfidf": best_char_config,
            "feature_weights": {
                "word": best_wc_weights[0],
                "char": best_wc_weights[1],
                "minilm": best_embedding_weight
            },
            "minilm_model": EMBEDDING_MODEL
        }

        print("Selected TF-IDF + MiniLM.")

    else:
        print("Selected word + character TF-IDF only.")


# ============================================================
# EXPERIMENT 5: LOGISTIC REGRESSION C
# Every row records TF-IDF configs and MiniLM weight.
# ============================================================

print("\n========== LOGISTIC REGRESSION C ==========")

best_lr_auc = -1
best_lr_C = None

for C in [0.001, 0.01, 0.1, 1, 10, 100]:

    auc = evaluate_lr(
        X_train_final,
        X_val_final,
        C,
        best_feature_metadata,
        f"Logistic Regression C={C}"
    )

    if auc > best_lr_auc:
        best_lr_auc = auc
        best_lr_C = C

print("Best Logistic Regression C:", best_lr_C)


# ============================================================
# EXPERIMENT 6: TRUNCATED SVD
# ============================================================

print("\n========== TRUNCATED SVD ==========")

svd_results = {}

for components in [100, 200]:

    if components >= min(
        X_train_final.shape[0],
        X_train_final.shape[1]
    ):
        continue

    print(f"\nSVD components: {components}")

    svd = TruncatedSVD(
        n_components=components,
        n_iter=7,
        random_state=args.seed
    )

    Xtr_svd = svd.fit_transform(X_train_final)
    Xva_svd = svd.transform(X_val_final)

    svd_results[components] = (
        Xtr_svd,
        Xva_svd,
        float(svd.explained_variance_ratio_.sum())
    )

    print(
        "Explained variance ratio:",
        svd.explained_variance_ratio_.sum()
    )


# ============================================================
# EXPERIMENT 7: RANDOM FOREST
# ============================================================

print("\n========== RANDOM FOREST ==========")

rf_configs = [
    {
        "n_estimators": 200,
        "max_features": "sqrt",
        "max_depth": None,
        "min_samples_leaf": 1
    },
    {
        "n_estimators": 300,
        "max_features": "sqrt",
        "max_depth": None,
        "min_samples_leaf": 2
    },
    {
        "n_estimators": 300,
        "max_features": 0.5,
        "max_depth": None,
        "min_samples_leaf": 2
    },
    {
        "n_estimators": 300,
        "max_features": "sqrt",
        "max_depth": 30,
        "min_samples_leaf": 5
    },
    {
        "n_estimators": 400,
        "max_features": 0.5,
        "max_depth": 30,
        "min_samples_leaf": 5
    }
]

best_rf_auc = -1
best_rf_config = None
best_rf_svd = None

for components, (Xtr_svd, Xva_svd, variance) in svd_results.items():

    for i, config in enumerate(rf_configs, start=1):

        metadata = {
            **best_feature_metadata,
            "svd_components": components,
            "svd_explained_variance": variance,
            "svd_random_state": args.seed
        }

        model = RandomForestClassifier(
            **config,
            random_state=args.seed,
            n_jobs=-1,
            class_weight="balanced"
        )

        model.fit(Xtr_svd, y_train)

        probabilities = model.predict_proba(Xva_svd)[:, 1]

        predictions = (probabilities >= 0.5).astype(int)

        auc = roc_auc_score(y_val, probabilities)

        row = {
            "Experiment": f"Random Forest SVD={components} config={i}",
            "Model": "RandomForestClassifier",
            "AUROC": auc,
            "Accuracy": accuracy_score(y_val, predictions),
            "F1": f1_score(y_val, predictions),
            "Log_Loss": log_loss(y_val, probabilities),
            "Model_Parameters": json.dumps({
                **config,
                "random_state": args.seed,
                "class_weight": "balanced"
            }),
            "Feature_Configuration": json.dumps(metadata),

            # Explicit feature metadata
            "Word_Ngrams": str(best_word_config["ngram_range"]),
            "Word_Max_Features": best_word_config["max_features"],
            "Word_Min_DF": best_word_config["min_df"],
            "Word_Max_DF": best_word_config["max_df"],
            "Word_Sublinear_TF": best_word_config["sublinear_tf"],

            "Character_Ngrams": str(best_char_config["ngram_range"]),
            "Character_Max_Features": best_char_config["max_features"],
            "Character_Min_DF": best_char_config["min_df"],
            "Character_Max_DF": best_char_config["max_df"],
            "Character_Sublinear_TF": best_char_config["sublinear_tf"],

            "Word_Weight": best_wc_weights[0],
            "Character_Weight": best_wc_weights[1],
            "MiniLM_Weight": best_embedding_weight,
            "MiniLM_Model": (
                EMBEDDING_MODEL if best_embedding_weight > 0 else None
            ),

            "SVD_Components": components,
            "SVD_Explained_Variance": variance,

            "C": None,
            "N_Estimators": config["n_estimators"],
            "Max_Depth": config["max_depth"],
            "Max_Features": config["max_features"],
            "Min_Samples_Leaf": config["min_samples_leaf"]
        }

        RESULTS.append(row)

        print(
            f"RF SVD={components}, config={i} | "
            f"AUROC={auc:.5f}"
        )

        if auc > best_rf_auc:
            best_rf_auc = auc
            best_rf_config = config
            best_rf_svd = components


# ============================================================
# SAVE RESULTS
# ============================================================

print("\n========== SAVING RESULTS ==========")

results_df = pd.DataFrame(RESULTS)

results_df = results_df.sort_values(
    "AUROC",
    ascending=False
)

results_file = OUTPUT_DIR / "rf_experiment_results.csv"

results_df.to_csv(results_file, index=False)

best_config = {
    "best_word_tfidf": best_word_config,
    "best_character_tfidf": best_char_config,
    "best_word_character_weights": {
        "word": best_wc_weights[0],
        "character": best_wc_weights[1]
    },
    "selected_feature_metadata": best_feature_metadata,
    "best_logistic_regression_C": best_lr_C,
    "best_logistic_regression_AUROC": best_lr_auc,
    "best_random_forest": {
        "parameters": best_rf_config,
        "svd_components": best_rf_svd,
        "AUROC": best_rf_auc
    }
}

with open(
    OUTPUT_DIR / "rf_best_configurations.json",
    "w"
) as f:
    json.dump(best_config, f, indent=2)

print("\nTOP RESULTS:")
print(
    results_df[
        [
            "Experiment",
            "Model",
            "AUROC",
            "Accuracy",
            "F1",
            "C",
            "N_Estimators",
            "SVD_Components",
            "MiniLM_Weight"
        ]
    ].head(20).to_string(index=False)
)

print("\nResults saved:", results_file)
print("Best configurations saved: rf_best_configurations.json")
print("\nAll experiments completed.")
