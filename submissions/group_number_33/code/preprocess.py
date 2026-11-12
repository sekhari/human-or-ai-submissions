import numpy as np
import pandas as pd
import re
from pathlib import Path
from scipy.sparse import hstack
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from models.text_generalization.style_search import StyleFeatures


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def split_validation_set(file, random_state=12):
    df = pd.read_csv(file)
    # Hold out the same proportion of each label for validation.
    df_train, df_validation = train_test_split(
        df, test_size=0.2, random_state=random_state, stratify=df["label"]
    )

    print(f"Train set size: {len(df_train)}")
    print(f"Validation set size: {len(df_validation)}")
    return df_train, df_validation

def unlabel_data(df):
    X = df["text"]
    y = df['label']
    return X, y

def clean_text(text, lowercase=True):
    # Lowercase by default; optionally preserve capitalization.
    if lowercase:
        text = text.lower()

    #remove urls
    text = re.sub(r'https?://\S+|www\.\S+', '', text, flags=re.IGNORECASE)

    # Remove Unix-style paths
    text = re.sub(r'(/[^\s]+)+', ' FILEPATH ', text)

    # Remove Windows-style paths
    text = re.sub(r'[A-Za-z]:\\[^\s]+', ' FILEPATH ', text)

    # Remove long numbers
    text = re.sub(r'\b\d{5,}\b', '', text)

    # Replace UUIDs
    text = re.sub(
        r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-'
        r'[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-'
        r'[0-9a-fA-F]{12}\b',
        'UUID',
        text
    )

    # Remove hashes
    text = re.sub(r'\b[a-fA-F0-9]{8,}\b', '', text)

    #normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tfidf_data_char(X_train, X_validation, X_test):
    #tf-idf
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        max_features=20000
    )

    X_train_clean = X_train.apply(clean_text)
    X_train_tfidf = vectorizer.fit_transform(X_train_clean)

    X_validation_clean = X_validation.apply(clean_text)
    X_validation_tfidf = vectorizer.transform(X_validation_clean)

    X_test_clean = X_test.apply(clean_text)
    X_test_tfidf = vectorizer.transform(X_test_clean)

    return X_train_tfidf, X_validation_tfidf, X_test_tfidf

def tfidf_data(X_train, X_validation, X_test):
    #tf-idf
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    X_train_clean = X_train.apply(clean_text)
    X_train_tfidf = vectorizer.fit_transform(X_train_clean)

    X_validation_clean = X_validation.apply(clean_text)
    X_validation_tfidf = vectorizer.transform(X_validation_clean)

    X_test_clean = X_test.apply(clean_text)
    X_test_tfidf = vectorizer.transform(X_test_clean)

    return X_train_tfidf, X_validation_tfidf, X_test_tfidf


def tfidf_data_combo(X_train, X_validation, X_test):

    # Word-level TF-IDF
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000
    )

    # Character-level TF-IDF
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        max_features=20000
    )

    # Fit on training data
    X_train_word = word_vectorizer.fit_transform(X_train)
    X_train_char = char_vectorizer.fit_transform(X_train)

    # Transform validation and test data
    X_validation_word = word_vectorizer.transform(X_validation)
    X_validation_char = char_vectorizer.transform(X_validation)

    X_test_word = word_vectorizer.transform(X_test)
    X_test_char = char_vectorizer.transform(X_test)

    # Combine word and character features
    X_train_combined = hstack([X_train_word, X_train_char])
    X_validation_combined = hstack([X_validation_word, X_validation_char])
    X_test_combined = hstack([X_test_word, X_test_char])

    return X_train_combined, X_validation_combined, X_test_combined


def word2vec(X_train, X_validation, X_test):

    tokenized_train = [re.findall(r"\b\w+\b", text.lower()) for text in X_train]

    model = Word2Vec(
        sentences=tokenized_train,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        seed=12
    )

    def document_vector(text):
        words = re.findall(r"\b\w+\b", text.lower())

        vectors = [
            model.wv[word]
            for word in words
            if word in model.wv
        ]

        if len(vectors) == 0:
            return np.zeros(100)

        return np.mean(vectors, axis=0)

    X_train = np.array([document_vector(text) for text in X_train])
    X_validation = np.array([document_vector(text) for text in X_validation])
    X_test = np.array([document_vector(text) for text in X_test])

    return X_train, X_validation, X_test


def sentence_embeddings(X_train, X_validation, X_test):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    X_train = model.encode(X_train.tolist(), show_progress_bar=True)
    X_validation = model.encode(X_validation.tolist(), show_progress_bar=True)
    X_test = model.encode(X_test.tolist(), show_progress_bar=True)

    return X_train, X_validation, X_test

def dave_145_stylization(X_train, X_validation, X_test):
    # Implement the specific stylization logic here

    vectorizer = StyleFeatures()

    X_train_145_style = vectorizer.fit_transform(X_train)

    X_validation_145_style = vectorizer.transform(X_validation)

    X_test_145_style = vectorizer.transform(X_test)

    return X_train_145_style, X_validation_145_style, X_test_145_style


def dave_145_stylization_full_data(X_train, X_test):
    # Implement the specific stylization logic here

    vectorizer = StyleFeatures()

    X_train_145_style = vectorizer.fit_transform(X_train)


    X_test_145_style = vectorizer.transform(X_test)

    return X_train_145_style, X_test_145_style



def preprocess_data(featurization=tfidf_data, n=8000, random_state=12):
    
    df_train, df_validation = split_validation_set(_data_file("train.csv"), random_state=random_state)
    
    if n != len(df_train):
        df_train = df_train.sample(n=n, random_state=random_state)

    X_train, y_train = unlabel_data(df_train)
    X_validation, y_validation = unlabel_data(df_validation)

    X_test = pd.read_csv(_data_file("test.csv"))["text"]
    X_train, X_validation, X_test = featurization(X_train, X_validation, X_test)
    return X_train, y_train, X_validation, y_validation, X_test

def process_full_data(featurization=dave_145_stylization_full_data, random_state=12):
    df_train = pd.read_csv(_data_file("train.csv"))
    X_train, y_train = unlabel_data(df_train)

    X_test = pd.read_csv(_data_file("test.csv"))["text"]
    X_train, X_test = featurization(X_train, X_test)
    return X_train, y_train, X_test

def _data_file(name):
    """Use the same repository data files regardless of working directory."""
    return Path(__file__).resolve().parent / "data" / name


def export_split(file, output_dir, random_state=12):
    """Write the raw shared split without changing the source CSV."""
    output_dir = Path(output_dir)
    train_file = output_dir / "train_split.csv"
    validation_file = output_dir / "validation_split.csv"
    if Path(file).resolve() in {train_file.resolve(), validation_file.resolve()}:
        raise ValueError("The source CSV must not be one of the output files")
    df_train, df_validation = split_validation_set(file, random_state=random_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(train_file, index=False)
    df_validation.to_csv(validation_file, index=False)
    return train_file, validation_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export the shared stratified split")
    parser.add_argument("--export-split", action="store_true", help="write the raw train and validation CSVs")
    parser.add_argument("--input", default=str(Path(__file__).resolve().parent / "data" / "train.csv"))
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "data" / "splits"))
    parser.add_argument("--random-state", type=int, default=12)
    args = parser.parse_args()
    if args.export_split:
        for path in export_split(args.input, args.output_dir, args.random_state):
            print(f"Saved: {path}")
    else:
        parser.print_help()
