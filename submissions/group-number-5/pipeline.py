"""Shared helpers for the Human-or-AI pipeline.

Pipeline: text -> EmbeddingGemma-2:270m embedding (Ollama) -> SVC(kernel="poly", C=10).

Embeddings are cached as ``<csv-stem>_embeddinggemma2_270m.npy`` next to the CSV
so repeated training/evaluation runs do not re-embed the same text. The cache is
keyed by row order, so pass the same CSV file the cache was built from (or
``--no-cache`` to disable reading it).
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42
EMBED_MODEL = "embeddinggemma-2:270m"
DEFAULT_KERNEL = "poly"
DEFAULT_C = 10.0
DEFAULT_BATCH_SIZE = 32

# The course repo runs scikit-learn 1.9, which deprecates SVC(probability=True)
# (removal planned for 1.11). The notebook and the committed submission.csv were
# produced with that estimator, so keep it and silence only this warning.
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

#: File-name fragment used for embedding caches, e.g. ``embeddinggemma2_270m``.
EMBED_SLUG = EMBED_MODEL.replace("-", "").replace(":", "_")


def embed_texts(texts, model: str = EMBED_MODEL, batch_size: int = DEFAULT_BATCH_SIZE):
    """Embed ``texts`` with Ollama in small batches.

    Returns a ``(len(texts), dim)`` float32 array. Requires a running Ollama
    server with ``model`` pulled (``ollama pull embeddinggemma-2:270m``).
    """
    import ollama

    texts = [str(t) for t in texts]
    vectors: list[list[float]] = []
    start_time = time.time()

    for start in range(0, len(texts), batch_size):
        response = ollama.embed(
            model=model, input=texts[start : start + batch_size]
        )
        vectors.extend(response.embeddings)
        done = min(start + batch_size, len(texts))
        if done == len(texts) or done % (batch_size * 25) == 0:
            elapsed = time.time() - start_time
            print(f"  embedded {done}/{len(texts)} texts ({elapsed:.1f}s)")

    return np.asarray(vectors, dtype=np.float32)


def default_cache_path(csv_path) -> Path:
    """Cache location next to the CSV, e.g. ``data/train_embeddinggemma2_270m.npy``."""
    csv_path = Path(csv_path)
    return csv_path.with_name(f"{csv_path.stem}_{EMBED_SLUG}.npy")


def load_embeddings(
    csv_path,
    cache_path=None,
    text_col: str = "text",
    embed_model: str = EMBED_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    write_cache: bool = True,
    use_cache: bool = True,
):
    """Return the embedding matrix for the text column of ``csv_path``.

    Reads ``cache_path`` when it exists and matches the CSV row count, otherwise
    embeds the texts and (optionally) writes the cache back to disk.
    """
    csv_path = Path(csv_path)
    cache_path = Path(cache_path) if cache_path is not None else default_cache_path(csv_path)

    texts = pd.read_csv(csv_path, usecols=[text_col])[text_col]

    if use_cache and cache_path.exists():
        cached = np.load(cache_path)
        if len(cached) == len(texts):
            print(f"Loaded cached embeddings {cache_path} {cached.shape}")
            return cached
        print(
            f"Ignoring stale cache {cache_path}: has {len(cached)} rows, "
            f"{csv_path} has {len(texts)}"
        )

    print(f"Embedding {len(texts)} texts from {csv_path} with {embed_model}")
    embeddings = embed_texts(texts, model=embed_model, batch_size=batch_size)

    if write_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        print(f"Wrote embedding cache {cache_path}")

    return embeddings


def build_model(C: float = DEFAULT_C, kernel: str = DEFAULT_KERNEL, seed: int = RANDOM_SEED):
    """The team's classifier: Platt-calibrated polynomial-kernel SVC.

    ``probability=True`` is deprecated in scikit-learn 1.9; it is kept because it
    reproduces the committed submission.csv exactly. On scikit-learn >= 1.11 the
    equivalent estimator is ``CalibratedClassifierCV(SVC(...), ensemble=False)``.
    """
    from sklearn.svm import SVC

    return SVC(kernel=kernel, C=C, probability=True, random_state=seed)


def predict_scores(model, X) -> np.ndarray:
    """P(AI-generated) for each row of ``X``."""
    return model.predict_proba(X)[:, 1]
