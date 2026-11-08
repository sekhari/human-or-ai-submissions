"""Cache pretrained-model features for all splits under cache/.

GPT-2 features are saved in parts, shortest documents first, so progress is visible and an
interrupted run resumes from the last finished part. Finished outputs are skipped.

Usage:
    python -m experiments.extract_neural
"""
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_all
from src.neural import curvature_features, language_model_features, sentence_embeddings, surprisal_features

CACHE = Path(__file__).resolve().parents[1] / "cache"
PART_SIZE = 500
LANGUAGE_MODELS = {"facebook/opt-125m": "opt", "gpt2-medium": "medium"}


def extract_in_parts(df, name, extract, output):
    """Run extract over df in saved parts, then join them in df's row order."""
    if (CACHE / output).exists():
        return
    parts_dir = CACHE / name
    parts_dir.mkdir(parents=True, exist_ok=True)
    order = df.text.str.len().sort_values(kind="stable").index
    starts = range(0, len(df), PART_SIZE)
    t0 = time.time()
    for k, start in enumerate(starts):
        path = parts_dir / f"part_{k:03d}.csv"
        if path.exists():
            continue
        chunk = df.loc[order[start:start + PART_SIZE]]
        extract(chunk.text.tolist()).assign(id=chunk.id.values).to_csv(path, index=False)
        print(f"{name} part {k + 1}/{len(starts)} done, {time.time() - t0:.0f}s elapsed", flush=True)
    parts = pd.concat(pd.read_csv(parts_dir / f"part_{k:03d}.csv") for k in range(len(starts)))
    parts.set_index("id").loc[df.id].reset_index().to_csv(CACHE / output, index=False)


def main():
    df = load_all().reset_index(drop=True)
    extract_in_parts(df, "gpt2_parts", surprisal_features, "gpt2_surprisal.csv")
    extract_in_parts(df, "curvature_parts", curvature_features, "gpt2_curvature.csv")
    for model_name, prefix in LANGUAGE_MODELS.items():
        extract_in_parts(df, f"{prefix}_parts", partial(language_model_features, model_name=model_name, prefix=prefix),
                         f"{prefix}_lm.csv")
    if not (CACHE / "minilm.npz").exists():
        t0 = time.time()
        np.savez_compressed(CACHE / "minilm.npz", id=df.id.values, emb=sentence_embeddings(df.text.tolist()))
        print(f"minilm done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
