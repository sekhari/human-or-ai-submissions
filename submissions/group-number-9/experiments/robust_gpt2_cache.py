"""Resumable one-time inference cache for original and flattened labeled text.

Run --smoke first. Default caches labeled text only; --test adds original test
passages, only needed after model selection. Never reads unlabeled.csv.
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.features import strip_layout
from src.gpt2_robust_features import extract_batch, text_hash

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "robust_boost"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(4)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps":
        torch.mps.set_per_process_memory_fraction(.5)
    spec = json.loads((ROOT / ".cache" / "gpt2_model.json").read_text())
    path = ROOT / spec["path"]
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
                                                torch_dtype=torch.float32).eval().to(device)
    train = pd.read_csv(ROOT / "Data" / "train.csv")
    texts = train.text.tolist() + train.text.map(strip_layout).tolist()
    if args.test:
        texts += pd.read_csv(ROOT / "Data" / "test.csv").text.tolist()
    unique = {text_hash(t): t for t in texts}
    CACHE.mkdir(exist_ok=True, parents=True)
    metadata = {"model": spec, "max_length": 512, "dtype": "float32", "features": 19,
                "source_sha256": text_hash((ROOT / "src/gpt2_robust_features.py").read_text()),
                "no_unlabeled_data": True}
    meta_path = CACHE / "features_metadata.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != metadata:
        raise ValueError("Feature cache metadata changed; use a separate cache directory")
    meta_path.write_text(json.dumps(metadata, indent=2))
    shards = CACHE / "feature_shards"
    shards.mkdir(exist_ok=True)
    done = set()
    for p in sorted(shards.glob("*.csv")):
        done.update(pd.read_csv(p, usecols=["text_sha256"]).text_sha256)
    pending = sorted([(h, t) for h, t in unique.items() if h not in done],
                     key=lambda item: -len(item[1]))
    if args.smoke:
        pending = pending[:32]
    print(json.dumps({"device": device, "unique_total": len(unique), "cached": len(done),
                      "pending": len(pending), "batch_size": args.batch_size}), flush=True)
    started = time.monotonic()
    saved = 0
    for start in range(0, len(pending), 128):
        chunk = pending[start:start + 128]
        rows = []
        for b in range(0, len(chunk), args.batch_size):
            items = chunk[b:b + args.batch_size]
            feats = extract_batch([t for _, t in items], tok, model, device)
            feats.insert(0, "text_sha256", [h for h, _ in items])
            rows.append(feats)
        frame = pd.concat(rows, ignore_index=True)
        name = text_hash("".join(frame.text_sha256))[:20]
        temp = shards / f"{name}.tmp"
        frame.to_csv(temp, index=False)
        temp.replace(shards / f"{name}.csv")
        saved += len(frame)
        elapsed = time.monotonic() - started
        print(json.dumps({"saved": saved, "seconds": round(elapsed, 2),
                          "docs_per_second": round(saved / elapsed, 3),
                          "eta_minutes": round((len(pending) - saved) * elapsed / saved / 60, 1)}), flush=True)
    all_features = pd.concat([pd.read_csv(p) for p in sorted(shards.glob("*.csv"))], ignore_index=True)
    if all_features.text_sha256.duplicated().any():
        raise ValueError("Duplicate cached hashes")
    all_features.to_csv(CACHE / "gpt2_features.csv", index=False)
    print(f"wrote {len(all_features)} cached documents", flush=True)


if __name__ == "__main__":
    main()
