"""Model registry. Imports are lazy so that sklearn-only models work without
torch/transformers installed."""
from __future__ import annotations

import importlib

# name -> "module:ClassName" (module relative to src.models)
REGISTRY = {
    "tfidf_logreg": "tfidf_logreg:TfidfLogReg",
    "stylometric_gbm": "stylometric_gbm:StylometricGBM",
    "lm_perplexity": "lm_perplexity:LMPerplexity",
    "transformer": "transformer:TransformerClassifier",
}


def get_model_class(name: str):
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(REGISTRY)}")
    module, cls = REGISTRY[name].split(":")
    return getattr(importlib.import_module(f"src.models.{module}"), cls)


def build_model(model_cfg: dict):
    return get_model_class(model_cfg["name"])(**(model_cfg.get("params") or {}))
