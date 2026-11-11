"""Zero-shot-style features from pretrained causal language models, fed to a
light classifier (or used directly).

Idea (GLTR / DetectGPT / Binoculars family; Ma et al. 2023 also find
perplexity among the strongest single signals): LLM text consists of tokens
a language model finds *likely* (low surprisal, high rank) with less
variation than human text. These features don't depend on topic vocabulary.

Per text, from the observer LM we compute statistics of token log-probs,
ranks and predictive entropy. If `performer_name` is set (same tokenizer
family, e.g. Qwen2.5-1.5B + Qwen2.5-1.5B-Instruct) we also compute the
Binoculars score (Hans et al., ICML 2024):

    binoculars = log PPL_performer(x) / X-PPL(observer || performer)(x)

where X-PPL is the average cross-entropy between the observer's and the
performer's next-token distributions. Low values => machine-generated.

Features are cached on disk (outputs/cache/) keyed by text hash, so CV folds
and re-runs don't recompute them. Stats are computed one sequence at a time
so large vocabularies (Qwen: 152k) fit on a 12 GB GPU.

head:
  logreg    : StandardScaler + logistic regression on the features
  lightgbm  : gradient boosting on the features
  zero_shot : no training; score = -binoculars (if performer set) else
              -(mean surprisal / mean entropy). Reference point for "how far
              does a training-free detector get".

device: auto | cuda | cuda:<i> | mps | cpu (filled from config `device:` / `--device`).
dtype:  auto (bf16/fp16 on CUDA, fp32 elsewhere) | fp32 | fp16 | bf16
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import joblib
import numpy as np
from tqdm.auto import tqdm

from ..config import ROOT
from ..features import stylometric_features
from .base import BaseModel

LM_FEATURE_NAMES = ["lp_mean", "lp_std", "lp_min", "lp_p10", "lp_p25", "lp_p50", "lp_p75",
                    "lp_p90", "ent_mean", "ent_std", "surprisal_over_entropy",
                    "rank1_frac", "rank10_frac", "logrank_mean", "n_tokens"]
BINO_FEATURE_NAMES = ["perf_lp_mean", "x_ppl", "binoculars"]


def _md5(t: str) -> str:
    return hashlib.md5(t.encode("utf-8")).hexdigest()


def _safe(name: str) -> str:
    return name.replace("/", "__").replace("\\", "__").replace(":", "_")


class LMFeatureExtractor:
    def __init__(self, lm_name="gpt2", max_length=512, batch_size=8, device="auto",
                 performer_name=None, dtype="auto"):
        self.lm_name, self.performer_name = lm_name, performer_name
        self.max_length, self.batch_size = max_length, batch_size
        self.device_pref, self.dtype = device, dtype
        tag = _safe(lm_name) + (f"+{_safe(performer_name)}" if performer_name else "")
        self.cache_name = f"lm_{tag}_{max_length}.joblib"
        self._cache = None
        self._model = None

    @property
    def names(self):
        return LM_FEATURE_NAMES + (BINO_FEATURE_NAMES if self.performer_name else [])

    @property
    def cache_path(self) -> Path:
        return ROOT / "outputs" / "cache" / self.cache_name

    def _load_cache(self):
        if self._cache is None:
            self._cache = joblib.load(self.cache_path) if self.cache_path.exists() else {}
        return self._cache

    def _save_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(f".tmp{os.getpid()}")
        joblib.dump(self._cache, tmp)
        os.replace(tmp, self.cache_path)  # atomic: a crash never leaves a half-written cache

    def _torch_dtype(self, torch, dev_type):
        d = str(self.dtype).lower()
        if d == "auto":
            if dev_type == "cuda":
                return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return torch.float32
        return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[d]

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from ..utils import device as pick_device, device_type
        self.torch = torch
        self.device = pick_device(self.device_pref)
        dt = self._torch_dtype(torch, device_type(self.device))
        from ..utils import hf_source
        self.tok = AutoTokenizer.from_pretrained(hf_source(self.lm_name))
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"

        def load(name):
            try:
                m = AutoModelForCausalLM.from_pretrained(hf_source(name), dtype=dt)
            except TypeError:  # transformers < 4.56 uses torch_dtype
                m = AutoModelForCausalLM.from_pretrained(hf_source(name), torch_dtype=dt)
            return m.to(self.device).eval()

        self._model = load(self.lm_name)
        self._perf = load(self.performer_name) if self.performer_name else None
        if self._perf is not None:
            v1 = self._model.get_output_embeddings().weight.shape[0]
            v2 = self._perf.get_output_embeddings().weight.shape[0]
            if v1 != v2:
                raise ValueError(f"Binoculars needs models with the same vocabulary: "
                                 f"{self.lm_name} ({v1}) vs {self.performer_name} ({v2})")
        print(f"  LM features: {self.lm_name}" + (f" + {self.performer_name}" if self.performer_name else "")
              + f" on {self.device} ({str(dt).replace('torch.', '')})", flush=True)

    @staticmethod
    def _stats(lp, ent, rank):
        if len(lp) == 0:
            return [0.0] * len(LM_FEATURE_NAMES)
        q = np.percentile(lp, [10, 25, 50, 75, 90])
        return [lp.mean(), lp.std(), lp.min(), *q, ent.mean(), ent.std(),
                (-lp.mean()) / (ent.mean() + 1e-6), (rank == 0).mean(), (rank < 10).mean(),
                np.log1p(rank).mean(), float(len(lp))]

    def _compute(self, texts):
        if self._model is None:
            self._load()
        torch = self.torch
        out = []
        bos = self.tok.bos_token
        for i in tqdm(range(0, len(texts), self.batch_size), desc="LM features", leave=False):
            batch = texts[i:i + self.batch_size]
            enc = self.tok([(bos + t) if bos else t for t in batch], return_tensors="pt",
                           padding=True, truncation=True, max_length=self.max_length).to(self.device)
            with torch.inference_mode():
                logits_o = self._model(**enc).logits
                logits_p = self._perf(**enc).logits if self._perf is not None else None
            for j in range(len(batch)):
                m = enc["attention_mask"][j, 1:].bool()
                tgt = enc["input_ids"][j, 1:][m]
                lo = torch.log_softmax(logits_o[j, :-1][m].float(), -1)       # (T, V)
                lp = lo.gather(-1, tgt[:, None]).squeeze(-1)
                ent = -(lo.exp() * lo).sum(-1)
                rank = (lo > lp[:, None]).sum(-1)
                feats = self._stats(lp.cpu().numpy(), ent.cpu().numpy(), rank.cpu().numpy())
                if logits_p is not None:
                    lpp_all = torch.log_softmax(logits_p[j, :-1][m].float(), -1)
                    lpp = lpp_all.gather(-1, tgt[:, None]).squeeze(-1)
                    x_ent = -(lo.exp() * lpp_all).sum(-1)   # observer probs x performer log-probs
                    ppl_log = float(-lpp.mean()) if len(lpp) else 0.0
                    x_ppl = float(x_ent.mean()) if len(x_ent) else 1.0
                    feats += [float(lpp.mean()) if len(lpp) else 0.0, x_ppl, ppl_log / (x_ppl + 1e-6)]
                out.append(feats)
            del logits_o, logits_p
        return out

    def transform(self, texts) -> np.ndarray:
        texts = list(texts)
        cache = self._load_cache()
        keys = [_md5(t) for t in texts]
        todo = list({k: t for k, t in zip(keys, texts) if k not in cache}.items())
        if todo:
            for s in range(0, len(todo), 2000):          # save progress every 2000 texts
                chunk = todo[s:s + 2000]
                feats = self._compute([t for _, t in chunk])
                cache.update({k: f for (k, _), f in zip(chunk, feats)})
                self._save_cache()
        return np.array([cache[k] for k in keys], dtype=np.float32)

    def release(self):
        self._model = self._perf = None
        if hasattr(self, "torch"):
            import gc
            gc.collect()
            if self.device.startswith("cuda"):
                self.torch.cuda.empty_cache()

    def __getstate__(self):  # never pickle the LMs / cache into model.joblib
        s = self.__dict__.copy()
        for k in ("_model", "_perf", "tok", "torch", "device"):
            s.pop(k, None)
        s["_cache"], s["_model"] = None, None
        s["device_pref"] = "auto"  # devices are machine-specific; re-detected on load
        return s

    def __setstate__(self, s):
        s.setdefault("performer_name", None)
        s.setdefault("dtype", "auto")
        if "cache_name" not in s:  # runs saved by an older version
            s["cache_name"] = Path(str(s.get("cache_path", "lm_cache.joblib"))).name
        s.pop("cache_path", None)
        self.__dict__.update(s)


class LMPerplexity(BaseModel):
    name = "lm_perplexity"

    def __init__(self, lm_name: str = "gpt2", performer_name: str | None = None,
                 max_length: int = 512, batch_size: int = 8, dtype: str = "auto",
                 head: str = "lightgbm", add_stylometric: bool = True, C: float = 1.0,
                 lgbm: dict | None = None, seed: int = 42, device: str = "auto"):
        super().__init__(lm_name=lm_name, performer_name=performer_name, head=head,
                         add_stylometric=add_stylometric)
        self.extractor = LMFeatureExtractor(lm_name, max_length, batch_size, device,
                                            performer_name, dtype)
        self.head, self.add_stylometric, self.C = head, add_stylometric, C
        self.lgbm_params, self.seed = lgbm or {}, seed

    def set_device(self, device: str) -> None:
        self.extractor.device_pref = device
        self.extractor._model = None

    def precompute(self, texts) -> None:
        """Compute + cache LM features for all texts in one pass (train.py calls
        this once before CV so the LM is loaded only once)."""
        self.extractor.transform(texts)
        self.extractor.release()

    def _features(self, texts):
        F = self.extractor.transform(texts)
        if self.add_stylometric:
            F = np.hstack([F, stylometric_features(texts)])
        return F

    def fit(self, texts, y, val_texts=None, val_y=None, unlabeled_texts=None):
        if self.head == "zero_shot":
            return self
        X = self._features(texts)
        if self.head == "logreg":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            self.clf = make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=5000))
        elif self.head == "lightgbm":
            import lightgbm as lgb
            p = dict(n_estimators=500, learning_rate=0.03, num_leaves=15, subsample=0.8,
                     subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=self.seed)
            p.update(self.lgbm_params)
            self.clf = lgb.LGBMClassifier(**p)
        else:
            raise ValueError(f"Unknown head {self.head}")
        self.clf.fit(X, y)
        return self

    def predict_proba(self, texts):
        if self.head == "zero_shot":
            F = self.extractor.transform(texts)
            names = self.extractor.names
            col = "binoculars" if "binoculars" in names else "surprisal_over_entropy"
            s = F[:, names.index(col)]
            return 1.0 / (1.0 + np.exp((s - np.median(s)) / (s.std() + 1e-6)))  # lower => more AI-like
        return self.clf.predict_proba(self._features(texts))[:, 1]
