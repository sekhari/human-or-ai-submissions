"""Fine-tune a pretrained Hugging Face encoder (RoBERTa, DeBERTa-v3, ELECTRA,
ModernBERT, the RoBERTa OpenAI-detector, ...) for binary sequence classification.

Device: `device` param (filled from config `device:` / `--device`):
auto | cuda | cuda:<i> | mps | cpu. The device is not saved with the model, so
a model trained on CUDA can be loaded on a Mac (and vice versa).

precision: auto | fp32 | fp16 | bf16
  auto -> bf16 on CUDA GPUs that support it (RTX 30xx / 40xx / 50xx), fp16 on
  older CUDA GPUs, fp32 on MPS / CPU. bf16 avoids the fp16 overflow problems
  DeBERTa-v3 is known for and needs no loss scaling.

gradient_checkpointing: trades ~30% speed for much less activation memory --
turn on for *-large models on 12 GB cards.

Inference sorts texts by length so batches carry little padding (2-3x faster),
then restores the original order.

checkpoint_every_epoch: after each epoch, save model + optimizer + scheduler +
grad scaler + RNG state + epoch count to <checkpoint_dir>/last.pt (train.py sets
the directory per CV fold: outputs/checkpoints/<run>/fold_<k>/). A later fit()
with the same directory, hyper-parameters and training data resumes after the
last finished epoch (a partly done epoch is redone from its start). Only the
latest checkpoint is kept. `history` holds train loss / val AUROC per epoch.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .base import BaseModel


class TransformerClassifier(BaseModel):
    name = "transformer"

    def __init__(self, model_name: str = "distilroberta-base", max_length: int = 512,
                 batch_size: int = 16, eval_batch_size: int = 64, lr: float = 2e-5,
                 epochs: int = 3, weight_decay: float = 0.01, warmup_ratio: float = 0.1,
                 grad_accum: int = 1, precision: str = "auto", gradient_checkpointing: bool = False,
                 keep_best_epoch: bool = False, max_grad_norm: float = 1.0,
                 checkpoint_every_epoch: bool = False,
                 seed: int = 42, device: str = "auto", fp16: bool | None = None):
        if fp16 is not None:  # backwards compatibility with older configs
            precision = "fp16" if fp16 else "fp32"
        self.hp = dict(model_name=model_name, max_length=max_length, batch_size=batch_size,
                       eval_batch_size=eval_batch_size, lr=lr, epochs=epochs,
                       weight_decay=weight_decay, warmup_ratio=warmup_ratio, grad_accum=grad_accum,
                       precision=precision, gradient_checkpointing=gradient_checkpointing,
                       keep_best_epoch=keep_best_epoch, max_grad_norm=max_grad_norm,
                       checkpoint_every_epoch=checkpoint_every_epoch, seed=seed)
        super().__init__(**self.hp)
        self.device_pref = device
        self.model = self.tok = None
        self.checkpoint_dir = None
        self.history = []

    # ------------------------------------------------------------- setup
    def _setup(self, path=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from ..utils import device, device_type
        self.torch = torch
        self.device = device(self.device_pref)
        from ..utils import hf_source
        src = path or hf_source(self.hp["model_name"])
        self.tok = AutoTokenizer.from_pretrained(src)
        if self.tok.pad_token is None:  # e.g. GPT-style tokenizers
            self.tok.pad_token = self.tok.eos_token or self.tok.unk_token
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(src)
        ai_index = None
        if path is None:  # fresh from the hub: map an existing 2-way detector head onto our labels
            names = {int(k): str(v).lower() for k, v in (getattr(config, "id2label", None) or {}).items()}
            hits = [i for i, n in names.items() if any(w in n for w in ("fake", "machine", "generated", "ai"))
                    and not n.startswith("label_")]
            if config.num_labels == 2 and len(hits) == 1:
                ai_index = hits[0]
                print(f"  pretrained head labels {names} -> AI class is index {ai_index}")
            config.num_labels = 2
            config.id2label, config.label2id = {0: "human", 1: "ai"}, {"human": 0, "ai": 1}
        if getattr(config, "model_type", "") == "modernbert":
            config.reference_compile = False  # torch.compile needs Triton, which Windows lacks
        # Master weights must be fp32: transformers>=5 otherwise keeps the checkpoint's stored dtype
        # (DeBERTa-v3 ships fp16), and AdamW on fp16 weights turns eps=1e-8 into 0 -> NaN after
        # the first step. Mixed precision comes from autocast, not from the stored weights.
        kw = dict(config=config, ignore_mismatched_sizes=True)  # other heads are re-initialised
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(src, dtype=torch.float32, **kw)
        except TypeError:  # transformers < 4.56 uses torch_dtype
            self.model = AutoModelForSequenceClassification.from_pretrained(src, torch_dtype=torch.float32, **kw)
        if ai_index == 0:  # swap the two output rows so index 1 = AI, keeping the detector's knowledge
            out = [m for m in self.model.modules() if isinstance(m, torch.nn.Linear) and m.out_features == 2][-1]
            with torch.no_grad():
                out.weight.copy_(out.weight.flip(0))
                if out.bias is not None:
                    out.bias.copy_(out.bias.flip(0))
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tok.pad_token_id
        self.model.to(self.device)
        if self.hp["gradient_checkpointing"] and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        self.amp_dtype = self._resolve_precision(torch, device_type(self.device))
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"  transformer {src} ({n_params:.0f}M params) on {self.device}, "
              f"precision={self.amp_dtype or 'fp32'}")

    def _resolve_precision(self, torch, dev_type):
        p = str(self.hp["precision"]).lower()
        if dev_type != "cuda":
            if p in ("fp16", "bf16"):
                print(f"  precision={p} is only used on CUDA; running fp32 on {dev_type}")
            return None
        if p == "auto":
            return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        if p == "bf16" and not torch.cuda.is_bf16_supported():
            print("  bf16 not supported on this GPU -> fp16")
            return "fp16"
        return None if p == "fp32" else p

    def _autocast(self):
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        dt = self.torch.bfloat16 if self.amp_dtype == "bf16" else self.torch.float16
        return self.torch.autocast(device_type="cuda", dtype=dt)

    def _encode(self, texts):
        enc = self.tok(list(texts), truncation=True, max_length=self.hp["max_length"],
                       padding=True, return_tensors="pt")
        if "token_type_ids" in enc and "token_type_ids" not in self._forward_args():
            enc.pop("token_type_ids")
        return {k: v.to(self.device) for k, v in enc.items()}

    def _forward_args(self):
        if not hasattr(self, "_fwd_args"):
            import inspect
            self._fwd_args = set(inspect.signature(self.model.forward).parameters)
        return self._fwd_args

    def _empty_cache(self):
        dt = self.device.split(":")[0]
        if dt == "cuda":
            self.torch.cuda.empty_cache()
        elif dt == "mps" and hasattr(self.torch, "mps"):
            self.torch.mps.empty_cache()

    # ------------------------------------------------------------- checkpoints
    def set_checkpoint_dir(self, path) -> None:
        """Where fit() keeps its per-epoch checkpoint (used if checkpoint_every_epoch)."""
        self.checkpoint_dir = Path(path) if path else None

    def _ckpt_file(self):
        if not (self.hp["checkpoint_every_epoch"] and self.checkpoint_dir):
            return None
        return self.checkpoint_dir / "last.pt"

    def _fingerprint(self, texts, y) -> dict:
        """What a checkpoint must match to be resumed: training hyper-parameters + training data."""
        h = hashlib.md5()
        for t, label in zip(texts, y):
            h.update(t.encode("utf-8"))
            h.update(b"\x01" if label else b"\x00")
        hp = {k: v for k, v in self.hp.items() if k not in ("eval_batch_size", "checkpoint_every_epoch")}
        return {"hp": json.dumps(hp, sort_keys=True), "data": h.hexdigest(), "n": len(texts)}

    def _save_checkpoint(self, path, epoch, opt, sched, scaler, rng, fingerprint, best):
        torch = self.torch
        state = {
            "epochs_done": epoch, "fingerprint": fingerprint, "history": self.history,
            "model": self.model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict() if scaler else None,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "shuffle": rng.bit_generator.state,
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
            "best": best,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        torch.save(state, tmp)
        os.replace(tmp, path)  # atomic: a kill during saving leaves the previous checkpoint intact
        print(f"  checkpoint: epoch {epoch} saved to {path}", flush=True)

    def _load_checkpoint(self, path, opt, sched, scaler, rng, fingerprint):
        """Restore training state; returns (epochs_done, best) or None if the checkpoint doesn't fit."""
        torch = self.torch
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["fingerprint"] != fingerprint:
            print(f"  checkpoint {path} was made with other hyper-parameters or data -> starting this fit from "
                  f"scratch", flush=True)
            return None
        self.model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        if scaler is not None and state["scaler"] is not None:
            scaler.load_state_dict(state["scaler"])
        r = state["rng"]
        random.setstate(r["python"])
        np.random.set_state(r["numpy"])
        rng.bit_generator.state = r["shuffle"]
        torch.set_rng_state(r["torch"])
        if r["cuda"] is not None and torch.cuda.is_available() and len(r["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(r["cuda"])
        self.history = list(state["history"])
        print(f"  checkpoint: resumed from {path} after epoch {state['epochs_done']}/{self.hp['epochs']}", flush=True)
        return state["epochs_done"], state["best"]

    # ------------------------------------------------------------- API
    def fit(self, texts, y, val_texts=None, val_y=None, unlabeled_texts=None):
        from transformers import get_linear_schedule_with_warmup
        from ..utils import auroc
        self._setup()
        torch, hp = self.torch, self.hp
        texts, y = list(texts), np.asarray(y)
        rng = np.random.default_rng(hp["seed"])
        no_decay = ("bias", "LayerNorm.weight", "layernorm", "norm.weight")
        named = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        groups = [
            {"params": [p for n, p in named if not any(k in n for k in no_decay)], "weight_decay": hp["weight_decay"]},
            {"params": [p for n, p in named if any(k in n for k in no_decay)], "weight_decay": 0.0},
        ]
        opt = torch.optim.AdamW(groups, lr=hp["lr"])
        n_batches = math.ceil(len(texts) / hp["batch_size"])
        total = math.ceil(n_batches / hp["grad_accum"]) * hp["epochs"]
        sched = get_linear_schedule_with_warmup(opt, int(hp["warmup_ratio"] * total), total)
        scaler = torch.amp.GradScaler("cuda") if self.amp_dtype == "fp16" else None
        best_auc, best_state = -1.0, None
        self.history, start_epoch = [], 0
        ckpt = self._ckpt_file()
        fingerprint = self._fingerprint(texts, y) if ckpt else None
        if ckpt is not None and ckpt.exists():
            restored = self._load_checkpoint(ckpt, opt, sched, scaler, rng, fingerprint)
            if restored is not None:
                start_epoch, (best_auc, best_state) = restored[0], restored[1]

        for epoch in range(start_epoch, hp["epochs"]):
            self.model.train()
            order = rng.permutation(len(texts))
            bar = tqdm(range(n_batches), desc=f"epoch {epoch + 1}/{hp['epochs']}", leave=False)
            running = None
            for step in bar:
                b = order[step * hp["batch_size"]:(step + 1) * hp["batch_size"]]
                batch = self._encode([texts[j] for j in b])
                batch["labels"] = torch.tensor(y[b], dtype=torch.long, device=self.device)
                with self._autocast():
                    loss = self.model(**batch).loss / hp["grad_accum"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at epoch {epoch + 1} step {step} "
                                             f"(try model.params.precision=fp32 or a lower lr)")
                (scaler.scale(loss) if scaler else loss).backward()
                if (step + 1) % hp["grad_accum"] == 0 or step + 1 == n_batches:
                    if scaler:
                        scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), hp["max_grad_norm"])
                    if scaler:
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad(set_to_none=True)
                    sched.step()
                l = loss.item() * hp["grad_accum"]
                running = l if running is None else 0.98 * running + 0.02 * l
                bar.set_postfix(loss=f"{running:.4f}", lr=f"{sched.get_last_lr()[0]:.1e}", refresh=False)
            self._empty_cache()
            entry = {"epoch": epoch + 1, "train_loss": round(float(running), 5)}
            if val_texts is not None and val_y is not None:
                # Logged only (the epoch count is fixed), so OOF predictions stay honest --
                # except with keep_best_epoch, which selects on this fold's validation data.
                auc = auroc(val_y, self.predict_proba(val_texts))
                entry["val_auc"] = round(float(auc), 5)
                print(f"  epoch {epoch + 1}: train loss {running:.4f}  val AUROC {auc:.4f}", flush=True)
                if hp["keep_best_epoch"] and auc > best_auc:
                    best_auc = auc
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            self.history.append(entry)
            if ckpt is not None:
                self._save_checkpoint(ckpt, epoch + 1, opt, sched, scaler, rng, fingerprint,
                                      (best_auc, best_state))
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, texts):
        torch = self.torch
        self.model.eval()
        texts = list(texts)
        order = np.argsort([len(t) for t in texts])[::-1]  # longest first: an OOM shows up immediately
        bs = self.hp["eval_batch_size"]
        out = np.empty(len(texts), dtype=np.float32)
        with torch.inference_mode():
            for i in tqdm(range(0, len(texts), bs), desc="predict", leave=False):
                idx = order[i:i + bs]
                with self._autocast():
                    logits = self.model(**self._encode([texts[j] for j in idx])).logits.float()
                out[idx] = torch.softmax(logits, -1)[:, 1].cpu().numpy()
        return out

    def set_device(self, device: str) -> None:
        self.device_pref = device

    def release(self) -> None:
        """Free GPU memory (called by train.py between CV folds)."""
        self.model = None
        if hasattr(self, "torch"):
            import gc
            gc.collect()
            self._empty_cache()

    # ------------------------------------------------------------- persistence
    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path / "hf_model")
        self.tok.save_pretrained(path / "hf_model")
        with open(path / "transformer_hparams.json", "w", encoding="utf-8") as f:
            json.dump(self.hp, f, indent=2)

    @classmethod
    def load(cls, path, device: str | None = None):
        path = Path(path)
        with open(path / "transformer_hparams.json", encoding="utf-8") as f:
            hp = json.load(f)
        hp.pop("device", None)
        if "fp16" in hp:  # older saved runs
            hp["precision"] = "fp16" if hp.pop("fp16") else "fp32"
        obj = cls(**hp, device=device or "auto")
        obj._setup(str(path / "hf_model"))
        return obj
