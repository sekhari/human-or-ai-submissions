"""Seeding, metrics, run bookkeeping, the results log (with a cross-process
file lock so parallel jobs can log safely), and torch device helpers."""
from __future__ import annotations

import contextlib
import csv
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from .config import ROOT, resolve

RESULT_FIELDS = ["timestamp", "run_name", "model", "config", "author", "cv_scheme", "n_folds",
                 "cv_auc", "cv_auc_std", "fold_aucs", "tpr_at_1fpr", "train_frac", "semi_supervised",
                 "external_files", "preprocessing", "params", "device", "train_time_s",
                 "kaggle_public", "notes", "eval_auc"]  # new columns go at the end (older rows get migrated)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if "torch" in sys.modules:  # don't pay the torch import for sklearn-only runs
        torch = sys.modules["torch"]
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def auroc(y_true, y_score) -> float:
    return float(roc_auc_score(y_true, y_score))


def tpr_at_fpr(y_true, y_score, max_fpr: float) -> float:
    """Recall on AI text while flagging at most `max_fpr` of human text
    (the operating point GPTZero reports)."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ok = fpr <= max_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def binary_metrics(y_true, y_score) -> dict:
    y_true = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(y_score, dtype=float), 1e-6, 1 - 1e-6)
    pred = (p >= 0.5).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    return {
        "auroc": auroc(y_true, p),
        "tpr_at_1fpr": tpr_at_fpr(y_true, p, 0.01),
        "tpr_at_5fpr": tpr_at_fpr(y_true, p, 0.05),
        "accuracy": float((pred == y_true).mean()),
        "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "log_loss": float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))),
        "brier": float(np.mean((p - y_true) ** 2)),
    }


def make_run_name(model: str, tag: str | None = None) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}_{model}" + (f"_{tag}" if tag else "")


def git_user() -> str:
    try:
        return subprocess.check_output(["git", "config", "user.name"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip() or "unknown"
    except Exception:
        return os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"


@contextlib.contextmanager
def file_lock(path, timeout: float = 120.0):
    """Tiny cross-platform lock (works on Windows/macOS/Linux) based on
    exclusive file creation. Stale locks older than `timeout` are broken."""
    lock = Path(str(path) + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > timeout:
                    lock.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.time() - t0 > timeout * 2:
                raise TimeoutError(f"Could not acquire {lock}")
            time.sleep(0.1)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def append_result(cfg: dict, row: dict) -> None:
    """Append one run to experiments/results.csv. If the file was written with
    an older column set, it is rewritten with the union of columns."""
    path = resolve(cfg["output"]["results_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path):
        if path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                header = reader.fieldnames or []
                old_rows = list(reader) if header != RESULT_FIELDS else None
            if old_rows is not None:  # migrate to the current column set
                fields = RESULT_FIELDS + [c for c in header if c not in RESULT_FIELDS]
                with open(path, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=fields)
                    w.writeheader()
                    w.writerows(old_rows)
                header = fields
        else:
            header = RESULT_FIELDS
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=header).writeheader()
        with open(path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=header).writerow({k: row.get(k, "") for k in header})


def free_torch_memory() -> None:
    """Release cached GPU memory between folds (no-op if torch isn't loaded)."""
    import gc
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def torch_runtime_info() -> dict:
    """GPU name / peak memory for the metrics file (empty if torch not loaded)."""
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return {}
    i = torch.cuda.current_device()
    return {"gpu_name": torch.cuda.get_device_name(i),
            "gpu_peak_mem_gb": round(torch.cuda.max_memory_allocated(i) / 1e9, 2)}


def hf_source(name: str) -> str:
    """Use a local copy of a Hugging Face model if scripts/download_models.py
    saved one under pretrained/<org>__<name>; otherwise return the hub id
    (downloaded once into the normal Hugging Face cache)."""
    if not name or os.path.isdir(name):
        return name
    local = ROOT / "pretrained" / name.replace("/", "__")
    return str(local) if (local / "config.json").exists() else name


def to_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def _mps_available(torch) -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def device(preference: str | None = None) -> str:
    """Resolve which torch device to use.

    preference (first one set wins):
      1. the argument, unless it is "auto" -- from `--device` / config `device:` / a model's `device` param
      2. env var AIVH_DEVICE  (e.g. `export AIVH_DEVICE=cuda` on a GPU box)
      3. auto-detect
    Values: auto | cuda | cuda:<index> | mps | cpu.  "auto" picks cuda -> mps -> cpu.
    Asking for a device that isn't available raises a clear error instead of
    silently falling back, so you always know where a run actually trained.
    """
    import torch
    explicit = str(preference).strip().lower() if preference else "auto"
    pref = explicit if explicit != "auto" else os.environ.get("AIVH_DEVICE", "auto").strip().lower()
    if pref == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "mps" if _mps_available(torch) else "cpu"
    if pref.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"device={pref} requested but CUDA is not available "
                               f"(on a Mac use --device mps or auto).")
        if ":" in pref and int(pref.split(":")[1]) >= torch.cuda.device_count():
            raise RuntimeError(f"device={pref} requested but only {torch.cuda.device_count()} GPU(s) found.")
        return pref
    if pref == "mps":
        if not _mps_available(torch):
            raise RuntimeError("device=mps requested but MPS is not available "
                               "(needs Apple silicon + a torch build with MPS; on NVIDIA use --device cuda).")
        return "mps"
    if pref == "cpu":
        return "cpu"
    raise ValueError(f"Unknown device '{preference}'. Use auto, cuda, cuda:<i>, mps or cpu.")


def device_type(dev: str) -> str:
    """'cuda:1' -> 'cuda' (what torch.autocast / GradScaler expect)."""
    return dev.split(":")[0]


def ensure_dir(p) -> Path:
    p = resolve(p)
    p.mkdir(parents=True, exist_ok=True)
    return p
