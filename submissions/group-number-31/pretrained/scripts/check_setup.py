"""Check that this machine can run everything: Python packages, data files,
GPUs (including whether the installed PyTorch build supports them), and
optionally that every pretrained model loads and runs one forward pass.

  python scripts/check_setup.py              # environment + GPUs + data (seconds)
  python scripts/check_setup.py --models     # + load every pretrained model and run 2 texts through it
  python scripts/check_setup.py --models --configs ft_roberta_base lm_gpt2

Exit code 0 = no FAIL.
"""
from __future__ import annotations

import argparse
import importlib
import os
import platform
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
if sys.platform == "darwin":  # LightGBM + PyTorch each ship libomp on macOS; allow both in one process
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = []


def report(status, what, detail=""):
    RESULTS.append(status)
    print(f"  [{status:<4}] {what}" + (f" -- {detail}" if detail else ""))


def check_packages():
    print("Packages")
    report("ok", "python", f"{platform.python_version()} on {platform.system()} {platform.machine()}")
    if sys.version_info < (3, 9):
        report("FAIL", "python >= 3.9 required")
    for mod, need in [("numpy", True), ("pandas", True), ("sklearn", True), ("scipy", True), ("yaml", True),
                      ("tqdm", True), ("matplotlib", True), ("lightgbm", True), ("torch", False),
                      ("transformers", False), ("tokenizers", False), ("sentencepiece", False),
                      ("huggingface_hub", False)]:
        try:
            m = importlib.import_module(mod)
            report("ok", mod, getattr(m, "__version__", ""))
        except Exception as e:
            report("FAIL" if need else "WARN", mod,
                   f"not importable ({type(e).__name__}) -- " + ("required" if need else "needed for pretrained models"))
    try:
        import transformers
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        if (major, minor) < (4, 48):
            report("WARN", "transformers >= 4.48 needed for recent model classes")
    except Exception:
        pass


def check_torch():
    print("Torch devices")
    try:
        import torch
    except ImportError:
        report("WARN", "torch not installed -- only the CPU models (tfidf_*, stylometric_*) can run")
        return
    report("ok", "torch", f"{torch.__version__}, CUDA build {torch.version.cuda}")
    if torch.cuda.is_available():
        archs = torch.cuda.get_arch_list()
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            sm = f"sm_{p.major}{p.minor}"
            desc = f"GPU {i}: {p.name}, {p.total_memory / 1e9:.1f} GB, {sm}, bf16={torch.cuda.is_bf16_supported()}"
            if archs and sm not in archs and f"compute_{p.major}{p.minor}" not in archs:
                report("FAIL", desc, f"this PyTorch build only has kernels for {', '.join(archs)}. For RTX 50xx "
                       "(Blackwell, sm_120) install a CUDA 12.8+ build:  pip install --upgrade torch "
                       "--index-url https://download.pytorch.org/whl/cu128")
                continue
            try:
                with torch.cuda.device(i):
                    a = torch.randn(512, 512, device=f"cuda:{i}")
                    (a @ a).sum().item()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        (a @ a).sum().item()
                report("ok", desc, "matmul + bf16 autocast test passed")
            except Exception as e:
                report("FAIL", desc, f"{type(e).__name__}: {str(e).splitlines()[0][:160]}")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        try:
            a = torch.randn(256, 256, device="mps")
            (a @ a).sum().item()
            report("ok", "Apple MPS GPU", "matmul test passed")
        except Exception as e:
            report("FAIL", "Apple MPS GPU", str(e)[:160])
    else:
        report("WARN", "no GPU visible to torch", "neural models will run on CPU (slow). On an NVIDIA PC check "
               "`nvidia-smi` and that you installed a CUDA build of torch (torch.version.cuda is None for CPU builds)")


def check_data():
    print("Data")
    import pandas as pd
    for f, n in [("train.csv", 10000), ("test.csv", 3000), ("unlabeled.csv", 3000), ("sample_submission.csv", 3000),
                 ("train_starter_2000.csv", 2000)]:
        p = ROOT / "data" / f
        if not p.exists():
            report("FAIL", f"data/{f}", "missing -- download from the Kaggle Data tab into data/")
            continue
        d = pd.read_csv(p)
        report("ok" if len(d) == n else "WARN", f"data/{f}", f"{len(d)} rows")


def check_models(configs, device):
    print(f"Pretrained models (one forward pass each on {device})")
    from src.config import available_configs, load_config
    from src.models import build_model, get_model_class
    texts = ["The results suggest that the proposed method improves robustness across domains.",
             "lol my cat just knocked my coffee off the table again, third time this week"]
    import inspect
    for c in configs or available_configs():
        cfg = load_config(c)
        name = cfg["model"]["name"]
        if name not in ("transformer", "lm_perplexity"):
            continue
        params = dict(cfg["model"]["params"])
        if "device" in inspect.signature(get_model_class(name).__init__).parameters:
            params["device"] = device
        try:
            m = build_model({"name": name, "params": params})
            if name == "transformer":
                m._setup()
                p = m.predict_proba(texts)
            else:
                ext = m.extractor
                f = ext.transform(texts)
                p = f[:, 0]
                ext.release()
            ok = len(p) == 2 and all(map(lambda v: v == v, p))
            report("ok" if ok else "FAIL", c, f"{name} -> output shape ok")
            if hasattr(m, "release"):
                m.release()
        except Exception as e:
            report("FAIL", c, f"{type(e).__name__}: {str(e).splitlines()[0][:200]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", action="store_true", help="also load + run every pretrained model")
    ap.add_argument("--configs", nargs="*", help="limit --models to these configs")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    check_packages()
    check_torch()
    check_data()
    if args.models:
        from src.utils import device
        check_models(args.configs, device(args.device))
    n_fail, n_warn = RESULTS.count("FAIL"), RESULTS.count("WARN")
    print(f"\n{n_fail} FAIL, {n_warn} WARN")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
