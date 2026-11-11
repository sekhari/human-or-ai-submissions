"""Download the pretrained Hugging Face weights used by the model configs.

  python scripts/download_models.py                    # every model in configs/models/
  python scripts/download_models.py --profile mac      # only what a run_all profile needs
  python scripts/download_models.py --configs ft_roberta_base lm_gpt2
  python scripts/download_models.py --ids microsoft/deberta-v3-base   # any Hugging Face repo id
  python scripts/download_models.py --local            # also keep a copy in ./pretrained/ (git-ignored)
  python scripts/download_models.py --list             # just show which models / sizes

By default weights go to the normal Hugging Face cache (~/.cache/huggingface, or
%USERPROFILE%\\.cache\\huggingface on Windows), which every script uses automatically.
With --local they are saved under pretrained/<org>__<name>/, and the code prefers
that copy (handy for offline work or for keeping everything inside the project).
Only PyTorch weights are fetched (no TF / Flax / ONNX duplicates).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import available_configs, read_yaml  # noqa: E402

KEYS = ("model_name", "lm_name", "performer_name")
SKIP = ["*.h5", "*.msgpack", "*.ot", "*.onnx", "onnx/*", "tf_model*", "flax_model*", "rust_model*",
        "*.tflite", "coreml/*", "openvino/*", "*.gguf", "*.mlmodel"]


def hf_ids(configs) -> dict[str, list[str]]:
    ids = {}
    for c in configs:
        params = (read_yaml(ROOT / "configs" / "models" / f"{c}.yaml").get("model") or {}).get("params") or {}
        for k in KEYS:
            v = params.get(k)
            if v and not Path(str(v)).exists():
                ids.setdefault(v, []).append(c)
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="*", help="config names (default: all)")
    ap.add_argument("--profile", help="take the configs from configs/experiments/<profile>.yaml")
    ap.add_argument("--ids", nargs="*", help="explicit Hugging Face repo ids (used by run_all.py)")
    ap.add_argument("--local", action="store_true", help="also save a copy under ./pretrained/")
    ap.add_argument("--list", action="store_true", help="only list the models")
    args = ap.parse_args()

    if args.profile:
        prof = read_yaml(ROOT / "configs" / "experiments" / f"{args.profile}.yaml")
        configs = sorted({j.get("config", j["name"]) for j in prof["jobs"]})
    else:
        configs = args.configs or available_configs()
    ids = {i: ["run_all"] for i in args.ids} if args.ids else hf_ids(configs)
    if not ids:
        print("No Hugging Face models needed for these configs.")
        return

    from huggingface_hub import HfApi, snapshot_download
    from tqdm.auto import tqdm

    api = HfApi()
    failed = []
    print(f"{len(ids)} pretrained model(s):")
    for repo, users in ids.items():
        print(f"  {repo:<50} used by {', '.join(users)}")
    if args.list:
        return
    for repo in tqdm(list(ids), desc="models", unit="model"):
        try:
            files = api.list_repo_files(repo)
            skip = list(SKIP)
            if any(f.endswith(".safetensors") for f in files):
                skip += ["*.bin", "*.pt"]  # prefer safetensors, don't fetch duplicate .bin weights
            path = snapshot_download(repo, ignore_patterns=skip)
            if args.local:
                dest = ROOT / "pretrained" / repo.replace("/", "__")
                snapshot_download(repo, ignore_patterns=skip, local_dir=dest)
                path = dest
            size = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file()) / 1e9
            tqdm.write(f"  ok   {repo}  ({size:.2f} GB) -> {path}")
        except Exception as e:  # keep going; report at the end
            failed.append(repo)
            tqdm.write(f"  FAIL {repo}: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
    if failed:
        print(f"\n{len(failed)} download(s) failed: {', '.join(failed)}")
        print("Check your internet connection / Hugging Face availability and re-run; finished ones are cached.")
        sys.exit(1)
    print("\nAll pretrained models are available.")


if __name__ == "__main__":
    main()
