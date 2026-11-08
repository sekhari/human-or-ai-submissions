"""Download the exact public GPT-2 revision used by robust_boost, once."""
import json
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REPO = "openai-community/gpt2"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


def main():
    path = snapshot_download(
        repo_id=REPO, revision=REVISION, token=False,
        allow_patterns=["config.json", "model.safetensors", "vocab.json", "merges.txt",
                        "tokenizer.json", "tokenizer_config.json"],
        cache_dir=ROOT / ".cache/huggingface")
    (ROOT / ".cache/gpt2_model.json").write_text(json.dumps(
        {"repo": REPO, "revision": REVISION, "path": str(Path(path).relative_to(ROOT))}, indent=2))
    print(path)


if __name__ == "__main__":
    main()
