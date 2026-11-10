import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
DATA_ZIP = Path(os.environ.get("COMPETITION_ZIP", "cornell-tech-cs-5785-combined-2026-ai-vs-human-text.zip")).expanduser().resolve()
OUT = Path(os.environ.get("MODEL_OUTPUT_DIR", str(ROOT / "results"))).expanduser().resolve()
OUT.mkdir(parents=True, exist_ok=True)
