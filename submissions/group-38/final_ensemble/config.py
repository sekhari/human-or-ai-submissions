import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
OUT = Path(os.environ.get("EXPERIMENT_DIR", str(ROOT / "results"))).expanduser().resolve()
OUT.mkdir(parents=True, exist_ok=True)
DATA_ZIP = Path(os.environ.get("COMPETITION_ZIP", "cornell-tech-cs-5785-combined-2026-ai-vs-human-text.zip")).expanduser().resolve()
PRIOR_RESULTS = Path(os.environ.get("PRIOR_RESULTS", str(OUT / "prior"))).expanduser().resolve()
DEST = OUT / "deliverables"
DEST.mkdir(parents=True, exist_ok=True)
