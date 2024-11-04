from pathlib import Path

FLORENCE_MODEL_ID = "microsoft/Florence-2-large-ft"
CACHE_DIR = Path(__file__).parent.parent / "models" / "diffusers_cache" / "florence2"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
