from pathlib import Path

FLUX_SCHNELL_MODEL_ID = "black-forest-labs/FLUX.1-schnell"
CACHE_DIR = (
    Path(__file__).parent.parent.parent.parent.parent.parent
    / "models"
    / "diffusers_cache"
    / "flux_1"
)

CACHE_DIR.mkdir(parents=True, exist_ok=True)
