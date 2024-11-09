from pathlib import Path

SAM2_MODEL_ID = "facebook/sam2.1-hiera-large"
CACHE_DIR = (
    Path(__file__).parent.parent.parent.parent / "models" / "diffusers_cache" / "sam2"
)

CACHE_DIR.mkdir(parents=True, exist_ok=True)
