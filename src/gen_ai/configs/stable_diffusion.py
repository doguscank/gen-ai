from pathlib import Path

TEXT2IMG_MODEL_ID = "sd-legacy/stable-diffusion-v1-5"
IMG2IMG_MODEL_ID = "sd-legacy/stable-diffusion-v1-5"
INPAINTING_MODEL_ID = "sd-legacy/stable-diffusion-inpainting"
CACHE_DIR = (
    Path(__file__).parent.parent / "models" / "diffusers_cache" / "stable_diffusion"
)

CACHE_DIR.mkdir(parents=True, exist_ok=True)
