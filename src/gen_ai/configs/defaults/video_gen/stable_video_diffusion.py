from pathlib import Path

SVD_MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt"
CACHE_DIR = (
    Path(__file__).parent.parent.parent.parent.parent.parent
    / "models"
    / "diffusers_cache"
    / "stable_video_diffusion"
)

CACHE_DIR.mkdir(parents=True, exist_ok=True)
