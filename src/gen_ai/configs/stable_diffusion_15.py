from pathlib import Path

from gen_ai.constants.image_gen_task_types import ImageGenTaskTypes

TEXT2IMG_MODEL_ID = "sd-legacy/stable-diffusion-v1-5"
IMG2IMG_MODEL_ID = "sd-legacy/stable-diffusion-v1-5"
INPAINTING_MODEL_ID = "sd-legacy/stable-diffusion-inpainting"
CACHE_DIR = (
    Path(__file__).parent.parent.parent.parent
    / "models"
    / "diffusers_cache"
    / "stable_diffusion"
)

TASK_TYPE_MODEL_MAP = {
    ImageGenTaskTypes.TEXT2IMG: TEXT2IMG_MODEL_ID,
    ImageGenTaskTypes.IMG2IMG: IMG2IMG_MODEL_ID,
    ImageGenTaskTypes.INPAINTING: INPAINTING_MODEL_ID,
}

CACHE_DIR.mkdir(parents=True, exist_ok=True)
