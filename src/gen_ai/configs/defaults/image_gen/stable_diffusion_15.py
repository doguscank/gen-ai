from pathlib import Path

from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
)

from gen_ai.constants.task_types.image_gen_task_types import ImageGenTaskTypes

TEXT2IMG_MODEL_ID = "sd-legacy/stable-diffusion-v1-5"
IMG2IMG_MODEL_ID = "sd-legacy/stable-diffusion-v1-5"
INPAINTING_MODEL_ID = "sd-legacy/stable-diffusion-inpainting"
CACHE_DIR = (
    Path(__file__).parent.parent.parent.parent.parent.parent
    / "models"
    / "diffusers_cache"
    / "stable_diffusion"
)

TASK_TYPE_MODEL_MAP = {
    ImageGenTaskTypes.TEXT2IMG: TEXT2IMG_MODEL_ID,
    ImageGenTaskTypes.IMG2IMG: IMG2IMG_MODEL_ID,
    ImageGenTaskTypes.INPAINTING: INPAINTING_MODEL_ID,
}

PIPELINE_CLS_MAP = {
    ImageGenTaskTypes.TEXT2IMG: StableDiffusionPipeline,
    ImageGenTaskTypes.IMG2IMG: StableDiffusionImg2ImgPipeline,
    ImageGenTaskTypes.INPAINTING: StableDiffusionInpaintPipeline,
}


CACHE_DIR.mkdir(parents=True, exist_ok=True)
