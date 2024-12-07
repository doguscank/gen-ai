from pathlib import Path

from diffusers import (
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
)

from gen_ai.constants.task_types.image_gen_task_types import ImageGenTaskTypes

TEXT2IMG_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
TEXT2IMG_REFINER_MODEL_ID = "stabilityai/stable-diffusion-xl-refiner-1.0"
IMG2IMG_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
IMG2IMG_REFINER_MODEL_ID = "stabilityai/stable-diffusion-xl-refiner-1.0"
INPAINTING_MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

SDXL_VAE_FP16_FIX_MODEL_ID = "madebyollin/sdxl-vae-fp16-fix"

CACHE_DIR = (
    Path(__file__).parent.parent.parent.parent.parent.parent
    / "models"
    / "diffusers_cache"
    / "stable_diffusion_xl"
)

TASK_TYPE_MODEL_MAP = {
    ImageGenTaskTypes.TEXT2IMG: TEXT2IMG_REFINER_MODEL_ID,
    ImageGenTaskTypes.IMG2IMG: IMG2IMG_REFINER_MODEL_ID,
    ImageGenTaskTypes.INPAINTING: INPAINTING_MODEL_ID,
}

PIPELINE_CLS_MAP = {
    ImageGenTaskTypes.TEXT2IMG: StableDiffusionXLPipeline,
    ImageGenTaskTypes.IMG2IMG: StableDiffusionXLImg2ImgPipeline,
    ImageGenTaskTypes.INPAINTING: StableDiffusionXLInpaintPipeline,
}

CACHE_DIR.mkdir(parents=True, exist_ok=True)
