from pathlib import Path

from gen_ai.constants.diffusion_noise_scheduler_types import SchedulerTypes
from gen_ai.constants.inpainting_configuration_types import (
    InpaintingBlendingTypes,
    InpaintingPostProcessTypes,
    InpaintingPreProcessTypes,
)
from gen_ai.constants.task_types.image_gen_task_types import ImageGenTaskTypes
from gen_ai.tasks.image_gen.stable_diffusion_15.input import StableDiffusionInput
from gen_ai.tasks.image_gen.stable_diffusion_15.model_config import (
    StableDiffusionModelConfig,
)
from gen_ai.tasks.image_gen.stable_diffusion_15.stable_diffusion import StableDiffusion
from gen_ai.utils import file_ops, img_utils, measure_time

if __name__ == "__main__":
    sd_model_cfg = StableDiffusionModelConfig(
        model_path=Path(
            "E:\\Scripting Workspace\\Python\\GenAI\\gen-ai\\models\\dreamshaper_8Inpainting.safetensors"
        ),
        lora_dir=Path("E:\\Scripting Workspace\\Python\\GenAI\\gen-ai\\lora\\sd_15"),
        device="cuda",
        task_type=ImageGenTaskTypes.INPAINTING,
        seed=None,
    )

    image_path = "E:\\Scripting Workspace\\Python\\GenAI\\input1.jpg"

    image = file_ops.load_image(image_path)

    mask_path = "E:\\Scripting Workspace\\Python\\GenAI\\result_mask1.png"

    mask = file_ops.load_image(mask_path)
    mask = img_utils.preprocess_mask(mask)
    mask = img_utils.pad_mask(mask, padding=30, iterations=1)

    sd_input = StableDiffusionInput.create_inpainting_config(
        prompt="RAW photo of a man wearing a red and white fur coat",
        negative_prompt="bad quality, low quality",
        image=image,
        mask_image=mask,
        height=512,
        width=512,
        preprocess_type=InpaintingPreProcessTypes.CROP_AND_RESIZE,
        postprocess_type=InpaintingPostProcessTypes.BLEND,
        blending_type=InpaintingBlendingTypes.POISSON_BLENDING,
        denoising_strength=0.7,
        guidance_scale=7.5,
        num_batches=4,
        num_inference_steps=30,
        scheduler_type=SchedulerTypes.DPMPP_2M_KARRAS,
    )

    with measure_time("Stable Diffusion Model Initialization"):
        sd_model = StableDiffusion(config=sd_model_cfg)

    with measure_time("Stable Diffusion Prediction"):
        sd_output = sd_model(
            config=sd_input,
            output_dir=Path("E:\\Scripting Workspace\\Python\\GenAI\\output"),
        )
