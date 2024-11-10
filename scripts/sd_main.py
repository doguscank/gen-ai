from pathlib import Path

from gen_ai.constants.diffusion_noise_scheduler_types import SchedulerTypes
from gen_ai.constants.image_gen_task_types import ImageGenTaskTypes
from gen_ai.constants.inpainting_configuration_types import (
    InpaintingPostProcessTypes,
    InpaintingPreProcessTypes,
)
from gen_ai.image_gen.stable_diffusion_15.stable_diffusion import StableDiffusion
from gen_ai.image_gen.stable_diffusion_15.stable_diffusion_input_config import (
    StableDiffusionInputConfig,
)
from gen_ai.image_gen.stable_diffusion_15.stable_diffusion_model_config import (
    StableDiffusionModelConfig,
)
from gen_ai.utils import img_utils, measure_time

if __name__ == "__main__":
    sd_model_cfg = StableDiffusionModelConfig(
        model_path=Path(
            "E:\\Scripting Workspace\\Python\\GenAI\\gen-ai\\models\\dreamshaper_8Inpainting.safetensors"
        ),
        device="cuda",
        task_type=ImageGenTaskTypes.INPAINTING,
        seed=1,
    )

    image_path = "E:\\Scripting Workspace\\Python\\GenAI\\input1.jpg"

    image = img_utils.load_image(image_path)

    mask_path = "E:\\Scripting Workspace\\Python\\GenAI\\result_mask1.png"

    mask = img_utils.load_image(mask_path)
    mask = img_utils.pad_mask(mask, padding=30, iterations=1)

    sd_input = StableDiffusionInputConfig.create_inpainting_config(
        prompt="RAW photo of a man wearing a red and white fur coat",
        negative_prompt="bad quality, low quality",
        image=image,
        mask_image=mask,
        height=512,
        width=512,
        preprocess_type=InpaintingPreProcessTypes.CROP_AND_RESIZE,
        postprocess_type=InpaintingPostProcessTypes.DIRECT_REPLACE,
        denoising_strength=0.7,
        guidance_scale=7.5,
        num_batches=1,
        num_inference_steps=30,
        scheduler_type=SchedulerTypes.DPMPP_2M_KARRAS,
    )

    with measure_time("Stable Diffusion Model Initialization"):
        sd_model = StableDiffusion(config=sd_model_cfg)

    with measure_time("Stable Diffusion Prediction"):
        sd_output = sd_model.generate_images(
            config=sd_input,
            output_dir=Path("E:\\Scripting Workspace\\Python\\GenAI\\output"),
        )
