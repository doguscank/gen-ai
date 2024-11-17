from pathlib import Path

from gen_ai.constants.diffusion_noise_scheduler_types import SchedulerTypes
from gen_ai.constants.image_gen_task_types import ImageGenTaskTypes
from gen_ai.constants.inpainting_configuration_types import (
    InpaintingBlendingTypes,
    InpaintingPostProcessTypes,
    InpaintingPreProcessTypes,
)
from gen_ai.image_gen.clip.prompt_weighting import process
from gen_ai.image_gen.stable_diffusion_15.stable_diffusion import StableDiffusion
from gen_ai.image_gen.stable_diffusion_15.stable_diffusion_input_config import (
    StableDiffusionInputConfig,
)
from gen_ai.image_gen.stable_diffusion_15.stable_diffusion_model_config import (
    StableDiffusionModelConfig,
)
from gen_ai.utils import file_ops, img_utils, measure_time

if __name__ == "__main__":
    sd_model_cfg = StableDiffusionModelConfig(
        model_path=Path(
            "E:\\Scripting Workspace\\Python\\GenAI\\gen-ai\\models\\dreamshaper_8Inpainting.safetensors"
        ),
        device="cuda",
        task_type=ImageGenTaskTypes.INPAINTING,
        seed=42,
    )

    image_path = "E:\\Scripting Workspace\\Python\\GenAI\\input1.jpg"

    image = file_ops.load_image(image_path)

    mask_path = "E:\\Scripting Workspace\\Python\\GenAI\\result_mask1.png"

    mask = file_ops.load_image(mask_path)
    mask = img_utils.preprocess_mask(mask)
    mask = img_utils.pad_mask(mask, padding=30, iterations=1)

    prompt = "RAW photo of a man wearing a (red:0.2) and (white:1.8) (fur coat:1.5)"
    negative_prompt = "bad quality, low quality"

    with measure_time("Stable Diffusion Model Initialization"):
        sd_model = StableDiffusion(config=sd_model_cfg)

    with measure_time("Prompt Processing"):
        prompt_embeds = process(
            prompt, tokenizer=sd_model.tokenizer, model=sd_model.text_encoder
        ).to(device=sd_model.device)
        negative_prompt_embeds = process(
            negative_prompt, tokenizer=sd_model.tokenizer, model=sd_model.text_encoder
        ).to(device=sd_model.device)

    sd_input = StableDiffusionInputConfig.create_inpainting_config(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        # prompt=prompt,
        # negative_prompt=negative_prompt,
        image=image,
        mask_image=mask,
        height=512,
        width=512,
        preprocess_type=InpaintingPreProcessTypes.CROP_AND_RESIZE,
        postprocess_type=InpaintingPostProcessTypes.BLEND,
        blending_type=InpaintingBlendingTypes.POISSON_BLENDING,
        denoising_strength=0.7,
        guidance_scale=7.5,
        num_batches=1,
        num_inference_steps=30,
        scheduler_type=SchedulerTypes.DPMPP_2M_KARRAS,
    )

    with measure_time("Stable Diffusion Prediction"):
        sd_output = sd_model.generate_images(
            config=sd_input,
            output_dir=Path("E:\\Scripting Workspace\\Python\\GenAI\\output"),
        )
