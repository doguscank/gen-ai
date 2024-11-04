from gen_ai.image_gen.stable_diffusion_input_config import (
    StableDiffusionInputConfig,
)
from gen_ai.image_gen.stable_diffusion_model_config import StableDiffusionModelConfig
from gen_ai.constants.image_gen_task_types import ImageGenTaskTypes
from gen_ai.image_gen.stable_diffusion import StableDiffusion
from gen_ai.img_utils import load_image, create_spherical_mask_on_center

if __name__ == "__main__":
    model_cfg = StableDiffusionModelConfig(
        task_type=ImageGenTaskTypes.TEXT2IMG,
        check_nsfw=False,
        seed=None,
    )
    sd = StableDiffusion(config=model_cfg)

    # input_config = StableDiffusionInputConfig.create_inpainting_config(
    #     prompt="a ship on the sea, floating in the water",
    #     image=load_image(
    #         "E:\\Scripting Workspace\\Python\\GenAI\\outputs\\image_1.png"
    #     ),
    #     mask_image=create_spherical_mask_on_center(512, 512, 150),
    #     height=512,
    #     width=512,
    #     denoising_strength=0.8,
    #     num_batches=2,
    #     num_inference_steps=20,
    #     guidance_scale=10,
    # )
    input_config = StableDiffusionInputConfig.create_text2img_config(
        prompt="a ship on the sea, floating in the water",
        height=512,
        width=512,
        batch_size=1,
        num_batches=1,
        num_inference_steps=20,
        guidance_scale=7.5,
    )

    output_folder = "outputs"

    sd.generate_images(config=input_config, output_dir=output_folder)
