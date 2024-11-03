from pathlib import Path
from typing import Dict, List, Optional

import torch
from diffusers import (
    DiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
)
from PIL import Image

from gen_ai.configs import stable_diffusion as sd_config
from gen_ai.constants.task_types import TaskType
from gen_ai.image_gen.stable_diffusion_input_config import (
    StableDiffusionInputConfig,
    create_inpainting_config,
)
from gen_ai.image_gen.stable_diffusion_model_config import StableDiffusionModelConfig
from gen_ai.img_utils import create_spherical_mask_on_center, load_image, save_images
from gen_ai.utils import pathify_strings

PIPELINE_CLS_MAP: Dict[TaskType, DiffusionPipeline] = {
    TaskType.TEXT2IMG: StableDiffusionPipeline,
    TaskType.IMG2IMG: StableDiffusionImg2ImgPipeline,
    TaskType.INPAINTING: StableDiffusionInpaintPipeline,
}


class StableDiffusion:
    def __init__(
        self,
        *,
        config: Optional[StableDiffusionModelConfig] = None,
    ) -> None:
        """
        Initialize the Stable Diffusion model for image generation.

        Parameters
        ----------
        config : Optional[StableDiffusionModelConfig], optional
            The configuration for the Stable Diffusion model, by default None
        """

        self.pipe = None
        self.model_config = config

        if self.model_config is not None:
            if (
                self.model_config.hf_model_id is not None
                or self.model_config.model_path is not None
            ):
                self._load_pipeline(
                    hf_model_id=self.model_config.hf_model_id,
                    model_path=self.model_config.model_path,
                    device=self.model_config.device,
                )

    def _load_pipeline(
        self,
        hf_model_id: Optional[str] = None,
        model_path: Optional[Path] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Load the Stable Diffusion pipeline.

        Parameters
        ----------
        hf_model_id : str, optional
            The HuggingFace model ID to use.
        model_path : Path, optional
            The path to the model.
        device : str, optional
            The device to run the model on.

        Returns
        -------
        None
        """

        if hf_model_id is None and model_path is None:
            raise ValueError("Either hf_model_id or model_path must be provided.")

        if hf_model_id is not None and model_path is not None:
            raise ValueError(
                "Only one of hf_model_id or model_path should be provided."
            )

        model_descriptor = None

        if hf_model_id is not None:
            if self.model_config.hf_model_id == hf_model_id and self.pipe is not None:
                return

            model_descriptor = hf_model_id

        if model_path is not None:
            if self.model_config.model_path == model_path and self.pipe is not None:
                return

            model_descriptor = model_path

        if device is None:
            device = self.model_config.device

        pipeline_cls = PIPELINE_CLS_MAP[self.model_config.task_type]

        self.pipe = pipeline_cls.from_pretrained(
            model_descriptor, torch_dtype=torch.float16, cache_dir=sd_config.CACHE_DIR
        ).to(device)

        if not self.model_config.check_nsfw:
            self.pipe.safety_checker = None

        self.model_config.hf_model_id = hf_model_id
        self.model_config.model_path = model_path
        self.model_config.device = device

    def update_pipeline(self, model_config: StableDiffusionModelConfig) -> None:
        """
        Update the pipeline with a new configuration.

        Parameters
        ----------
        model_config : StableDiffusionModelConfig
            The new configuration.

        Returns
        -------
        None
        """

        self._load_pipeline(
            hf_model_id=model_config.hf_model_id,
            model_path=model_config.model_path,
            device=model_config.device,
        )

        self.model_config = model_config

    @pathify_strings
    def _generate_images_text2img(
        self, config: StableDiffusionInputConfig, output_dir: Optional[Path] = None
    ) -> List[Image.Image]:
        """
        Generate images using the Stable Diffusion model.

        Parameters
        ----------
        config : StableDiffusionConfig
            Configuration for the Stable Diffusion model.

        Returns
        -------
        List[Image.Image]
            A list of generated images.
        """

        if self.model_config.hf_model_id is not None:
            self._load_pipeline(hf_model_id=self.model_config.hf_model_id)
        elif self.model_config.model_path is not None:
            self._load_pipeline(model_path=self.model_config.model_path)
        else:
            self._load_pipeline(hf_model_id=sd_config.TEXT2IMG_MODEL_ID)

        images = []

        for _ in range(config.num_batches):
            pipeline_images = self.pipe(
                prompt=config.prompt,
                negative_prompt=config.negative_prompt,
                height=config.height,
                width=config.width,
                num_images_per_prompt=config.num_images_per_prompt,
                num_inference_steps=config.num_inference_steps,
                timesteps=config.timesteps,
                sigmas=config.sigmas,
                guidance_scale=config.guidance_scale,
                eta=config.eta,
                generator=self.model_config.generator,
                prompt_embeds=config.prompt_embeds,
                negative_prompt_embeds=config.negative_prompt_embeds,
                cross_attention_kwargs=config.cross_attention_kwargs,
                guidance_rescale=config.guidance_rescale,
                clip_skip=config.clip_skip,
                callback_on_step_end=config.callback_on_step_end,
            ).images

            images.extend(pipeline_images)

        if output_dir:
            save_images(images=images, output_dir=output_dir, auto_index=True)

        return images

    @pathify_strings
    def _generate_images_img2img(
        self, config: StableDiffusionInputConfig, output_dir: Optional[Path] = None
    ) -> List[Image.Image]:
        """
        Generate images using the Stable Diffusion model.

        Parameters
        ----------
        config : StableDiffusionConfig
            Configuration for the Stable Diffusion model.

        Returns
        -------
        List[Image.Image]
            A list of generated images.
        """

        if self.model_config.hf_model_id is not None:
            self._load_pipeline(hf_model_id=self.model_config.hf_model_id)
        elif self.model_config.model_path is not None:
            self._load_pipeline(model_path=self.model_config.model_path)
        else:
            self._load_pipeline(hf_model_id=sd_config.IMG2IMG_MODEL_ID)

        images = []

        for _ in range(config.num_batches):
            pipeline_images = self.pipe(
                prompt=config.prompt,
                image=config.image,
                strength=config.denoising_strength,
                num_images_per_prompt=config.num_images_per_prompt,
                num_inference_steps=config.num_inference_steps,
                timesteps=config.timesteps,
                sigmas=config.sigmas,
                guidance_scale=config.guidance_scale,
                eta=config.eta,
                generator=self.model_config.generator,
                prompt_embeds=config.prompt_embeds,
                negative_prompt_embeds=config.negative_prompt_embeds,
                cross_attention_kwargs=config.cross_attention_kwargs,
                guidance_rescale=config.guidance_rescale,
                clip_skip=config.clip_skip,
                callback_on_step_end=config.callback_on_step_end,
            ).images

            images.extend(pipeline_images)

        if output_dir:
            save_images(images=images, output_dir=output_dir, auto_index=True)

        return images

    @pathify_strings
    def _generate_images_inpainting(
        self, config: StableDiffusionInputConfig, output_dir: Optional[Path] = None
    ) -> List[Image.Image]:
        """
        Generate images using the Stable Diffusion model.

        Parameters
        ----------
        config : StableDiffusionConfig
            Configuration for the Stable Diffusion model.

        Returns
        -------
        List[Image.Image]
            A list of generated images.
        """

        if self.model_config.hf_model_id is not None:
            self._load_pipeline(hf_model_id=self.model_config.hf_model_id)
        elif self.model_config.model_path is not None:
            self._load_pipeline(model_path=self.model_config.model_path)
        else:
            self._load_pipeline(hf_model_id=sd_config.INPAINTING_MODEL_ID)

        images = []

        for _ in range(config.num_batches):
            pipeline_images = self.pipe(
                prompt=config.prompt,
                image=config.image,
                mask_image=config.mask_image,
                masked_image_latents=config.masked_image_latents,
                height=config.height,
                width=config.width,
                padding_mask_crop=config.padding_mask_crop,
                strength=config.denoising_strength,
                num_images_per_prompt=config.num_images_per_prompt,
                num_inference_steps=config.num_inference_steps,
                timesteps=config.timesteps,
                sigmas=config.sigmas,
                guidance_scale=config.guidance_scale,
                eta=config.eta,
                generator=self.model_config.generator,
                prompt_embeds=config.prompt_embeds,
                negative_prompt_embeds=config.negative_prompt_embeds,
                cross_attention_kwargs=config.cross_attention_kwargs,
                guidance_rescale=config.guidance_rescale,
                clip_skip=config.clip_skip,
                callback_on_step_end=config.callback_on_step_end,
            ).images

            images.extend(pipeline_images)

        if output_dir:
            save_images(images=images, output_dir=output_dir, auto_index=True)

        return images

    @pathify_strings
    def generate_images(
        self, config: StableDiffusionInputConfig, output_dir: Optional[Path] = None
    ) -> List[Image.Image]:
        """
        Generate images using the Stable Diffusion model.

        Parameters
        ----------
        config : StableDiffusionConfig
            Configuration for the Stable Diffusion model.

        Returns
        -------
        List[Image.Image]
            A list of generated images.
        """

        output_dir.mkdir(parents=True, exist_ok=True)

        if self.model_config.task_type == TaskType.TEXT2IMG:
            images = self._generate_images_text2img(
                config=config, output_dir=output_dir
            )
        elif self.model_config.task_type == TaskType.IMG2IMG:
            images = self._generate_images_img2img(config=config, output_dir=output_dir)
        elif self.model_config.task_type == TaskType.INPAINTING:
            images = self._generate_images_inpainting(
                config=config, output_dir=output_dir
            )
        else:
            raise ValueError(f"Unsupported task type: {self.model_config.task_type}")

        return images


model_cfg = StableDiffusionModelConfig(
    task_type=TaskType.INPAINTING,
    check_nsfw=False,
    seed=None,
)
sd = StableDiffusion(config=model_cfg)

mask = create_spherical_mask_on_center(512, 512, 150)
mask.save("mask.png")

input_config = create_inpainting_config(
    prompt="a little sheep",
    image=load_image("E:\\Scripting Workspace\\Python\\GenAI\\outputs\\image_0.png"),
    mask_image=load_image("mask.png"),
    height=512,
    width=512,
    denoising_strength=0.8,
    num_batches=1,
    num_inference_steps=30,
    guidance_scale=8,
)

output_folder = "outputs"

sd.generate_images(config=input_config, output_dir=output_folder)
