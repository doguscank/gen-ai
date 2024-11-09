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
from safetensors.torch import load_file as load_safetensor_file

from gen_ai.configs import stable_diffusion as sd_config
from gen_ai.constants.image_gen_task_types import ImageGenTaskTypes
from gen_ai.image_gen.inpainting_utils import postprocess_outputs, preprocess_inputs
from gen_ai.image_gen.scheduler_utils import get_scheduler
from gen_ai.image_gen.stable_diffusion_input_config import StableDiffusionInputConfig
from gen_ai.image_gen.stable_diffusion_model_config import StableDiffusionModelConfig
from gen_ai.img_utils import save_images
from gen_ai.utils import pathify_strings

PIPELINE_CLS_MAP: Dict[ImageGenTaskTypes, DiffusionPipeline] = {
    ImageGenTaskTypes.TEXT2IMG: StableDiffusionPipeline,
    ImageGenTaskTypes.IMG2IMG: StableDiffusionImg2ImgPipeline,
    ImageGenTaskTypes.INPAINTING: StableDiffusionInpaintPipeline,
}

PIPELINE_MODEL_MAP: Dict[ImageGenTaskTypes, str] = {
    ImageGenTaskTypes.TEXT2IMG: sd_config.TEXT2IMG_MODEL_ID,
    ImageGenTaskTypes.IMG2IMG: sd_config.IMG2IMG_MODEL_ID,
    ImageGenTaskTypes.INPAINTING: sd_config.INPAINTING_MODEL_ID,
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

    def _load_finetuned_weights(self, model_path: Path) -> None:
        """
        Load the finetuned weights for the Stable Diffusion model.

        Parameters
        ----------
        model_path : Path
            The path to the finetuned weights.

        Returns
        -------
        None
        """

        if self.pipe is None:
            raise ValueError("Pipeline must be loaded before loading weights.")

        state_dict = load_safetensor_file(model_path)
        self.pipe.unet.load_state_dict(state_dict)
        self.pipe.to(self.model_config.device)

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
        is_finetuned = False

        if hf_model_id is not None:
            if self.model_config.hf_model_id == hf_model_id and self.pipe is not None:
                return

            model_descriptor = hf_model_id

        if model_path is not None:
            if self.model_config.model_path == model_path and self.pipe is not None:
                return

            if "diffusers_cache" not in model_path.parts:
                is_finetuned = True
            model_descriptor = model_path

        if device is None:
            device = self.model_config.device

        pipeline_cls = PIPELINE_CLS_MAP[self.model_config.task_type]

        if is_finetuned:
            self.pipe = pipeline_cls.from_single_file(
                model_descriptor,
                cache_dir=sd_config.CACHE_DIR,
                local_files_only=True,
            ).to(device)
        else:
            self.pipe = pipeline_cls.from_pretrained(
                model_descriptor,
                torch_dtype=torch.float16,
                cache_dir=sd_config.CACHE_DIR,
                local_files_only=True,
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
            image, mask_image = preprocess_inputs(
                image=config.image,
                mask=config.mask_image,
                pre_process_type=config.preprocess_type,
                output_width=config.width,
                output_height=config.height,
            )

            pipeline_images = self.pipe(
                prompt=config.prompt,
                image=image,
                mask_image=mask_image,
                masked_image_latents=config.masked_image_latents,
                latents=config.latents,
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

            pipeline_images = [
                postprocess_outputs(
                    image=config.image,
                    mask=config.mask_image,
                    inpainted_image=inpainted_image,
                    pre_process_type=config.preprocess_type,
                    post_process_type=config.postprocess_type,
                    blending_type=config.blending_type,
                )
                for inpainted_image in pipeline_images
            ]

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

        if config.scheduler_type is not None:
            self.pipe.scheduler = get_scheduler(scheduler_type=config.scheduler_type)

        if self.model_config.task_type == ImageGenTaskTypes.TEXT2IMG:
            images = self._generate_images_text2img(
                config=config, output_dir=output_dir
            )
        elif self.model_config.task_type == ImageGenTaskTypes.IMG2IMG:
            images = self._generate_images_img2img(config=config, output_dir=output_dir)
        elif self.model_config.task_type == ImageGenTaskTypes.INPAINTING:
            images = self._generate_images_inpainting(
                config=config, output_dir=output_dir
            )
        else:
            raise ValueError(f"Unsupported task type: {self.model_config.task_type}")

        return images
