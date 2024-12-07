from pathlib import Path
from typing import Optional, Union

import torch
from diffusers import (
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
)
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from gen_ai.base.model import Model
from gen_ai.configs.defaults.image_gen import stable_diffusion_xl as sdxl_config
from gen_ai.constants.task_types.image_gen_task_types import ImageGenTaskTypes
from gen_ai.logger import logger
from gen_ai.tasks.image_gen.stable_diffusion_xl.input import StableDiffusionXLInput
from gen_ai.tasks.image_gen.stable_diffusion_xl.model_config import (
    StableDiffusionXLModelConfig,
)
from gen_ai.tasks.image_gen.stable_diffusion_xl.output import StableDiffusionXLOutput
from gen_ai.tasks.image_gen.utils.inpainting_utils import (
    postprocess_outputs,
    preprocess_inputs,
)
from gen_ai.tasks.image_gen.utils.scheduler_utils import get_scheduler
from gen_ai.utils import check_if_hf_cache_exists, pathify_strings
from gen_ai.utils.file_ops import save_images

_PipelineType = Union[
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
]


class StableDiffusionXL(Model):
    def __init__(
        self,
        *,
        config: Optional[StableDiffusionXLModelConfig] = None,
    ) -> None:
        """
        Initialize the Stable Diffusion XL model for image generation.

        Parameters
        ----------
        config : Optional[StableDiffusionXLModelConfig], optional
            The configuration for the Stable Diffusion XL model, by default None
        """

        self._pipeline: Optional[_PipelineType] = None
        self._model_config = config

        if self._model_config is not None:
            if (
                self._model_config.hf_model_id is not None
                or self._model_config.model_path is not None
            ):
                logger.info(
                    f"Loading the Stable Diffusion XL pipeline with config: {self._model_config}"
                )

                self._load_pipeline(
                    hf_model_id=self._model_config.hf_model_id,
                    model_path=self._model_config.model_path,
                    device=self._model_config.device,
                    optimize=self._model_config.optimize,
                )

    @property
    def tokenizer(self) -> CLIPTokenizer:
        return self._pipeline.tokenizer

    @property
    def tokenizer_2(self) -> CLIPTokenizer:
        return self._pipeline.tokenizer_2

    @property
    def text_encoder(self) -> CLIPTextModel:
        return self._pipeline.text_encoder

    @property
    def text_encoder_2(self) -> CLIPTextModelWithProjection:
        return self._pipeline.text_encoder_2

    @property
    def device(self) -> str:
        return self._pipeline.device

    @property
    def model_config(self) -> StableDiffusionXLModelConfig:
        return self._model_config

    @property
    def pipeline(self) -> _PipelineType:
        return self._pipeline

    def check_model_ready(self) -> bool:
        """
        Check if the model is ready.

        Returns
        -------
        bool
            True if the model is ready, False otherwise.
        """

        return self._pipeline is not None

    def optimize(self) -> None:
        """
        Optimize the model.

        Returns
        -------
        None
        """

        self._pipeline.enable_model_cpu_offload()
        self._pipeline.enable_xformers_memory_efficient_attention()

    def _load_pipeline(
        self,
        hf_model_id: Optional[str] = None,
        model_path: Optional[Path] = None,
        device: Optional[str] = None,
        optimize: bool = False,
    ) -> None:
        """
        Load the Stable Diffusion XL pipeline.

        Parameters
        ----------
        hf_model_id : str, optional
            The HuggingFace model ID to use.
        model_path : Path, optional
            The path to the model.
        device : str, optional
            The device to run the model on.
        optimize : bool, optional
            Whether to optimize the model.

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
            if (
                self._model_config.hf_model_id == hf_model_id
                and self._pipeline is not None
            ):
                return

            model_descriptor = hf_model_id

        if model_path is not None:
            if (
                self._model_config.model_path == model_path
                and self._pipeline is not None
            ):
                return

            if "diffusers_cache" not in model_path.parts:
                is_finetuned = True
            model_descriptor = str(model_path)

        if device is None:
            device = self._model_config.device

        pipeline_cls = sdxl_config.PIPELINE_CLS_MAP[self._model_config.task_type]

        if is_finetuned:
            self._pipeline = pipeline_cls.from_single_file(
                model_descriptor,
                cache_dir=sdxl_config.CACHE_DIR,
                local_files_only=True,
            ).to(device)
        else:
            self._pipeline = pipeline_cls.from_pretrained(
                model_descriptor,
                torch_dtype=torch.float16,
                cache_dir=sdxl_config.CACHE_DIR,
                local_files_only=check_if_hf_cache_exists(
                    cache_dir=sdxl_config.CACHE_DIR,
                    model_id=model_descriptor,
                ),
            ).to(device)

        if not self._model_config.check_nsfw:
            self._pipeline.safety_checker = None

        self._model_config.hf_model_id = hf_model_id
        self._model_config.model_path = model_path
        self._model_config.device = device

        if optimize:
            self.optimize()

    def update_pipeline(self, model_config: StableDiffusionXLModelConfig) -> None:
        """
        Update the pipeline with a new configuration.

        Parameters
        ----------
        model_config : StableDiffusionXLModelConfig
            The new configuration.

        Returns
        -------
        None
        """

        self._load_pipeline(
            hf_model_id=self._model_config.hf_model_id,
            model_path=self._model_config.model_path,
            device=self._model_config.device,
        )

        self._model_config = model_config

    def _load_model_hard_set(self) -> None:
        """
        Load the model with the hard set configuration. This is used when the model
        configuration is set by the user directly instead of using the loading
        functions.

        Returns
        -------
        None
        """

        if self._model_config.hf_model_id is not None:
            self._load_pipeline(hf_model_id=self._model_config.hf_model_id)
        elif self._model_config.model_path is not None:
            self._load_pipeline(model_path=self._model_config.model_path)
        else:
            self._load_pipeline(
                hf_model_id=sdxl_config.TASK_TYPE_MODEL_MAP[
                    self._model_config.task_type
                ],
            )

    @pathify_strings
    def _generate_images_text2img(
        self, input: StableDiffusionXLInput, output_dir: Optional[Path] = None
    ) -> StableDiffusionXLOutput:
        """
        Generate images using the Stable Diffusion XL model.

        Parameters
        ----------
        input : StableDiffusionXLInput
            The input for the model.
        output_dir : Path, optional
            The output directory to save the images, by default None

        Returns
        -------
        StableDiffusionXLOutput
            Model output.
        """

        self._load_model_hard_set()

        images = []

        for _ in range(input.num_batches):
            pipeline_images = self._pipeline(
                prompt=input.prompt,
                prompt_2=input.prompt_2,
                height=input.height,
                width=input.width,
                num_inference_steps=input.num_inference_steps,
                timesteps=input.timesteps,
                sigmas=input.sigmas,
                denoising_end=input.denoising_end,
                guidance_scale=input.guidance_scale,
                negative_prompt=input.negative_prompt,
                negative_prompt_2=input.negative_prompt_2,
                num_images_per_prompt=input.num_images_per_prompt,
                eta=input.eta,
                generator=self._model_config.generator,
                latents=input.latents,
                prompt_embeds=input.prompt_embeds,
                negative_prompt_embeds=input.negative_prompt_embeds,
                pooled_prompt_embeds=input.pooled_prompt_embeds,
                negative_pooled_prompt_embeds=input.negative_pooled_prompt_embeds,
                ip_adapter_image=input.ip_adapter_image,
                ip_adapter_image_embeds=input.ip_adapter_image_embeds,
                cross_attention_kwargs=input.cross_attention_kwargs,
                guidance_rescale=input.guidance_rescale,
                original_size=input.original_size,
                crops_coords_top_left=input.crops_coords_top_left,
                target_size=input.target_size,
                negative_original_size=input.negative_original_size,
                negative_crops_coords_top_left=input.negative_crops_coords_top_left,
                negative_target_size=input.negative_target_size,
                clip_skip=input.clip_skip,
                callback_on_step_end=input.callback_on_step_end,
            ).images

            images.extend(pipeline_images)

        if output_dir:
            save_images(images=images, output_dir=output_dir, auto_index=True)

        return StableDiffusionXLOutput(images=images)

    @pathify_strings
    def _generate_images_img2img(
        self, input: StableDiffusionXLInput, output_dir: Optional[Path] = None
    ) -> StableDiffusionXLOutput:
        """
        Generate images using the Stable Diffusion XL model.

        Parameters
        ----------
        input : StableDiffusionXLInput
            The input for the model.
        output_dir : Path, optional
            The output directory to save the images, by default None

        Returns
        -------
        StableDiffusionXLOutput
            Model output.
        """

        self._load_model_hard_set()

        images = []

        for _ in range(input.num_batches):
            pipeline_images = self._pipeline(
                prompt=input.prompt,
                prompt_2=input.prompt_2,
                image=input.image,
                strength=input.denoising_strength,
                num_inference_steps=input.num_inference_steps,
                timesteps=input.timesteps,
                sigmas=input.sigmas,
                denoising_start=input.denoising_start,
                denoising_end=input.denoising_end,
                guidance_scale=input.guidance_scale,
                negative_prompt=input.negative_prompt,
                negative_prompt_2=input.negative_prompt_2,
                num_images_per_prompt=input.num_images_per_prompt,
                eta=input.eta,
                generator=self._model_config.generator,
                latents=input.latents,
                prompt_embeds=input.prompt_embeds,
                negative_prompt_embeds=input.negative_prompt_embeds,
                pooled_prompt_embeds=input.pooled_prompt_embeds,
                negative_pooled_prompt_embeds=input.negative_pooled_prompt_embeds,
                ip_adapter_image=input.ip_adapter_image,
                ip_adapter_image_embeds=input.ip_adapter_image_embeds,
                cross_attention_kwargs=input.cross_attention_kwargs,
                guidance_rescale=input.guidance_rescale,
                original_size=input.original_size,
                crops_coords_top_left=input.crops_coords_top_left,
                target_size=input.target_size,
                negative_original_size=input.negative_original_size,
                negative_crops_coords_top_left=input.negative_crops_coords_top_left,
                negative_target_size=input.negative_target_size,
                aesthetic_score=input.aesthetic_score,
                negative_aesthetic_score=input.negative_aesthetic_score,
                clip_skip=input.clip_skip,
                callback_on_step_end=input.callback_on_step_end,
            ).images

            images.extend(pipeline_images)

        if output_dir:
            save_images(images=images, output_dir=output_dir, auto_index=True)

        return StableDiffusionXLOutput(images=images)

    @pathify_strings
    def _generate_images_inpainting(
        self, input: StableDiffusionXLInput, output_dir: Optional[Path] = None
    ) -> StableDiffusionXLOutput:
        """
        Generate images using the Stable Diffusion XL model.

        Parameters
        ----------
        input : StableDiffusionXLInput
            The input for the model.
        output_dir : Path, optional
            The output directory to save the images, by default None

        Returns
        -------
        StableDiffusionXLOutput
            Model output.
        """

        self._load_model_hard_set()

        images = []

        for _ in range(input.num_batches):
            image, mask_image = preprocess_inputs(
                image=input.image,
                mask=input.mask_image,
                pre_process_type=input.preprocess_type,
                output_width=input.width,
                output_height=input.height,
            )

            pipeline_images = self._pipeline(
                prompt=input.prompt,
                prompt_2=input.prompt_2,
                image=image,
                mask_image=mask_image,
                masked_image_latents=input.masked_image_latents,
                height=input.height,
                width=input.width,
                padding_mask_crop=input.padding_mask_crop,
                strength=input.denoising_strength,
                num_inference_steps=input.num_inference_steps,
                timesteps=input.timesteps,
                sigmas=input.sigmas,
                denoising_start=input.denoising_start,
                denoising_end=input.denoising_end,
                guidance_scale=input.guidance_scale,
                negative_prompt=input.negative_prompt,
                negative_prompt_2=input.negative_prompt_2,
                num_images_per_prompt=input.num_images_per_prompt,
                eta=input.eta,
                generator=self._model_config.generator,
                latents=input.latents,
                prompt_embeds=input.prompt_embeds,
                negative_prompt_embeds=input.negative_prompt_embeds,
                pooled_prompt_embeds=input.pooled_prompt_embeds,
                negative_pooled_prompt_embeds=input.negative_pooled_prompt_embeds,
                ip_adapter_image=input.ip_adapter_image,
                ip_adapter_image_embeds=input.ip_adapter_image_embeds,
                cross_attention_kwargs=input.cross_attention_kwargs,
                guidance_rescale=input.guidance_rescale,
                original_size=input.original_size,
                crops_coords_top_left=input.crops_coords_top_left,
                target_size=input.target_size,
                negative_original_size=input.negative_original_size,
                negative_crops_coords_top_left=input.negative_crops_coords_top_left,
                negative_target_size=input.negative_target_size,
                aesthetic_score=input.aesthetic_score,
                negative_aesthetic_score=input.negative_aesthetic_score,
                clip_skip=input.clip_skip,
                callback_on_step_end=input.callback_on_step_end,
            ).images

            pipeline_images = [
                postprocess_outputs(
                    image=input.image,
                    mask=input.mask_image,
                    inpainted_image=inpainted_image,
                    pre_process_type=input.preprocess_type,
                    post_process_type=input.postprocess_type,
                    blending_type=input.blending_type,
                )
                for inpainted_image in pipeline_images
            ]

            images.extend(pipeline_images)

        if output_dir:
            save_images(images=images, output_dir=output_dir, auto_index=True)

        return StableDiffusionXLOutput(images=images)

    @pathify_strings
    def __call__(
        self,
        input: StableDiffusionXLInput,
        output_dir: Optional[Path] = None,
    ) -> StableDiffusionXLOutput:
        """
        Generate images using the Stable Diffusion XL model.

        Parameters
        ----------
        input : StableDiffusionXLInput
            The input for the model.
        output_dir : Path, optional
            The output directory to save the images, by default None

        Returns
        -------
        StableDiffusionXLOutput
            Model output.
        """

        if not self.check_model_ready():
            logger.error("Model not ready. Please load the model first.")
            return []

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        if input.scheduler_type is not None:
            self._pipeline.scheduler = get_scheduler(
                scheduler_type=input.scheduler_type
            )

        if self._model_config.task_type == ImageGenTaskTypes.TEXT2IMG:
            images = self._generate_images_text2img(input=input, output_dir=output_dir)
        elif self._model_config.task_type == ImageGenTaskTypes.IMG2IMG:
            images = self._generate_images_img2img(input=input, output_dir=output_dir)
        elif self._model_config.task_type == ImageGenTaskTypes.INPAINTING:
            images = self._generate_images_inpainting(
                input=input, output_dir=output_dir
            )
        else:
            raise ValueError(f"Unsupported task type: {self._model_config.task_type}")

        return images
