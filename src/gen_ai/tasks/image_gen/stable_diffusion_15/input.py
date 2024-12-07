import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from PIL import Image
from pydantic import Field

from gen_ai.base.input import Input
from gen_ai.constants.diffusion_noise_scheduler_types import SchedulerTypes
from gen_ai.constants.inpainting_configuration_types import (
    InpaintingBlendingTypes,
    InpaintingPostProcessTypes,
    InpaintingPreProcessTypes,
)
from gen_ai.utils.text_conditioning_utils import fix_conditioning_inputs


class StableDiffusionInput(Input):
    """
    Input class for Stable Diffusion.

    Parameters
    ----------
    prompt : Union[str, List[str]], optional
        The prompt to guide image generation.
    negative_prompt : str, optional
        The negative prompt to guide what to not include in image generation.
    prompt_embeds : torch.Tensor, optional
        Pre-generated text embeddings.
    negative_prompt_embeds : torch.Tensor, optional
        Pre-generated negative text embeddings.
    image : PIL.Image.Image, optional
        The image to use for inpainting and img2img tasks.
    mask : PIL.Image.Image, optional
        The mask to use for inpainting tasks.
    height : int, optional
        The height in pixels of the generated image. Defaults to 512.
    width : int, optional
        The width in pixels of the generated image. Defaults to 512.
    num_images_per_prompt : int, optional
        The number of images to generate per prompt. Defaults to 1.
    num_batches : int, optional
        The number of batches to generate. Defaults to 1.
    eta : float, optional
        Corresponds to parameter eta (η) from the DDIM paper. Defaults to 0.0.
    guidance_rescale : float, optional
        Guidance rescale factor. Defaults to 0.0.
    num_inference_steps : int, optional
        The number of denoising steps. Defaults to 20.
    guidance_scale : float, optional
        A higher guidance scale value encourages the model to generate images closely
        linked to the text prompt. Defaults to 7.5.
    denoising_strength : float, optional
        The strength of denoising for img2img and inpainting tasks. Defaults to 0.75.
    scheduler_type : SchedulerTypes, optional
        The scheduler type to use for the denoising process.
    timesteps : list of int, optional
        Custom timesteps to use for the denoising process.
    sigmas : list of float, optional
        Custom sigmas to use for the denoising process.
    clip_skip : int, optional
        Number of layers to be skipped from CLIP while computing the prompt embeddings.
    cross_attention_kwargs : dict, optional
        A kwargs dictionary passed along to the AttentionProcessor.
    callback_on_step_end : Callable[[int, int, Dict], optional
        A callback function to be called at the end of each step.
    callback_run_rate : int, optional
        The rate at which the callback function is called. Defaults to 5.
    preprocess_type : InpaintingPreProcessTypes, optional
        The pre-processing type for inpainting tasks.
    postprocess_type : InpaintingPostProcessTypes, optional
        The post-processing type for inpainting tasks.
    blending_type : InpaintingBlendingTypes, optional
        The blending type for inpainting tasks.
    """

    prompt: Optional[Union[str, List[str]]] = None
    negative_prompt: Optional[Union[str, List[str]]] = None
    prompt_embeds: Optional[torch.Tensor] = None
    negative_prompt_embeds: Optional[torch.Tensor] = None
    image: Optional[Image.Image] = None  # for inpainting and img2img
    mask_image: Optional[Image.Image] = None  # for inpainting
    masked_image_latents: Optional[torch.Tensor] = None  # for inpainting
    padding_mask_crop: Optional[int] = None  # for inpainting
    latents: Optional[torch.Tensor] = None  # for inpainting

    height: int = 512
    width: int = 512
    num_images_per_prompt: int = Field(default=1, ge=1)  # batch size
    num_batches: int = Field(default=1, ge=1)  # number of batches to generate
    eta: float = 0.0
    guidance_rescale: float = 0.0

    num_inference_steps: int = Field(default=20, ge=1)
    guidance_scale: float = Field(default=7.5, ge=0, le=30)
    denoising_strength: float = Field(
        default=0.75, ge=0.0, le=1.0
    )  # for img2img and inpainting
    scheduler_type: Optional[SchedulerTypes] = None

    timesteps: Optional[List[int]] = None
    sigmas: Optional[List[float]] = None
    clip_skip: Optional[int] = None

    cross_attention_kwargs: Optional[Dict[str, Any]] = None

    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None
    callback_run_rate: int = 5

    preprocess_type: Optional[InpaintingPreProcessTypes] = None
    postprocess_type: Optional[InpaintingPostProcessTypes] = None
    blending_type: Optional[InpaintingBlendingTypes] = None

    def model_post_init(self, __context) -> None:
        (
            self.prompt,
            self.negative_prompt,
            self.prompt_embeds,
            self.negative_prompt_embeds,
        ) = fix_conditioning_inputs(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            prompt_embeds=self.prompt_embeds,
            negative_prompt_embeds=self.negative_prompt_embeds,
        )

        if self.prompt is None and self.prompt_embeds is None:
            warnings.warn(
                "Prompt is not provided."
                "The model will generate images without any guidance."
            )
            self.prompt = ""

        if self.negative_prompt is None and self.negative_prompt_embeds is None:
            warnings.warn(
                "Negative prompt is not provided."
                "The model will generate images without any negative guidance."
            )
            self.negative_prompt = ""

    @classmethod
    def create_text2img_input(
        cls,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 512,
        width: int = 512,
        batch_size: int = 1,
        num_batches: int = 1,
        num_inference_steps: int = 20,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.5,
        scheduler_type: Optional[SchedulerTypes] = None,
        eta: float = 0.0,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    ) -> "StableDiffusionInput":
        """This function creates an input object for text2img task."""
        return cls(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_images_per_prompt=batch_size,
            num_batches=num_batches,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            sigmas=sigmas,
            guidance_scale=guidance_scale,
            scheduler_type=scheduler_type,
            eta=eta,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            guidance_rescale=guidance_rescale,
            clip_skip=clip_skip,
            callback_on_step_end=callback_on_step_end,
        )

    @classmethod
    def create_inpainting_input(
        cls,
        image: Image.Image,
        mask_image: Image.Image,
        preprocess_type: InpaintingPreProcessTypes,
        postprocess_type: InpaintingPostProcessTypes,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        blending_type: Optional[InpaintingBlendingTypes] = None,
        masked_image_latents: Optional[torch.Tensor] = None,
        latents: Optional[torch.Tensor] = None,
        height: int = 512,
        width: int = 512,
        padding_mask_crop: Optional[int] = None,
        denoising_strength: float = 0.75,
        num_batches: int = 1,
        num_images_per_prompt: int = 1,
        num_inference_steps: int = 20,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.5,
        scheduler_type: Optional[SchedulerTypes] = None,
        eta: float = 0.0,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    ) -> "StableDiffusionInput":
        """This function creates an input object for inpainting task."""
        return cls(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask_image,
            masked_image_latents=masked_image_latents,
            latents=latents,
            padding_mask_crop=padding_mask_crop,
            height=height,
            width=width,
            num_batches=num_batches,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            scheduler_type=scheduler_type,
            eta=eta,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            guidance_rescale=guidance_rescale,
            clip_skip=clip_skip,
            callback_on_step_end=callback_on_step_end,
            denoising_strength=denoising_strength,
            timesteps=timesteps,
            sigmas=sigmas,
            num_images_per_prompt=num_images_per_prompt,
            preprocess_type=preprocess_type,
            postprocess_type=postprocess_type,
            blending_type=blending_type,
        )

    @classmethod
    def create_img2img_input(
        cls,
        image: Image.Image,
        denoising_strength: float = 0.75,
        num_images_per_prompt: int = 1,
        num_batches: int = 1,
        num_inference_steps: int = 20,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.5,
        scheduler_type: Optional[SchedulerTypes] = None,
        eta: float = 0.0,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    ) -> "StableDiffusionInput":
        """This function creates an input object for img2img task."""
        return cls(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            num_images_per_prompt=num_images_per_prompt,
            num_batches=num_batches,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            scheduler_type=scheduler_type,
            denoising_strength=denoising_strength,
            eta=eta,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            guidance_rescale=guidance_rescale,
            clip_skip=clip_skip,
            callback_on_step_end=callback_on_step_end,
            timesteps=timesteps,
            sigmas=sigmas,
        )
