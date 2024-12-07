import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
from gen_ai.utils.text_conditioning_utils import fix_dual_conditioning_inputs


class StableDiffusionXLInput(Input):
    """
    Input class for Stable Diffusion XL model.

    Parameters
    ----------
    prompt : Union[str, List[str]], optional
        The primary text prompt(s) guiding image generation. If not defined, `prompt_embeds` must be provided.
    prompt_2 : Union[str, List[str]], optional
        Secondary text prompt(s) used by the secondary text encoder. Defaults to the same value as `prompt` if not provided.
    negative_prompt : Union[str, List[str]], optional
        Text prompt(s) that negatively influence image generation. Ignored when `guidance_scale` < 1.
    negative_prompt_2 : Union[str, List[str]], optional
        Secondary negative text prompt(s) for the secondary text encoder. Defaults to `negative_prompt` if not provided.
    prompt_embeds : torch.Tensor, optional
        Pre-computed text embeddings to guide image generation. Useful for advanced configurations, such as prompt weighting.
    negative_prompt_embeds : torch.Tensor, optional
        Pre-computed embeddings for negative prompts, enhancing control over undesired outputs.
    pooled_prompt_embeds : torch.Tensor, optional
        Pooled embeddings derived from `prompt_embeds`. Required when using `prompt_embeds`.
    negative_pooled_prompt_embeds : torch.Tensor, optional
        Pooled embeddings derived from `negative_prompt_embeds`. Required when using `negative_prompt_embeds`.
    image : PIL.Image.Image, optional
        Input image for img2img transformations or inpainting tasks.
    mask_image : PIL.Image.Image, optional
        Mask specifying areas for inpainting. White pixels indicate regions to inpaint.
    masked_image_latents : torch.Tensor, optional
        Latent representation of the masked image for inpainting purposes.
    padding_mask_crop : int, optional
        Size of the crop applied to the padding mask during inpainting.
    latents : torch.Tensor, optional
        Pre-generated latent tensors representing image states for generation.
    height : int, optional
        Height of the generated image in pixels. Must be a multiple of 8. Defaults to 1024.
    width : int, optional
        Width of the generated image in pixels. Must be a multiple of 8. Defaults to 1024.
    original_size : Tuple[int, int], optional
        Original size used for SDXL conditioning.
    crops_coords_top_left : Tuple[int, int], optional
        Top-left coordinates for cropping during conditioning. Defaults to (0, 0).
    target_size : Tuple[int, int], optional
        Desired output size used for SDXL conditioning.
    negative_original_size : Tuple[int, int], optional
        Original size for negative prompt conditioning.
    negative_crops_coords_top_left : Tuple[int, int], optional
        Top-left coordinates for cropping negative conditioning. Defaults to (0, 0).
    negative_target_size : Tuple[int, int], optional
        Desired target size for negative conditioning.
    num_images_per_prompt : int, optional
        Number of images generated per prompt. Defaults to 1.
    num_batches : int, optional
        Number of batches for image generation. Defaults to 1.
    eta : float, optional
        Parameter for DDIM scheduler. Defaults to 0.0.
    guidance_rescale : float, optional
        Factor to address overexposure issues during guidance. Defaults to 0.0.
    num_inference_steps : int, optional
        Number of denoising steps to perform. Higher values generally yield higher-quality images. Defaults to 50.
    guidance_scale : float, optional
        Strength of guidance for aligning generation with text prompts. Higher values result in closer adherence to prompts. Defaults to 5.0.
    denoising_strength : float, optional
        Degree of denoising for img2img. Must be between 0.0 and 1.0. Defaults to 0.75.
    denoising_start : float, optional
        Fraction of denoising steps to skip initially. Defaults to None.
    denoising_end : float, optional
        Fraction of denoising steps to terminate prematurely. Defaults to None.
    scheduler_type : SchedulerTypes, optional
        Type of noise scheduler to use. If not provided, defaults to an automatic selection.
    timesteps : List[int], optional
        Custom timesteps for the noise schedule. Must be in descending order.
    sigmas : List[float], optional
        Custom sigma values for the noise schedule.
    clip_skip : int, optional
        Number of layers skipped in the CLIP text encoder for computing prompt embeddings.
    ip_adapter_image : PIL.Image.Image, optional
        Image input for the inpainting adapter.
    ip_adapter_image_embeds : List[torch.Tensor], optional
        Embeddings for the inpainting adapter.
    cross_attention_kwargs : Dict[str, Any], optional
        Additional arguments for cross-attention configurations.
    aesthetic_score : float, optional
        Aesthetic score influencing positive text conditioning. Defaults to 6.0.
    negative_aesthetic_score : float, optional
        Aesthetic score influencing negative text conditioning. Defaults to 2.5.
    callback_on_step_end : Callable[[int, int, Dict], None], optional
        Callback executed at the end of each denoising step.
    callback_run_rate : int, optional
        Interval at which the callback is executed. Defaults to 5.
    preprocess_type : InpaintingPreProcessTypes, optional
        Type of preprocessing applied to inpainting tasks.
    postprocess_type : InpaintingPostProcessTypes, optional
        Type of postprocessing applied to inpainting tasks.
    blending_type : InpaintingBlendingTypes, optional
        Blending method used for inpainting.
    """

    prompt: Optional[Union[str, List[str]]] = None
    prompt_2: Optional[Union[str, List[str]]] = None
    negative_prompt: Optional[Union[str, List[str]]] = None
    negative_prompt_2: Optional[Union[str, List[str]]] = None
    prompt_embeds: Optional[torch.Tensor] = None
    negative_prompt_embeds: Optional[torch.Tensor] = None
    pooled_prompt_embeds: Optional[torch.Tensor] = None
    negative_pooled_prompt_embeds: Optional[torch.Tensor] = None
    image: Optional[Image.Image] = None  # for inpainting and img2img
    mask_image: Optional[Image.Image] = None  # for inpainting
    masked_image_latents: Optional[torch.Tensor] = None  # for inpainting
    padding_mask_crop: Optional[int] = None  # for inpainting
    latents: Optional[torch.Tensor] = None  # for inpainting

    height: int = Field(default=1024, multiple_of=8)
    width: int = Field(default=1024, multiple_of=8)
    original_size: Optional[Tuple[int, int]] = None
    crops_coords_top_left: Tuple[int, int] = (0, 0)
    target_size: Optional[Tuple[int, int]] = None
    negative_original_size: Optional[Tuple[int, int]] = None
    negative_crops_coords_top_left: Tuple[int, int] = (0, 0)
    negative_target_size: Optional[Tuple[int, int]] = None

    num_images_per_prompt: int = Field(default=1, ge=1)  # batch size
    num_batches: int = Field(default=1, ge=1)  # number of batches to generate
    eta: float = 0.0
    guidance_rescale: float = 0.0

    num_inference_steps: int = Field(default=50, ge=1)
    guidance_scale: float = 5.0
    denoising_strength: Optional[float] = Field(
        default=0.75, ge=0.0, le=1.0
    )  # for img2img and inpainting
    denoising_start: Optional[float] = None
    denoising_end: Optional[float] = None
    scheduler_type: Optional[SchedulerTypes] = None

    timesteps: Optional[List[int]] = None
    sigmas: Optional[List[float]] = None
    clip_skip: Optional[int] = None

    ip_adapter_image: Optional[Image.Image] = None
    ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None

    cross_attention_kwargs: Optional[Dict[str, Any]] = None

    aesthetic_score: float = 6.0
    negative_aesthetic_score: float = 2.5

    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None
    callback_run_rate: int = 5

    preprocess_type: Optional[InpaintingPreProcessTypes] = None
    postprocess_type: Optional[InpaintingPostProcessTypes] = None
    blending_type: Optional[InpaintingBlendingTypes] = None

    def model_post_init(self, __context) -> None:
        (
            self.prompt,
            self.prompt_2,
            self.negative_prompt,
            self.negative_prompt_2,
            self.prompt_embeds,
            self.negative_prompt_embeds,
        ) = fix_dual_conditioning_inputs(
            prompt=self.prompt,
            prompt_2=self.prompt_2,
            negative_prompt=self.negative_prompt,
            negative_prompt_2=self.negative_prompt_2,
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
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        batch_size: int = 1,
        eta: float = 0.0,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_run_rate: int = 5,
    ) -> "StableDiffusionXLInput":
        """This function creates an input object for text2img task."""
        return cls(
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            sigmas=sigmas,
            denoising_end=denoising_end,
            guidance_scale=guidance_scale,
            num_images_per_prompt=batch_size,
            eta=eta,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            guidance_rescale=guidance_rescale,
            crops_coords_top_left=crops_coords_top_left,
            target_size=target_size,
            negative_original_size=negative_original_size,
            negative_crops_coords_top_left=negative_crops_coords_top_left,
            negative_target_size=negative_target_size,
            clip_skip=clip_skip,
            callback_on_step_end=callback_on_step_end,
            callback_run_rate=callback_run_rate,
        )

    @classmethod
    def create_inpainting_input(
        cls,
        image: Image.Image,
        mask_image: Image.Image,
        preprocess_type: Optional[InpaintingPreProcessTypes] = None,
        postprocess_type: Optional[InpaintingPostProcessTypes] = None,
        prompt: Optional[Union[str, List[str]]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        blending_type: Optional[InpaintingBlendingTypes] = None,
        masked_image_latents: Optional[torch.Tensor] = None,
        height: int = 1024,
        width: int = 1024,
        padding_mask_crop: Optional[int] = None,
        denoising_strength: Optional[float] = 0.75,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        denoising_start: Optional[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 7.5,
        batch_size: int = 1,
        eta: float = 0.0,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_run_rate: int = 5,
    ) -> "StableDiffusionXLInput":
        return cls(
            image=image,
            mask_image=mask_image,
            preprocess_type=preprocess_type,
            postprocess_type=postprocess_type,
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            blending_type=blending_type,
            masked_image_latents=masked_image_latents,
            height=height,
            width=width,
            padding_mask_crop=padding_mask_crop,
            denoising_strength=denoising_strength,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            sigmas=sigmas,
            denoising_start=denoising_start,
            denoising_end=denoising_end,
            guidance_scale=guidance_scale,
            num_images_per_prompt=batch_size,
            eta=eta,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            guidance_rescale=guidance_rescale,
            original_size=original_size,
            crops_coords_top_left=crops_coords_top_left,
            target_size=target_size,
            negative_original_size=negative_original_size,
            negative_crops_coords_top_left=negative_crops_coords_top_left,
            negative_target_size=negative_target_size,
            aesthetic_score=aesthetic_score,
            negative_aesthetic_score=negative_aesthetic_score,
            clip_skip=clip_skip,
            callback_on_step_end=callback_on_step_end,
            callback_run_rate=callback_run_rate,
        )

    @classmethod
    def create_img2img_input(
        cls,
        image: Image.Image,
        prompt: Optional[Union[str, List[str]]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        denoising_strength: Optional[float] = 0.3,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        denoising_start: Optional[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        batch_size: int = 1,
        eta: float = 0.0,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_run_rate: int = 5,
    ) -> "StableDiffusionXLInput":
        return cls(
            image=image,
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            denoising_strength=denoising_strength,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            sigmas=sigmas,
            denoising_start=denoising_start,
            denoising_end=denoising_end,
            guidance_scale=guidance_scale,
            num_images_per_prompt=batch_size,
            eta=eta,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            guidance_rescale=guidance_rescale,
            original_size=original_size,
            crops_coords_top_left=crops_coords_top_left,
            target_size=target_size,
            negative_original_size=negative_original_size,
            negative_crops_coords_top_left=negative_crops_coords_top_left,
            negative_target_size=negative_target_size,
            aesthetic_score=aesthetic_score,
            negative_aesthetic_score=negative_aesthetic_score,
            clip_skip=clip_skip,
            callback_on_step_end=callback_on_step_end,
            callback_run_rate=callback_run_rate,
        )
