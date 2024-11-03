from typing import Any, Callable, Dict, List, Optional, Union

import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field


class StableDiffusionInputConfig(BaseModel):
    """
    Configuration class for Stable Diffusion.

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
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    prompt: Optional[Union[str, List[str]]] = None
    negative_prompt: Optional[str] = None
    prompt_embeds: Optional[torch.Tensor] = None
    negative_prompt_embeds: Optional[torch.Tensor] = None
    image: Optional[Image.Image] = None  # for inpainting and img2img
    mask_image: Optional[Image.Image] = None  # for inpainting
    masked_image_latents: Optional[torch.Tensor] = None  # for inpainting
    padding_mask_crop: Optional[int] = None  # for inpainting

    height: int = 512
    width: int = 512
    num_images_per_prompt: int = Field(default=1, ge=1)  # batch size
    num_batches: int = Field(default=1, ge=1)  # number of batches to generate
    eta: float = 0.0
    guidance_rescale: float = 0.0

    num_inference_steps: int = Field(default=20, ge=1)
    guidance_scale: float = Field(default=7.5, ge=0, le=30)
    denoising_strength: float = Field(
        default=0.75, ge=0, le=1
    )  # for img2img and inpainting

    timesteps: Optional[List[int]] = None
    sigmas: Optional[List[float]] = None
    clip_skip: Optional[int] = None

    cross_attention_kwargs: Optional[Dict[str, Any]] = None

    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None
    callback_run_rate: int = 5


def create_text2img_config(
    prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt: Optional[str] = None,
    height: int = 512,
    width: int = 512,
    batch_size: int = 1,
    num_batches: int = 1,
    num_inference_steps: int = 20,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 7.5,
    eta: float = 0.0,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
    clip_skip: Optional[int] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
) -> StableDiffusionInputConfig:
    """This function creates a configuration for text2img tasks."""
    return StableDiffusionInputConfig(
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
        eta=eta,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        cross_attention_kwargs=cross_attention_kwargs,
        guidance_rescale=guidance_rescale,
        clip_skip=clip_skip,
        callback_on_step_end=callback_on_step_end,
    )


def create_inpainting_config(
    prompt: Union[str, List[str]],
    image: Image.Image,
    mask_image: Image.Image,
    masked_image_latents: Optional[torch.Tensor] = None,
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
    eta: float = 0.0,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
    clip_skip: Optional[int] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
) -> StableDiffusionInputConfig:
    """This function creates a configuration for inpainting tasks."""
    return StableDiffusionInputConfig(
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        masked_image_latents=masked_image_latents,
        padding_mask_crop=padding_mask_crop,
        height=height,
        width=width,
        num_batches=num_batches,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
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
    )


def create_img2img_config(
    prompt: Union[str, List[str]],
    image: Image.Image,
    denoising_strength: float = 0.75,
    num_images_per_prompt: int = 1,
    num_batches: int = 1,
    num_inference_steps: int = 20,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 7.5,
    eta: float = 0.0,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
    clip_skip: Optional[int] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
) -> StableDiffusionInputConfig:
    """This function creates a configuration for img2img tasks."""
    return StableDiffusionInputConfig(
        prompt=prompt,
        image=image,
        num_images_per_prompt=num_images_per_prompt,
        num_batches=num_batches,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
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
