from diffusers import StableDiffusionPipeline, DiffusionPipeline
import torch
from gen_ai.configs import stable_diffusion as sd_config
from pydantic import BaseModel, ConfigDict
from typing import Optional, Callable, Dict, List, Any
from PIL import Image
from pathlib import Path


class StableDiffusionConfig(BaseModel):
    """
    Configuration class for Stable Diffusion.

    Parameters
    ----------
    prompt : str, optional
        The prompt to guide image generation.
    neg_prompt : str, optional
        The negative prompt to guide what to not include in image generation.
    prompt_embeds : torch.Tensor, optional
        Pre-generated text embeddings.
    neg_prompt_embeds : torch.Tensor, optional
        Pre-generated negative text embeddings.
    height : int, optional
        The height in pixels of the generated image. Defaults to 512.
    width : int, optional
        The width in pixels of the generated image. Defaults to 512.
    num_images_per_prompt : int, optional
        The number of images to generate per prompt. Defaults to 1.
    eta : float, optional
        Corresponds to parameter eta (η) from the DDIM paper. Defaults to 0.0.
    guidance_rescale : float, optional
        Guidance rescale factor. Defaults to 0.0.
    num_inference_steps : int, optional
        The number of denoising steps. Defaults to 20.
    guidance_scale : float, optional
        A higher guidance scale value encourages the model to generate images closely
        linked to the text prompt. Defaults to 7.5.
    timesteps : list of int, optional
        Custom timesteps to use for the denoising process.
    sigmas : list of float, optional
        Custom sigmas to use for the denoising process.
    clip_skip : int, optional
        Number of layers to be skipped from CLIP while computing the prompt embeddings.
    cross_attention_kwargs : dict, optional
        A kwargs dictionary passed along to the AttentionProcessor.
    check_nsfw : bool, optional
        Whether to check for NSFW content. Defaults to False.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None

    prompt_embeds: Optional[torch.Tensor] = None
    neg_prompt_embeds: Optional[torch.Tensor] = None

    height: int = 512
    width: int = 512
    num_images_per_prompt: int = 1
    eta: float = 0.0
    guidance_rescale: float = 0.0

    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    timesteps: Optional[List[int]] = None
    sigmas: Optional[List[float]] = None
    clip_skip: Optional[int] = None

    cross_attention_kwargs: Optional[Dict[str, Any]] = None

    check_nsfw: bool = False

    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None
    callback_run_rate: int = 5
