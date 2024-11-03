from diffusers import StableDiffusionPipeline, DiffusionPipeline
import torch
from gen_ai.configs import stable_diffusion as sd_config
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Callable, Dict, List, Any
from PIL import Image
from pathlib import Path
from gen_ai.constants.task_types import TaskType


class StableDiffusionConfig(BaseModel):
    """
    Configuration class for Stable Diffusion.

    Parameters
    ----------
    model_id : str
        The identifier of the model to use.
    device : str, optional
        The device to run the model on. Defaults to "cuda".
    task_type : TaskType, optional
        The type of task to perform. Defaults to TaskType.TXT2IMG.
    prompt : str, oaptional
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
        Number of layers to be skipped from CLIP while computaing the prompt embeddings.
    cross_attention_kwargs : dict, optional
        A kwargs dictionary passed along to the AttentionProcessor.
    check_nsfw : bool, optional
        Whether to check for NSFW content. Defaults to False.
    callback_on_step_end : Callable[[int, int, Dict], None], optional
        A callback function to be called at the end of each step.
    callback_run_rate : int, optional
        The rate at which the callback function is called. Defaults to 5.
    seed : int, optional
        The seed for random number generation. Defaults to -1.
    generator : torch.Generator, optional
        The generator for random number generation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_id: str
    device: str = "cuda"
    task_type: TaskType = TaskType.TXT2IMG

    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    prompt_embeds: Optional[torch.Tensor] = None
    negative_prompt_embeds: Optional[torch.Tensor] = None
    image: Optional[Image.Image] = None  # for inpainting and img2img
    mask: Optional[Image.Image] = None  # for inpainting

    prompt_embeds: Optional[torch.Tensor] = None
    neg_prompt_embeds: Optional[torch.Tensor] = None

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

    check_nsfw: bool = False

    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None
    callback_run_rate: int = 5

    seed: Optional[int] = -1
    generator: Optional[torch.Generator] = None

    def model_post_init(self, __context) -> "StableDiffusionConfig":
        # Check if CUDA is available
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

        # Initialize the generator
        if self.seed == -1:
            self.seed = torch.seed()
        if self.generator is None:
            self.generator = torch.Generator(device=self.device)
            self.generator.manual_seed(self.seed)

        return self
