from diffusers import StableDiffusionPipeline, DiffusionPipeline
import torch
from gen_ai.configs import stable_diffusion as sd_config
from pydantic import BaseModel
from typing import Optional, Callable, Dict, List, Any
from PIL import Image
from pathlib import Path
from gen_ai.image_gen.stable_diffusion_config import StableDiffusionConfig
from gen_ai.image_gen.utils import save_images
from gen_ai.utils import pathify_strings


class StableDiffusion:
    def __init__(
        self, model_id: str = sd_config.TEXT2IMG_MODEL_ID, use_cuda: bool = True
    ) -> None:
        """
        Initialize the Stable Diffusion model for image generation.

        Parameters
        ----------
        model_id : str, optional
            The model ID to use, by default sd_config.TEXT2IMG_MODEL_ID.
        use_cuda : bool, optional
            Whether to use the GPU, by default True
        """

        self.model_id = model_id
        self.use_cuda = use_cuda

        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id, torch_dtype=torch.float16
        )

        if self.use_cuda:
            self.pipe = self.pipe.to("cuda")

    @pathify_strings
    def generate_images_text2img(
        self, config: StableDiffusionConfig, output_dir: Optional[Path] = None
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

        images = self.pipe(**config.model_dump()).images

        if output_dir:
            save_images(images=images, output_dir=output_dir, auto_index=True)

        return images


sd = StableDiffusion()
sd_cfg = StableDiffusionConfig(
    prompt="A beautiful sunset over the ocean", width=768, num_images_per_prompt=1
)

sd.generate_images_text2img(sd_cfg, output_dir="outputs")
