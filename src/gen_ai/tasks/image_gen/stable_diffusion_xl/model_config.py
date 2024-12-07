from pathlib import Path
from typing import Optional
from warnings import warn

import torch

from gen_ai.base.model_config import ModelConfig
from gen_ai.configs.defaults import sd_cfg
from gen_ai.constants.task_types.image_gen_task_types import ImageGenTaskTypes


class StableDiffusionXLModelConfig(ModelConfig):
    """
    Configuration class for Stable Diffusion XL.

    Parameters
    ----------
    hf_model_id : str
        The identifier of the model to use.
    model_path : Path, optional
        The path to the model file. Defaults to None.
    device : str, optional
        The device to run the model on. Defaults to "cuda".
    task_type : ImageGenTaskTypes, optional
        The type of task to perform. Defaults to ImageGenTaskTypes.TEXT2IMG.
    check_nsfw : bool, optional
        Whether to check for NSFW content. Defaults to False.
    optimize : bool, optional
        Whether to optimize the model. Defaults to True.
    seed : int, optional
        The seed for random number generation. Defaults to -1.
    generator : torch.Generator, optional
        The generator for random number generation.
    """

    hf_model_id: Optional[str] = None
    model_path: Optional[Path] = None
    device: str = "cuda"
    task_type: ImageGenTaskTypes = ImageGenTaskTypes.TEXT2IMG

    check_nsfw: bool = False
    optimize: bool = True

    seed: Optional[int] = -1
    generator: Optional[torch.Generator] = None

    def model_post_init(self, __context) -> None:
        if self.hf_model_id is None and self.model_path is None:
            warn(
                "No model provided. Using the default model.\n"
                f"Task type: {self.task_type}\n"
                f"Model ID: {sd_cfg.TASK_TYPE_MODEL_MAP[self.task_type]}"
            )
            self.hf_model_id = sd_cfg.TASK_TYPE_MODEL_MAP[self.task_type]

        # Initialize the generator
        if self.seed == -1 or self.seed is None:
            self.seed = torch.seed()
        if self.generator is None:
            self.generator = torch.Generator(device=self.device)
            self.generator.manual_seed(self.seed)
