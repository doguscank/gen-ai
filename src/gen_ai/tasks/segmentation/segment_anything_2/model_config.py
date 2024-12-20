from pathlib import Path
from typing import Optional
from warnings import warn

import torch
from pydantic import Field

from gen_ai.base.model_config import ModelConfig
from gen_ai.configs.defaults import sam2_cfg


class SegmentAnything2ModelConfig(ModelConfig):
    """
    Configuration class for Segment Anything 2.

    Parameters
    ----------
    hf_model_id : str
        The identifier of the model to use.
    model_path : Path, optional
        The path to the model.
    device : str, optional
        The device to run the model on. Defaults to "cuda".
    """

    hf_model_id: Optional[str] = None
    model_path: Optional[Path] = None
    device: str = "cuda"

    torch_dtype: Optional[torch.dtype] = Field(None, init=False)

    def model_post_init(self, __context) -> None:
        if self.hf_model_id is None and self.model_path is None:
            warn(
                "No model provided. Using the default model "
                f"'{sam2_cfg.SAM2_MODEL_ID}'."
            )
            self.hf_model_id = sam2_cfg.SAM2_MODEL_ID

        self.torch_dtype = (
            torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
