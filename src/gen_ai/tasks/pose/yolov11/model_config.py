from pathlib import Path
from typing import Optional
from warnings import warn

from gen_ai.base.model_config import ModelConfig
from gen_ai.configs.defaults import yolov11_cfg


class YOLOv11ModelConfig(ModelConfig):
    """
    Configuration class for YOLOModel.

    Parameters
    ----------
    model_name : Optional[str], optional
        The name of the model. Defaults to None.
    model_path : Optional[Path], optional
        The path to the model. Defaults to None.
    device : str
        The device to use for the model
    """

    model_name: Optional[str] = None
    model_path: Optional[Path] = None
    device: str = "cuda"

    def model_post_init(self, __context) -> None:
        if self.model_name is None and self.model_path is None:
            warn(
                "No model provided. Using the default model.\n"
                f"Model name: {yolov11_cfg.DEFAULT_MODEL_NAME}"
            )
