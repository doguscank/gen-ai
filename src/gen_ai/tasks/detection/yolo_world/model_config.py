from pathlib import Path
from typing import List, Optional, Union

from gen_ai.base.model_config import ModelConfig


class YOLOWorldModelConfig(ModelConfig):
    """
    Configuration for YOLOWorld model.

    Parameters
    ----------
    model_name : Optional[str]
        Name of the YOLOWorld model.
    model_path : Optional[Path]
        Path to model weights.
    device : str
        Device to run inference on. Defaults to "cuda".
    classes : Union[str, List[str]]
        List of class names to detect.
    """

    model_name: Optional[str] = None
    model_path: Optional[Path] = None
    device: str = "cuda"
    classes: Union[str, List[str]]

    def model_post_init(self, __context) -> None:
        if isinstance(self.classes, str):
            if "," in self.classes:
                self.classes = self.classes.split(",")
            else:
                self.classes = [self.classes]

        if self.model_name is None and self.model_path is None:
            self.model_name = "yolov8s-world"
        if self.model_name is not None:
            if "." in self.model_name:  # remove file extension
                self.model_name = self.model_name.split(".")[0]
