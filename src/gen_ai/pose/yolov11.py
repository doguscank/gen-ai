from typing import List, Tuple, Optional
from ultralytics import YOLO
from gen_ai.pose.pose import Pose


class YOLOConfig:
    def __init__(self, model_name: str, model_path: str, device: str = "cuda") -> None:
        """
        Initialize the YOLOConfig.

        Parameters
        ----------
        model_name : str
            Name of the model.
        model_path : str
            Path to the model file.
        device : str, optional
            Device to run the model on (default is 'cuda').
        """

        self.model_name = model_name
        self.model_path = model_path
        self.device = device


class YOLOModel:
    def __init__(self, config: YOLOConfig) -> None:
        """
        Initialize the YOLOModel with the given configuration.

        Parameters
        ----------
        config : YOLOConfig
            YOLOConfig object containing model configuration.
        """

        self.config = config
        self.model = YOLO(config.model_path).to(config.device)

    def update_model(self, config: YOLOConfig) -> None:
        """
        Update the model with the given configuration.

        Parameters
        ----------
        config : YOLOConfig
            YOLOConfig object containing model configuration.
        """

        self.config = config
        self.model = YOLO(config.model_path).to(config.device)

    def detect(self, image) -> List[Pose]:
        """
        Detect objects in the given image.

        Parameters
        ----------
        image :
            Image in which to detect objects.

        Returns
        -------
        List[Pose]
            List of Pose objects representing detected objects.
        """

        results = self.model(image)
        return results
