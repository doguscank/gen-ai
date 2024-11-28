from pathlib import Path
from typing import List, Optional

from ultralytics import YOLOWorld as YOLOWorldBase

from gen_ai.base.model import Model
from gen_ai.tasks.detection.yolo_world.input import YOLOWorldInput
from gen_ai.tasks.detection.yolo_world.model_config import YOLOWorldModelConfig
from gen_ai.tasks.detection.yolo_world.output import Detections
from gen_ai.tasks.detection.yolo_world.output_parsers import parse_yolo_world_output


class YOLOWorld(Model):
    """
    YOLOWorld object detection model interface.
    """

    def __init__(self, *, config: Optional[YOLOWorldModelConfig] = None) -> None:
        """
        Initialize YOLOWorld model.

        Parameters
        ----------
        config : YOLOWorldModelConfig, optional
            Model configuration
        """

        self.config = config
        self.model = None

        if self.config is not None:
            self._load_model(
                model_path=self.config.model_path,
                model_name=self.config.model_name,
                device=self.config.device,
                class_names=self.config.classes,
            )

    def check_model_ready(self) -> bool:
        """
        Check if model is ready for inference.

        Returns
        -------
        bool
            True if model is ready, False otherwise
        """

        return self.model is not None

    def _load_model(
        self,
        model_path: Optional[Path] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        class_names: Optional[List[str]] = None,
    ) -> None:
        """
        Load model from path.

        Parameters
        ----------
        model_path : Path, optional
            Path to model file
        model_name : str, optional
            Name of the model
        device : str, optional
            Device to use for inference
        class_names : List[str], optional
            Class names.

        Returns
        -------
        None
        """

        if model_path is None and model_name is None:
            raise ValueError("Either model_path or model_name must be provided")

        if model_path is not None:
            if self.config and self.model and self.config.model_path == model_path:
                return
            self.model = YOLOWorldBase(model_path)
        else:
            if self.config and self.model and self.config.model_name == model_name:
                return
            self.model = YOLOWorldBase(model_name)

        if class_names:
            self.model.set_classes(class_names)
        else:
            self.model.set_classes(self.config.classes)

        if device is not None:
            self.model.to(device)

    def update_model(self, config: YOLOWorldModelConfig) -> None:
        """
        Update model configuration.

        Parameters
        ----------
        config : YOLOWorldModelConfig
            New configuration

        Returns
        -------
        None
        """

        self._load_model(
            model_path=config.model_path,
            model_name=config.model_name,
            device=config.device,
            class_names=config.classes,
        )

        self.config = config

    def __call__(self, input_data: YOLOWorldInput) -> Detections:
        """
        Detect objects on the given input.

        Parameters
        ----------
        input_data : YOLOWorldInput
            YOLOWorld input data

        Returns
        -------
        Detections
            Detection results
        """
        if not self.check_model_ready():
            raise ValueError("Model not ready for inference")

        results = self.model.predict(source=input_data.image)
        detections = parse_yolo_world_output(results, self.config.classes)

        return detections
