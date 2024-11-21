from typing import Optional

from PIL import Image
from ultralytics import YOLO

from gen_ai.base.model import Model
from gen_ai.tasks.pose.pose import Poses
from gen_ai.tasks.pose.yolov11.model_config import YOLOv11ModelConfig
from gen_ai.tasks.pose.yolov11.output_parsers import parse_yolov11_pose_output


class YOLOv11_Pose(Model):
    def __init__(self, *, config: Optional[YOLOv11ModelConfig] = None) -> None:
        """
        Initialize the YOLOModel with the given configuration.

        Parameters
        ----------
        config : YOLOv11ModelConfig
            YOLOv11ModelConfig object containing model configuration.
        """

        self.config = config
        self.model = None

        if self.config is not None:
            self.model = YOLO(config.model_path).to(config.device)

    def check_model_ready(self) -> bool:
        """
        Check if the model is ready for inference.

        Returns
        -------
        bool
            True if the model is ready for inference, False otherwise.
        """

        return self.model is not None

    def update_model(self, config: YOLOv11ModelConfig) -> None:
        """
        Update the model with the given configuration.

        Parameters
        ----------
        config : YOLOv11ModelConfig
            YOLOv11ModelConfig object containing model configuration.
        """

        if (
            self.config.model_path == config.model_path
            and self.config.model_path is not None
        ) or (
            self.config.model_name == config.model_name
            and self.config.model_name is not None
        ):
            return

        self.config = config
        self.model = YOLO(config.model_path).to(config.device)

    def __call__(self, image: Image.Image) -> Poses:
        """
        Detect objects in the given image.

        Parameters
        ----------
        image : Image.Image
            Image in which to detect objects.

        Returns
        -------
        Poses
            List of Pose objects representing detected objects.
        """

        if not self.check_model_ready():
            raise ValueError("Model is not ready for inference.")

        results = self.model.predict(image)
        poses = parse_yolov11_pose_output(results)

        return poses
