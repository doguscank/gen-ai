from gen_ai.tasks.detection.yolo_world import (
    Detections,
    YOLOWorld,
    YOLOWorldModelConfig,
)
from gen_ai.utils.file_ops import load_image

if __name__ == "__main__":
    image_path = "E:\\Scripting Workspace\\Python\\GenAI\\input1.jpg"
    image = load_image(image_path)

    yolo_world_model_cfg = YOLOWorldModelConfig(
        device="cuda",
        classes=["person"],
    )

    yolo_world_model = YOLOWorld(config=yolo_world_model_cfg)

    yolo_world_input = image

    yolo_world_output: Detections = yolo_world_model(yolo_world_input)

    print(yolo_world_output)
