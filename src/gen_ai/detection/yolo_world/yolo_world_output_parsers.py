from typing import List

from ultralytics.engine.results import Results

from gen_ai.detection.yolo_world.yolo_world_outputs import (
    BoundingBox,
    Detection,
    Detections,
)


def parse_yolo_world_output(results: Results, class_names: List[str]) -> Detections:
    """
    Parse YOLO World detection results.

    Parameters
    ----------
    results : Results
        Raw detection results from model
    class_names : List[str]
        List of class names used for detection

    Returns
    -------
    Detections
        Parsed detection results
    """

    boxes = results[0].boxes
    detections = []

    for i in range(len(boxes)):
        box = boxes.xyxy[i].cpu().numpy()
        conf = float(boxes.conf[i].cpu().numpy())
        cls_id = int(boxes.cls[i].cpu().numpy())
        label = class_names[cls_id]

        detections.append(
            Detection(
                label=label,
                confidence=conf,
                box=BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
            )
        )

    return Detections(detections=detections)
