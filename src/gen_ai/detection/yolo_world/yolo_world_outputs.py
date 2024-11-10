from typing import List

from pydantic import BaseModel


class BoundingBox(BaseModel):
    """
    Bounding box coordinates.

    Parameters
    ----------
    x1 : float
        Top-left x-coordinate
    y1 : float
        Top-left y-coordinate
    x2 : float
        Bottom-right x-coordinate
    y2 : float
        Bottom-right y-coordinate
    """

    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    """
    Single object detection result.

    Parameters
    ----------
    label : str
        Class name/label of detected object
    confidence : float
        Detection confidence score
    box : BoundingBox
        Bounding box of detected object.
    """

    label: str
    confidence: float
    box: BoundingBox


class Detections(BaseModel):
    """
    Collection of object detections.

    Parameters
    ----------
    detections : List[Detection]
        List of individual detections
    """

    detections: List[Detection]

    def __getitem__(self, idx: int) -> Detection:
        return self.detections[idx]

    def __len__(self) -> int:
        return len(self.detections)
