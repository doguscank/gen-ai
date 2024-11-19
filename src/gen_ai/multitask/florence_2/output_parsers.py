import re
from typing import Dict, List, Sequence, Union

from gen_ai.constants.florence_2_task_types import Florence2TaskTypes
from gen_ai.multitask.florence_2.outputs import (
    OCR,
    BoundingBox,
    BoundingBoxes,
    Caption,
    OpenVocabularyDetection,
    Point,
    Polygon,
    Polygons,
    QuadBox,
    QuadBoxes,
)

CAPTION_TASKS = [
    Florence2TaskTypes.CAPTION,
    Florence2TaskTypes.DETAILED_CAPTION,
    Florence2TaskTypes.MORE_DETAILED_CAPTION,
]
BBOX_TASKS = [
    Florence2TaskTypes.OBJECT_DETECTION,
    Florence2TaskTypes.DENSE_REGION_CAPTION,
    Florence2TaskTypes.REGIONAL_PROPOSAL,
    Florence2TaskTypes.CAPTION_TO_PHRASE_GROUNDING,
]
POLYGON_TASKS = [
    Florence2TaskTypes.REFERRING_EXPRESSION_SEGMENTATION,
    Florence2TaskTypes.REGION_TO_SEGMENTATION,
]
OVD_TASKS = [Florence2TaskTypes.OPEN_VOCABULARY_DETECTION]
CLASS_AND_LOC_TASKS = [
    Florence2TaskTypes.REGION_TO_CATEGORY,
    Florence2TaskTypes.REGION_TO_DESCRIPTION,
]
OCR_TASKS = [Florence2TaskTypes.OCR]
OCR_WITH_REGION_TASKS = [Florence2TaskTypes.OCR_WITH_REGION]

BBOXES_REQUIRED_KEYS = ["bboxes", "labels"]
POLYGONS_REQUIRED_KEYS = ["polygons", "labels"]
OVD_REQUIRED_KEYS = ["bboxes", "bboxes_labels", "polygons", "polygons_labels"]
OCR_WITH_REGION_REQUIRED_KEYS = ["quad_boxes", "labels"]


def _parse_caption(data: Dict[str, str]) -> Union[Caption, None]:
    if not all(key in CAPTION_TASKS for key in data):
        return None

    if len(data) != 1:
        return None

    task_type = list(data.keys())[0]
    return Caption(caption=data[task_type])


def _parse_od(data: Dict[str, Sequence]) -> Union[BoundingBoxes, None]:
    if not all(key in data for key in BBOXES_REQUIRED_KEYS):
        return None

    bboxes = data["bboxes"]
    labels = data["labels"]

    result_bboxes: List[BoundingBox] = []

    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = bbox

        if label == "":
            label = None

        bbox_inst = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=label)
        result_bboxes.append(bbox_inst)

    return BoundingBoxes(bboxes=result_bboxes)


def _parse_polygon(data: Dict[str, Sequence]) -> Union[Polygons, None]:
    if not all(key in data for key in POLYGONS_REQUIRED_KEYS):
        return None

    polygons = data["polygons"]
    labels = data["labels"]

    result_polygons: List[Polygon] = []

    for polygon, label in zip(polygons, labels):
        points: List[Point] = []

        for p_idx in range(len(polygon) // 2):
            x = polygon[p_idx * 2]
            y = polygon[p_idx * 2 + 1]
            points.append(Point(x=x, y=y))

        if label == "":
            label = None

        polygon_inst = Polygon(points=points, label=label)
        result_polygons.append(polygon_inst)

    return Polygons(polygons=result_polygons)


def _parse_ovd(data: Dict[str, Sequence]) -> Union[OpenVocabularyDetection, None]:
    if not all(key in data for key in OVD_REQUIRED_KEYS):
        return None

    bboxes = data["bboxes"]
    bboxes_labels = data["bboxes_labels"]
    polygons = data["polygons"]
    polygons_labels = data["polygons_labels"]

    result_bboxes: List[BoundingBox] = _parse_od(
        {"bboxes": bboxes, "labels": bboxes_labels}
    )
    result_polygons: List[Polygon] = _parse_polygon(
        {"polygons": polygons, "labels": polygons_labels}
    )

    return OpenVocabularyDetection(
        bounding_boxes=result_bboxes, polygons=result_polygons
    )


def _parse_class_and_loc(data: Dict[str, str]) -> Union[BoundingBoxes, None]:
    if not all(key in CLASS_AND_LOC_TASKS for key in data):
        return None

    task_type = list(data.keys())[0]
    result = data[task_type]

    pattern = r"^(.*)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>$"
    match = re.match(pattern, result)

    if match is None:
        return None

    label = match.group(1)
    x1 = float(match.group(2))
    y1 = float(match.group(3))
    x2 = float(match.group(4))
    y2 = float(match.group(5))

    bbox = BoundingBox(x1=x1, x2=x2, y1=y1, y2=y2, label=label)
    return BoundingBoxes(bboxes=[bbox])


def _parse_ocr(data: Dict[str, str]) -> Union[OCR, None]:
    if not all(key in OCR_TASKS for key in data):
        return None

    task_type = list(data.keys())[0]
    return OCR(text=data[task_type])


def _parse_ocr_with_region(data: Dict[str, Sequence]) -> Union[QuadBoxes, None]:
    if not all(key in data for key in OCR_WITH_REGION_REQUIRED_KEYS):
        return None

    quad_boxes = data["quad_boxes"]
    labels = data["labels"]

    result_quad_boxes: List[QuadBox] = []

    for quad_box, label in zip(quad_boxes, labels):
        x1, y1, x2, y2, x3, y3, x4, y4 = quad_box

        if label == "":
            label = None

        quad_box_inst = QuadBox(
            x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3, x4=x4, y4=y4, label=label
        )
        result_quad_boxes.append(quad_box_inst)

    return QuadBoxes(quad_boxes=result_quad_boxes)


def parse_output(
    data: Dict[str, Union[str, Sequence]], task_type: Florence2TaskTypes
) -> Union[
    Caption, BoundingBoxes, Polygons, OpenVocabularyDetection, OCR, QuadBoxes, None
]:
    try:
        data = data[task_type.value]
    except KeyError:
        pass

    if task_type in CAPTION_TASKS:
        return _parse_caption(data)
    if task_type in BBOX_TASKS:
        return _parse_od(data)
    if task_type in POLYGON_TASKS:
        return _parse_polygon(data)
    if task_type in OVD_TASKS:
        return _parse_ovd(data)
    if task_type in CLASS_AND_LOC_TASKS:
        return _parse_class_and_loc(data)
    if task_type in OCR_TASKS:
        return _parse_ocr(data)
    if task_type in OCR_WITH_REGION_TASKS:
        return _parse_ocr_with_region(data)

    return None
