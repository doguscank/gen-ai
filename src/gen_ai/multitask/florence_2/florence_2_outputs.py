from typing import Optional

import numpy as np
from pydantic import BaseModel


class Caption(BaseModel):
    """
    A caption for an image.

    Parameters
    ----------
    caption : str
        The caption for the image.
    """

    caption: str


class BoundingBox(BaseModel):
    """
    A bounding box.

    Parameters
    ----------
    x1 : float
        The x-coordinate of the top-left corner.
    y1 : float
        The y-coordinate of the top-left corner.
    x2 : float
        The x-coordinate of the bottom-right corner.
    y2 : float
        The y-coordinate of the bottom-right corner.
    label : str, optional
        The label of the bounding box.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    label: Optional[str] = None

    @property
    def coords(self) -> np.ndarray:
        """
        Get the coordinates of the bounding box in [x1, y1, x2, y2] format.

        Returns
        -------
        np.ndarray
            The coordinates of the bounding box.
        """

        return np.asarray([self.x1, self.y1, self.x2, self.y2])

    @property
    def coords_int(self) -> np.ndarray:
        """
        Get the coordinates of the bounding box as integers in
        [x1, y1, x2, y2] format.

        Returns
        -------
        np.ndarray
            The coordinates of the bounding box as integers.
        """

        return self.coords.astype(int)


class BoundingBoxes(BaseModel):
    """
    Bounding boxes.

    Parameters
    ----------
    bboxes : List[BoundingBox]
        The bounding boxes.
    """

    bboxes: list[BoundingBox]

    @property
    def coords(self) -> np.ndarray:
        """
        Get the coordinates of the bounding boxes in
        [[x1, y1, x2, y2], ...] format.

        Returns
        -------
        np.ndarray
            The coordinates of the bounding boxes.
        """

        return np.asarray([bbox.coords for bbox in self.bboxes])

    @property
    def coords_int(self) -> np.ndarray:
        """
        Get the coordinates of the bounding boxes as integers in
        [[x1, y1, x2, y2], ...] format.

        Returns
        -------
        np.ndarray
            The coordinates of the bounding boxes as integers.
        """

        return self.coords.astype(int)

    @property
    def labels(self) -> list[str]:
        """
        Get the labels of the bounding boxes.

        Returns
        -------
        list[str]
            The labels of the bounding boxes.
        """

        return [bbox.label for bbox in self.bboxes]


class Point(BaseModel):
    """
    A point.

    Parameters
    ----------
    x : float
        The x-coordinate of the point.
    y : float
        The y-coordinate of the point.
    """

    x: float
    y: float

    @property
    def coords(self) -> np.ndarray:
        """
        Get the coordinates of the point.

        Returns
        -------
        np.ndarray
            The coordinates of the point.
        """

        return np.asarray([self.x, self.y])

    @property
    def coords_int(self) -> np.ndarray:
        """
        Get the coordinates of the point as integers.

        Returns
        -------
        np.ndarray
            The coordinates of the point as integers.
        """

        return self.coords.astype(int)


class Polygon(BaseModel):
    """
    A polygon.

    Parameters
    ----------
    points : List[Point]
        The points of the polygon.
    label : str, optional
        The label of the polygon.
    """

    points: list[Point]
    label: Optional[str] = None

    @property
    def coords(self) -> np.ndarray:
        """
        Get the coordinates of the polygon in
        [[x1, y1], [x2, y2], ...] format.

        Returns
        -------
        np.ndarray
            The coordinates of the polygon.
        """

        return np.asarray([point.coords for point in self.points])

    @property
    def coords_int(self) -> np.ndarray:
        """
        Get the coordinates of the polygon as integers in
        [[x1, y1], [x2, y2], ...] format.

        Returns
        -------
        np.ndarray
            The coordinates of the polygon as integers.
        """

        return self.coords.astype(int)

    @property
    def coords_flatten(self) -> np.ndarray:
        """
        Get the coordinates of the polygon as a flattened array in
        [x1, y1, x2, y2, ...] format.

        Returns
        -------
        np.ndarray
            The coordinates of the polygon as a flattened array.
        """

        return self.coords.flatten()

    @property
    def coords_flatten_int(self) -> np.ndarray:
        """
        Get the coordinates of the polygon as a flattened array of integers in
        [x1, y1, x2, y2, ...] format.

        Returns
        -------
        np.ndarray
            The coordinates of the polygon as a flattened array of integers.
        """

        return self.coords_flatten.astype(int)


class Polygons(BaseModel):
    """
    Polygons.

    Parameters
    ----------
    polygons : List[Polygon]
        The polygons.
    """

    polygons: list[Polygon]

    @property
    def coords(self) -> np.ndarray:
        """
        Get the coordinates of the polygons in
        [[[x1, y1], [x2, y2], ...], ...] format.

        Returns
        -------
        np.ndarray
            The coordinates of the polygons.
        """

        return np.asarray([polygon.coords for polygon in self.polygons])

    @property
    def coords_int(self) -> np.ndarray:
        """
        Get the coordinates of the polygons as integers in
        [[[x1, y1], [x2, y2], ...], ...] format.

        Returns
        -------
        np.ndarray
            The coordinates of the polygons as integers.
        """

        return self.coords.astype(int)

    @property
    def coords_flatten(self) -> np.ndarray:
        """
        Get the coordinates of the polygons as a flattened array in
        [x1, y1, x2, y2, ...] format.

        Returns
        -------
        np.ndarray
            The coordinates of the polygons as a flattened array.
        """

        return np.asarray([polygon.coords_flatten for polygon in self.polygons])

    @property
    def coords_flatten_int(self) -> np.ndarray:
        """
        Get the coordinates of the polygons as a flattened array of integers in
        [x1, y1, x2, y2, ...] format.

        Returns
        -------
        np.ndarray
            The coordinates of the polygons as a flattened array of integers.
        """

        return self.coords_flatten.astype(int)

    @property
    def labels(self) -> list[str]:
        """
        Get the labels of the polygons.

        Returns
        -------
        list[str]
            The labels of the polygons.
        """

        return [polygon.label for polygon in self.polygons]


class OCR(BaseModel):
    """
    OCR.

    Parameters
    ----------
    text : str
        The text.
    """

    text: str


class QuadBox(BaseModel):
    """
    A quadrilateral bounding box.

    Parameters
    ----------
    x1 : float
        The x-coordinate of the top-left corner.
    y1 : float
        The y-coordinate of the top-left corner.
    x2 : float
        The x-coordinate of the top-right corner.
    y2 : float
        The y-coordinate of the top-right corner.
    x3 : float
        The x-coordinate of the bottom-right corner.
    y3 : float
        The y-coordinate of the bottom-right corner.
    x4 : float
        The x-coordinate of the bottom-left corner.
    y4 : float
    label : str, optional
        The label of the quadrilateral bounding box.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    x3: float
    y3: float
    x4: float
    y4: float
    label: Optional[str] = None

    @property
    def coords(self) -> np.ndarray:
        """
        Get the coordinates of the quadrilateral bounding box in
        [x1, y1, x2, y2, x3, y3, x4, y4] format.

        Returns
        -------
        np.ndarray
            The coordinates of the quadrilateral bounding box.
        """

        return np.asarray(
            [self.x1, self.y1, self.x2, self.y2, self.x3, self.y3, self.x4, self.y4]
        )

    @property
    def coords_int(self) -> np.ndarray:
        """
        Get the coordinates of the quadrilateral bounding box as integers in
        [x1, y1, x2, y2, x3, y3, x4, y4] format.

        Returns
        -------
        np.ndarray
            The coordinates of the quadrilateral bounding box as integers.
        """

        return self.coords.astype(int)


class QuadBoxes(BaseModel):
    """
    Quadrilateral bounding boxes.

    Parameters
    ----------
    quad_boxes : List[QuadBox]
        The quadrilateral bounding boxes.
    """

    quad_boxes: list[QuadBox]

    @property
    def coords(self) -> np.ndarray:
        """
        Get the coordinates of the quadrilateral bounding boxes in
        [[[x1, y1, x2, y2, x3, y3, x4, y4], ...] format.

        Returns
        -------
        np.ndarray
            The coordinates of the quadrilateral bounding boxes.
        """

        return np.asarray([quad_box.coords for quad_box in self.quad_boxes])

    @property
    def coords_int(self) -> np.ndarray:
        """
        Get the coordinates of the quadrilateral bounding boxes as integers
        in [[[x1, y1, x2, y2, x3, y3, x4, y4], ...] format

        Returns
        -------
        np.ndarray
            The coordinates of the quadrilateral bounding boxes as integers.
        """

        return self.coords.astype(int)

    @property
    def labels(self) -> list[str]:
        """
        Get the labels of the quadrilateral bounding boxes.

        Returns
        -------
        list[str]
            The labels of the quadrilateral bounding boxes.
        """

        return [quad_box.label for quad_box in self.quad_boxes]


class OpenVocabularyDetection(BaseModel):
    """
    Open vocabulary detection.

    Parameters
    ----------
    bounding_boxes : BoundingBoxes
        The bounding boxes.
    polygons : Polygons
        The polygons.
    """

    bounding_boxes: BoundingBoxes
    polygons: Polygons
