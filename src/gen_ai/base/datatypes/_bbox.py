import abc
from typing import Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from gen_ai.logger import logger


class BoundingBox(abc.ABC, BaseModel):
    """
    Represents a bounding box.

    Parameters
    ----------
    coords : np.ndarray
        The 2D coordinates array of shape (2,2).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    coords: np.ndarray

    @field_validator("coords")
    def validate_coords(cls, value) -> np.ndarray:  # pylint: disable=no-self-argument
        if not isinstance(value, np.ndarray):
            try:
                value = np.array(value)
            except ValueError as err:
                raise ValueError("coords must be convertible to numpy array") from err

        if value.shape != (2, 2):
            raise ValueError(f"coords must have shape (2,2), got shape {value.shape}")

        # Check for valid numeric data
        if not np.issubdtype(value.dtype, np.number):
            raise ValueError("coords must contain numeric values")

        return value

    @property
    def height(self) -> Union[int, float]:
        if isinstance(self.coords[0, 1], float):
            logger.warning(
                "Calculating height of bounding box with floating point coordinates."
            )

        return self.coords[:, 1].max() - self.coords[:, 1].min()

    @property
    def width(self) -> Union[int, float]:
        if isinstance(self.coords[0, 0], float):
            logger.warning(
                "Calculating width of bounding box with floating point coordinates."
            )

        return self.coords[:, 0].max() - self.coords[:, 0].min()

    @property
    def xyxy(
        self,
    ) -> Union[Tuple[int, int, int, int], Tuple[float, float, float, float]]:
        """
        Returns the bounding box coordinates in the format (x1, y1, x2, y2).
        """

        x1, y1 = self.coords.min(axis=0)
        x2, y2 = self.coords.max(axis=0)
        return x1, y1, x2, y2

    @property
    def xxyy(
        self,
    ) -> Union[Tuple[int, int, int, int], Tuple[float, float, float, float]]:
        """
        Returns the bounding box coordinates in the format (x1, x2, y1, y2).
        """

        x1, y1 = self.coords.min(axis=0)
        x2, y2 = self.coords.max(axis=0)
        return x1, x2, y1, y2

    @property
    def xywh(
        self,
    ) -> Union[Tuple[int, int, int, int], Tuple[float, float, float, float]]:
        """
        Returns the bounding box coordinates in the format (x, y, w, h).
        """

        x1, y1, x2, y2 = self.xyxy
        return x1, y1, x2 - x1, y2 - y1

    @property
    def center(self) -> Union[Tuple[int, int], Tuple[float, float]]:
        """
        Returns the center of the bounding box.
        """

        x1, y1, x2, y2 = self.xyxy

        if isinstance(x1, float):
            return (x1 + x2) / 2.0, (y1 + y2) / 2.0

        return (x1 + x2) // 2, (y1 + y2) // 2

    @property
    def area(self) -> Union[int, float]:
        """
        Returns the area of the bounding box.
        """

        if isinstance(self.coords[0, 0], float):
            logger.warning(
                "Calculating area of bounding box with floating point coordinates. "
                "This may not be accurate."
            )

        return self.height * self.width

    def to_percent_coords(self, width: int, height: int) -> np.ndarray:
        """
        Convert the coordinates to percentage coordinates.

        Parameters
        ----------
        width : int
            The width of the image.
        height : int
            The height of the image.

        Returns
        -------
        np.ndarray
            The percentage coordinates.
        """

        return self.coords / np.array([width, height])
