import abc
from typing import Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator


class OrientedBoundingBox(abc.ABC, BaseModel):
    """
    Represents an oriented bounding box.

    Parameters
    ----------
    coords : np.ndarray
        The 2D coordinates array of shape (4,2).
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

        if value.shape != (4, 2):
            raise ValueError(f"coords must have shape (4,2), got shape {value.shape}")

        # Check for valid numeric data
        if not np.issubdtype(value.dtype, np.number):
            raise ValueError("coords must contain numeric values")

        return value

    @property
    def xyxyxyxy(
        self,
    ) -> Union[
        Tuple[int, int, int, int, int, int, int, int],
        Tuple[float, float, float, float, float, float, float, float],
    ]:
        """
        Returns the bounding box coordinates in the format (x1, y1, x2, y2, x3, y3, x4, y4).
        """

        return self.coords.flatten()

    @property
    def xxxxyyyy(
        self,
    ) -> Union[
        Tuple[int, int, int, int, int, int, int, int],
        Tuple[float, float, float, float, float, float, float, float],
    ]:
        """
        Returns the bounding box coordinates in the format (x1, x2, x3, x4, y1, y2, y3, y4).
        """

        return self.coords.T.flatten()
