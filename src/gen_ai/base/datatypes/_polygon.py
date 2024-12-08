import abc

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Polygon(abc.ABC, BaseModel):
    """
    Represents a polygon.

    Parameters
    ----------
    coords : np.ndarray
        The 2D coordinates array of shape (n,2) where n is the number of vertices.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    num_vertices: int = Field(
        default=-1, frozen=True, init=False
    )  # Default to an invalid value

    coords: np.ndarray = Field(
        description="2D coordinates array of shape (n,2) where n is number of vertices",
        min_length=3,  # Minimum 3 vertices for any polygon
    )

    @field_validator("coords")
    def validate_coords(cls, value) -> np.ndarray:  # pylint: disable=no-self-argument
        if not isinstance(value, np.ndarray):
            try:
                value = np.array(value)
            except ValueError as err:
                raise ValueError("coords must be convertible to numpy array") from err

        if value.ndim != 2:
            raise ValueError(
                f"coords must be 2-dimensional, got {value.ndim} dimensions"
            )

        if value.shape[1] != 2:
            raise ValueError(f"coords must have shape (n,2), got shape {value.shape}")

        if value.shape[0] < 3:
            raise ValueError(
                f"polygon must have at least 3 vertices, got {value.shape[0]}"
            )

        if value.shape[0] != cls.num_vertices:
            raise ValueError(
                f"{cls.__name__} expects {cls.num_vertices} vertices, got {value.shape[0]}"
            )

        # Check for valid numeric data
        if not np.issubdtype(value.dtype, np.number):
            raise ValueError("coords must contain numeric values")

        return value

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
