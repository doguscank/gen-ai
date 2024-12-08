import abc
from typing import Tuple, Union

import numpy as np
from PIL import Image as _Image
from pydantic import BaseModel, ConfigDict, field_validator


class Image(abc.ABC, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    image: _Image.Image

    @field_validator
    def validate_image(
        cls, value: Union[_Image.Image, np.ndarray]
    ):  # pylint: disable=no-self-argument
        if isinstance(value, np.ndarray):
            value = _Image.fromarray(value)

        if not isinstance(value, _Image.Image):
            raise ValueError(f"Expected PIL Image, got {type(value)}")

    @property
    def height(self) -> int:
        return self.image.height

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def num_channels(self) -> int:
        return len(self.image.getbands())

    @property
    def shape(self) -> Tuple[int, ...]:
        return np.array(self.image).shape

    @property
    def size(self) -> int:
        return np.array(self.image).size

    @property
    def array(self) -> np.ndarray:
        return np.array(self.image)
