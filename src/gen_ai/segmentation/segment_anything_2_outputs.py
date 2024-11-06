from pydantic import BaseModel, Field, ConfigDict
from PIL import Image
import numpy as np
from typing import Optional


class SegmentAnything2Output(BaseModel):
    """
    Output class for Segment Anything 2.

    Parameters
    ----------
    mask : np.ndarray
        The output mask.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    mask: np.ndarray

    def model_post_init(self, __context) -> "SegmentAnything2Output":
        self.mask = self.mask.astype(np.uint8)

        return self
