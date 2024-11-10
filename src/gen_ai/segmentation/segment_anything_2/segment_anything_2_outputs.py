from typing import Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field


class Mask(BaseModel):
    """
    Output class for Segment Anything 2.

    Parameters
    ----------
    mask : np.ndarray
        The output mask.
    shape : Optional[Tuple[int, ...]], optional
        The shape of the mask.
    """

    mask: np.ndarray
    shape: Optional[Tuple[int, ...]] = None
    bbox: Optional[Tuple[int, int, int, int]] = Field(None, init=False, repr=False)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        if self.bbox is None:
            x1 = np.min(np.where(self.mask)[1])
            x2 = np.max(np.where(self.mask)[1])
            y1 = np.min(np.where(self.mask)[0])
            y2 = np.max(np.where(self.mask)[0])
            self.bbox = (x1, y1, x2, y2)
        return self.bbox

    def model_post_init(self, __context) -> "Mask":
        self.mask = self.mask.astype(np.uint8)

        if self.shape is None:
            self.shape = self.mask.shape

        return self
