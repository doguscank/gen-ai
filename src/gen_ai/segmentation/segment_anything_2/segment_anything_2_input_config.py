from typing import Optional

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict


class SegmentAnything2InputConfig(BaseModel):
    """
    Input class for Segment Anything 2.

    Parameters
    ----------
    image : Image.Image
        The image.
    bounding_box : np.ndarray, optional
        The bounding box. Defaults to None.
    point_coords : Optional[np.ndarray], optional
        The point coordinates. Defaults to None.
    point_labels : Optional[np.ndarray], optional
        The point labels. Defaults to None.
    mask_input : Optional[np.ndarray], optional
        The mask input. Defaults to None.
    multimask_output : bool, optional
        Whether to output multiple masks. Defaults to True.
    return_logits : bool, optional
        Whether to return logits. Defaults to False.
    normalize_coords : bool, optional
        Whether to normalize coordinates. Defaults to True.
    refine_mask : bool, optional
        Whether to refine the mask. Defaults to True.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    image: Image.Image
    bounding_box: Optional[np.ndarray] = None
    point_coords: Optional[np.ndarray] = None
    point_labels: Optional[np.ndarray] = None
    mask_input: Optional[np.ndarray] = None
    multimask_output: bool = True
    return_logits: bool = False  # has no effect for now
    normalize_coords: bool = True
    refine_mask: bool = True
