from typing import Optional

import numpy as np
import torch
from PIL import Image


def free_gpu_cache() -> None:
    """Free the GPU cache by emptying it."""

    torch.cuda.empty_cache()


def pil_image_to_tensor(
    image: Image.Image, device: Optional[str] = None
) -> torch.Tensor:
    """
    Convert a PIL image to a PyTorch tensor.

    Parameters
    ----------
    image : Image.Image
        The input image.
    device : Optional[str], optional
        The device to use, by default None.

    Returns
    -------
    torch.Tensor
        The output tensor.
    """

    tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

    if device is not None:
        tensor = tensor.to(device)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tensor = tensor.to(device)

    return tensor
