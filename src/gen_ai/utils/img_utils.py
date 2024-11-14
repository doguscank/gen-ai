from typing import Union

import cv2
import numpy as np
from PIL import Image


def smoothen_mask(
    mask: Union[Image.Image, np.ndarray],
    kernel_size: int = 5,
    dilation_iter: int = 1,
) -> Image.Image:
    """
    Post-process the mask by applying dilation.

    Parameters
    ----------
    mask : Union[Image.Image, np.ndarray]
        The mask to post-process.
    kernel_size : int, optional
        The kernel size to use for post-processing. Defaults to 5.

    Returns
    -------
    Union[Image.Image, np.ndarray]
        The post-processed mask.
    """

    if isinstance(mask, Image.Image):
        mask = np.array(mask)

    # apply gaussian blur
    mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

    # threshold the mask
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    mask = cv2.dilate(
        mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=dilation_iter
    )

    mask = mask.astype(np.uint8)

    return Image.fromarray(mask)


def pad_mask(mask: Image.Image, padding: int, iterations: int = 1) -> Image.Image:
    """
    Pad the mask with the specified padding.

    Parameters
    ----------
    mask : Image.Image
        The mask to pad.
    padding : int
        The padding to apply.
    iterations : int, optional
        The number of iterations for dilation, by default 1.

    Returns
    -------
    Image.Image
        The padded mask.
    """

    mask_np = np.array(mask)

    kernel_size = 2 * padding + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask_np, kernel, iterations=iterations)
    dilated_mask = np.where(dilated_mask > 127, 255, 0).astype(np.uint8)

    return Image.fromarray(dilated_mask)


def mask_image(image: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Mask the image with the specified mask.

    Parameters
    ----------
    image : Image.Image
        The image to mask.
    mask : Image.Image
        The mask to apply.

    Returns
    -------
    Image.Image
        The masked image.
    """

    image_np = np.array(image)  # can be 3 or 1 channel
    mask_np = np.array(mask)  # should be 1 channel

    mask_np = (mask_np > 0).astype(np.uint8)
    if mask_np.ndim == 2:
        mask_np = mask_np[:, :, None]

    image_np = np.where(mask_np, image_np, 0)

    return Image.fromarray(image_np)


def preprocess_mask(mask: Image.Image) -> Image.Image:
    """
    Preprocess the mask by applying dilation and thresholding.

    Parameters
    ----------
    mask : Image.Image
        The mask to preprocess.

    Returns
    -------
    Image.Image
        The preprocessed mask.
    """

    mask_np = np.array(mask.convert("L"))
    mask_np = np.where(mask_np > 127, 255, 0).astype(np.uint8)

    return Image.fromarray(mask_np)
