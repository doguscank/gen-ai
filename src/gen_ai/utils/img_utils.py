from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from gen_ai.utils import pathify_strings


def load_image(image_path: Path) -> Image.Image:
    """
    Load an image from the specified path.

    Parameters
    ----------
    image_path : Path
        The path to the image.

    Returns
    -------
    Image.Image
        The loaded image.
    """

    return Image.open(image_path)


def _get_image_name(default_image_name: str, idx: int) -> str:
    """
    Get the image name excluding the extension.

    Note: This function is added for future extensibility.

    Parameters
    ----------
    default_image_name : str
        The default image name to use.
    idx : int
        The index of the image.

    Returns
    -------
    str
        The image name excluding the extension.
    """

    return f"{default_image_name}_{idx}"


@pathify_strings
def _get_next_index(output_dir: Path) -> int:
    """
    Get the next index for the image.

    Parameters
    ----------
    output_dir : Path
        The output directory to save the images.

    Returns
    -------
    int
        The next index for the image.
    """

    return len(list(output_dir.glob("*.png")))


@pathify_strings
def save_images(
    images: List[Image.Image],
    output_dir: str,
    default_image_name: Optional[str] = "image",
    start_idx: Optional[int] = 0,
    extension: Optional[str] = "png",
    auto_index: Optional[bool] = False,
) -> None:
    """
    Save images to the output directory.

    Parameters
    ----------
    images : List[Image.Image]
        A list of images to save.
    output_dir : str
        The output directory to save the images.
    default_image_name : str, optional
        The default image name to use, by default "image"
    start_idx : Optional[int], optional
        The starting index for the images, by default None
    extension : Optional[str], optional
        The extension to use for the images, by default "png"
    auto_index : Optional[bool], optional
        Whether to automatically index the images, by default False
    """

    if auto_index:
        start_idx = _get_next_index(output_dir)

    for idx, image in tqdm(
        enumerate(images), desc="Saving images", total=len(images), unit="image"
    ):
        img_name = _get_image_name(default_image_name, start_idx + idx)

        image.save(f"{output_dir}/{img_name}.{extension}")


def postprocess_mask(
    mask: Union[Image.Image, np.ndarray],
    kernel_size: int = 5,
    dilation_iter: int = 1,
) -> Image.Image:
    """
    Post-process the mask.

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
    # Convert PIL Image to numpy array
    mask_np = np.array(mask)

    # Create kernel for dilation
    kernel_size = 2 * padding + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply dilation
    dilated_mask = cv2.dilate(mask_np, kernel, iterations=iterations)

    # Convert back to PIL Image
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
