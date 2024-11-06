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


def create_spherical_mask_on_center(
    height: int, width: int, radius: int
) -> Image.Image:
    """
    Create a spherical mask on the center of the image.

    Parameters
    ----------
    height : int
        The height of the image.
    width : int
        The width of the image.
    radius : int
        The radius of the spherical mask.

    Returns
    -------
    Image.Image
        The spherical mask.
    """

    mask = np.zeros((height, width), np.uint8)
    cv2.circle(mask, (width // 2, height // 2), radius, 255, -1)

    return Image.fromarray(mask)


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
    mask: Union[Image.Image, np.ndarray], kernel_size: int = 5
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
    mask = mask.astype(np.uint8)

    return Image.fromarray(mask)
