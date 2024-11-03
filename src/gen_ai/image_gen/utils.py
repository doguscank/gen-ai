from PIL import Image
from typing import List, Any, Dict, Optional, Callable
from tqdm import tqdm
from gen_ai.utils import pathify_strings
from pathlib import Path


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
