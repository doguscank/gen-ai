from pathlib import Path
from typing import List, Optional, Union

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


def _get_file_name(default_file_name: str, idx: int) -> str:
    """
    Get the file name excluding the extension.

    Note: This function is added for future extensibility.

    Parameters
    ----------
    default_file_name : str
        The default file name to use.
    idx : int
        The index of the file.

    Returns
    -------
    str
        The file name excluding the extension.
    """

    return f"{default_file_name}_{idx}"


@pathify_strings
def _get_next_index(output_dir: Path, extension: str) -> int:
    """
    Get the next index for the file.

    Parameters
    ----------
    output_dir : Path
        The output directory to save the file.
    extension : str
        The extension to use for the file.

    Returns
    -------
    int
        The next index for the file.
    """

    if not extension.startswith("."):
        extension = f".{extension}"

    return len(list(output_dir.glob(f"*{extension}")))


@pathify_strings
def save_images(
    images: Union[Image.Image, List[Image.Image]],
    output_dir: str,
    default_file_name: Optional[str] = "image",
    start_idx: Optional[int] = 0,
    extension: Optional[str] = "png",
    auto_index: Optional[bool] = False,
) -> None:
    """
    Save images to the output directory.

    Parameters
    ----------
    images : Union[Image.Image, List[Image.Image]]
        A list of images to save.
    output_dir : str
        The output directory to save the images.
    default_file_name : str, optional
        The default image name to use, by default "image"
    start_idx : Optional[int], optional
        The starting index for the images, by default None
    extension : Optional[str], optional
        The extension to use for the images, by default "png"
    auto_index : Optional[bool], optional
        Whether to automatically index the images, by default False

    Returns
    -------
    None
    """

    if not isinstance(images, list):
        images = [images]

    if auto_index:
        start_idx = _get_next_index(output_dir, extension=extension)

    for idx, image in tqdm(
        enumerate(images), desc="Saving images", total=len(images), unit="image"
    ):
        img_name = _get_file_name(default_file_name, start_idx + idx)

        image.save(f"{output_dir}/{img_name}.{extension}")


@pathify_strings
def save_obj_file(
    obj_data: str,
    output_dir: str,
    default_file_name: Optional[str] = "object",
    start_idx: Optional[int] = 0,
    extension: Optional[str] = "obj",
    auto_index: Optional[bool] = False,
) -> None:
    """
    Save the OBJ file.

    Parameters
    ----------
    obj_data : str
        The OBJ data to save.
    output_dir : str
        The output directory to save the OBJ file.
    default_file_name : str, optional
        The default file name to use, by default "object"
    start_idx : Optional[int], optional
        The starting index for the file, by default None
    extension : Optional[str], optional
        The extension to use for the file, by default "obj"
    auto_index : Optional[bool], optional
        Whether to automatically index the file, by default False

    Returns
    -------
    None
    """

    if auto_index:
        start_idx = _get_next_index(output_dir, extension=extension)

    obj_name = _get_file_name(default_file_name, start_idx)

    with open(f"{output_dir}/{obj_name}.{extension}", "w", encoding="utf-8") as f:
        f.write(obj_data)
