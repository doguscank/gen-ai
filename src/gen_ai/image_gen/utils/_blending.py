from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageFilter

from gen_ai.constants.inpainting_configuration_types import (
    InpaintingBlendingTypes,
    InpaintingPreProcessTypes,
)
from gen_ai.image_gen.utils._bbox_utils import adjust_bounding_box
from gen_ai.logger import logger

_SMOOTH_BLEND_KERNEL_SIZE = 7
_SMOOTHER_BLEND_KERNEL_SIZE = 21
_SMOOTH_BLEND_SIGMA_X = 0
_SMOOTHER_BLEND_SIGMA_X = 11
_POISSON_BLEND_FLAGS = cv2.NORMAL_CLONE  # or cv2.NORMAL_CLONE


def _prepare_images_for_blending(
    image1: Image.Image,
    image2: Image.Image,
    mask: Image.Image,
    pre_process_type: InpaintingPreProcessTypes,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Tuple[int, int, int, int]]]:
    """
    Prepare images for blending based on pre-process type.

    Parameters
    ----------
    image1 : Image.Image
        The first source image.
    image2 : Image.Image
        The second source image.
    mask : Image.Image
        The blending mask.
    pre_process_type : InpaintingPreProcessTypes
        Type of preprocessing applied to the images.

    Returns
    -------
    img1_roi : np.ndarray
        First image region of interest as numpy array.
    img2 : np.ndarray
        Second image as numpy array, resized if needed.
    mask_arr : np.ndarray
        Mask as numpy array, resized if needed.
    bbox : Optional[Tuple[int, int, int, int]]
        Bounding box coordinates if using crop_and_resize, None otherwise.
    """

    if pre_process_type == InpaintingPreProcessTypes.CROP_AND_RESIZE:
        bbox = mask.getbbox()
        bbox = adjust_bounding_box(
            bbox,
            img_height=image1.height,
            img_width=image1.width,
            target_aspect_ratio=(image2.width / image2.height),
        )
        resized_image2 = image2.resize(
            size=(bbox[2] - bbox[0], bbox[3] - bbox[1]),
            resample=Image.Resampling.LANCZOS,
        )
        mask_roi = mask.crop(bbox)
        mask_roi = mask_roi.resize(
            size=(bbox[2] - bbox[0], bbox[3] - bbox[1]),
            resample=Image.Resampling.LANCZOS,
        )

        return (
            np.array(image1.crop(bbox)),
            np.array(resized_image2),
            np.array(mask_roi.convert("L")),
            bbox,
        )
    if pre_process_type == InpaintingPreProcessTypes.RESIZE:
        size = image1.size
        return (
            np.array(image1),
            np.array(image2.resize(size, resample=Image.Resampling.LANCZOS)),
            np.array(mask.resize(size, resample=Image.Resampling.LANCZOS).convert("L")),
            None,
        )

    raise ValueError(f"Invalid pre-process type: {pre_process_type}")


def _finalize_blend(
    blended_img: Image.Image,
    image1: Image.Image,
    bbox: Optional[Tuple[int, int, int, int]],
    pre_process_type: InpaintingPreProcessTypes,
) -> Image.Image:
    """
    Finalize the blending result based on pre-process type.

    Parameters
    ----------
    blended_img : Image.Image
        The blended image region.
    image1 : Image.Image
        The original first image.
    bbox : Optional[Tuple[int, int, int, int]]
        Bounding box coordinates if using crop_and_resize.
    pre_process_type : InpaintingPreProcessTypes
        Type of preprocessing applied to the images.

    Returns
    -------
    Image.Image
        The final blended image.
    """

    if pre_process_type == InpaintingPreProcessTypes.CROP_AND_RESIZE:
        result = image1.copy()
        result.paste(blended_img, box=bbox)
        return result
    return blended_img


def _smooth_blend(
    image1: Image.Image,
    image2: Image.Image,
    mask: Image.Image,
    pre_process_type: InpaintingPreProcessTypes,
) -> Image.Image:
    """
    Blend two images using a smoothed mask with a 7x7 Gaussian kernel.

    Parameters
    ----------
    image1 : Image.Image
        The first source image.
    image2 : Image.Image
        The second source image.
    mask : Image.Image
        The blending mask.
    pre_process_type : InpaintingPreProcessTypes
        Type of preprocessing applied to the images.

    Returns
    -------
    Image.Image
        The blended image.
    """

    img1_roi, img2, mask_arr, bbox = _prepare_images_for_blending(
        image1, image2, mask, pre_process_type
    )

    # Smooth the mask
    smooth_mask = cv2.GaussianBlur(
        mask_arr,
        (_SMOOTH_BLEND_KERNEL_SIZE, _SMOOTH_BLEND_KERNEL_SIZE),
        _SMOOTH_BLEND_SIGMA_X,
    )
    smooth_mask = smooth_mask / 255.0

    # Blend images
    blended = img1_roi * (1 - smooth_mask[..., None]) + img2 * smooth_mask[..., None]
    blended_img = Image.fromarray(blended.astype(np.uint8))

    return _finalize_blend(blended_img, image1, bbox, pre_process_type)


def _smoother_blend(
    image1: Image.Image,
    image2: Image.Image,
    mask: Image.Image,
    pre_process_type: InpaintingPreProcessTypes,
) -> Image.Image:
    """
    Blend two images using a heavily smoothed mask with a 21x21 Gaussian kernel.

    Parameters
    ----------
    image1 : Image.Image
        The first source image.
    image2 : Image.Image
        The second source image.
    mask : Image.Image
        The blending mask.
    pre_process_type : InpaintingPreProcessTypes
        Type of preprocessing applied to the images.

    Returns
    -------
    Image.Image
        The blended image.
    """

    img1_roi, img2, mask_arr, bbox = _prepare_images_for_blending(
        image1, image2, mask, pre_process_type
    )

    smooth_mask = cv2.GaussianBlur(
        mask_arr,
        (_SMOOTHER_BLEND_KERNEL_SIZE, _SMOOTHER_BLEND_KERNEL_SIZE),
        _SMOOTHER_BLEND_SIGMA_X,
    )
    smooth_mask = smooth_mask / 255.0

    # Blend images
    blended = img1_roi * (1 - smooth_mask[..., None]) + img2 * smooth_mask[..., None]
    blended_img = Image.fromarray(blended.astype(np.uint8))

    return _finalize_blend(blended_img, image1, bbox, pre_process_type)


def _linear_blend(
    image1: Image.Image,
    image2: Image.Image,
    mask: Image.Image,
    pre_process_type: InpaintingPreProcessTypes,
) -> Image.Image:
    """
    Blend two images using simple linear alpha blending.

    Parameters
    ----------
    image1 : Image.Image
        The first source image.
    image2 : Image.Image
        The second source image.
    mask : Image.Image
        The blending mask.
    pre_process_type : InpaintingPreProcessTypes
        Type of preprocessing applied to the images.

    Returns
    -------
    Image.Image
        The blended image.
    """

    img1_roi, img2, mask_arr, bbox = _prepare_images_for_blending(
        image1, image2, mask, pre_process_type
    )

    # Simple linear blend
    blended = (
        img1_roi * (1 - mask_arr[..., None] / 255.0)
        + img2 * mask_arr[..., None] / 255.0
    )
    blended_img = Image.fromarray(blended.astype(np.uint8))

    return _finalize_blend(blended_img, image1, bbox, pre_process_type)


def _gaussian_blend(
    image1: Image.Image,
    image2: Image.Image,
    mask: Image.Image,
    pre_process_type: InpaintingPreProcessTypes,
) -> Image.Image:
    """
    Blend two images using a Gaussian-blurred mask for smooth transitions.

    Parameters
    ----------
    image1 : Image.Image
        The first source image.
    image2 : Image.Image
        The second source image.
    mask : Image.Image
        The blending mask.
    pre_process_type : InpaintingPreProcessTypes
        Type of preprocessing applied to the images.

    Returns
    -------
    Image.Image
        The blended image.
    """

    img1_roi, img2, mask_arr, bbox = _prepare_images_for_blending(
        image1, image2, mask, pre_process_type
    )

    mask_blurred = Image.fromarray(mask_arr).filter(ImageFilter.GaussianBlur(radius=5))
    mask_arr = np.array(mask_blurred) / 255.0

    blended = img1_roi * (1 - mask_arr[..., None]) + img2 * mask_arr[..., None]
    blended_img = Image.fromarray(blended.astype(np.uint8))

    return _finalize_blend(blended_img, image1, bbox, pre_process_type)


def _poisson_blend(
    image1: Image.Image,
    image2: Image.Image,
    mask: Image.Image,
    pre_process_type: InpaintingPreProcessTypes,
) -> Image.Image:
    """
    Blend two images using Poisson blending (seamless cloning).

    Parameters
    ----------
    image1 : Image.Image
        The first source image.
    image2 : Image.Image
        The second source image.
    mask : Image.Image
        The blending mask.
    pre_process_type : InpaintingPreProcessTypes
        Type of preprocessing applied to the images.

    Returns
    -------
    Image.Image
        The blended image.
    """

    img1_roi, img2, mask_arr, bbox = _prepare_images_for_blending(
        image1, image2, mask, pre_process_type
    )

    y, x = np.where(mask_arr > 127)  # find valid mask pixels
    if len(x) == 0 or len(y) == 0:
        return image1

    center = (int(np.mean(x)), int(np.mean(y)))

    try:
        result = cv2.seamlessClone(
            img2.astype(np.uint8),
            img1_roi.astype(np.uint8),
            mask_arr,
            center,
            _POISSON_BLEND_FLAGS,
        )
        blended_img = Image.fromarray(result)
        return _finalize_blend(blended_img, image1, bbox, pre_process_type)
    except cv2.error:  # pylint: disable=catching-non-exception
        logger.warning("Poisson blending failed. Falling back to smooth blending.")
        return _smooth_blend(image1, image2, mask, pre_process_type)


def blend(
    image1: Image.Image,
    image2: Image.Image,
    mask: Image.Image,
    blending_type: InpaintingBlendingTypes,
    pre_process_type: InpaintingPreProcessTypes,
) -> Image.Image:
    """
    Blend two images using the given mask and blending type.

    Parameters
    ----------
    image1 : Image.Image
        The first source image.
    image2 : Image.Image
        The second source image.
    mask : Image.Image
        The blending mask.
    blending_type : InpaintingBlendingTypes
        Type of blending to apply.
    pre_process_type : InpaintingPreProcessTypes
        Type of preprocessing applied to the images.

    Returns
    -------
    Image.Image
        The blended image.
    """

    if blending_type == InpaintingBlendingTypes.SMOOTH_BLENDING:
        return _smooth_blend(image1, image2, mask, pre_process_type)
    if blending_type == InpaintingBlendingTypes.SMOOTHER_BLENDING:
        return _smoother_blend(image1, image2, mask, pre_process_type)
    if blending_type == InpaintingBlendingTypes.LINEAR_BLENDING:
        return _linear_blend(image1, image2, mask, pre_process_type)
    if blending_type == InpaintingBlendingTypes.GAUSSIAN_BLENDING:
        return _gaussian_blend(image1, image2, mask, pre_process_type)
    if blending_type == InpaintingBlendingTypes.POISSON_BLENDING:
        return _poisson_blend(image1, image2, mask, pre_process_type)

    raise ValueError(f"Invalid blending type: {blending_type}")
