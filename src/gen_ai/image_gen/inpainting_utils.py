from typing import Optional, Tuple

from PIL import Image

from gen_ai.constants.inpainting_configuration_types import (
    InpaintingBlendingTypes,
    InpaintingPostProcessTypes,
    InpaintingPreProcessTypes,
)
from gen_ai.logger import logger


def _adjust_bounding_box(
    bbox: Tuple[int, int, int, int],
    img_height: int,
    img_width: int,
    target_aspect_ratio: float,
) -> Tuple[int, int, int, int]:
    """
    Adjust the bounding box to match the aspect ratio of the given height and width. The
    function uses image height and width to make sure that the bounding box does not go
    out of the image boundaries.

    **Note:** This function is highly inspired by the implementation in AUTOMATIC1111's
    implementation.

    Link to the implementation: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/masking.py#L39

    Parameters
    ----------
    bbox : Tuple[int, int, int, int]
        The bounding box (x1, y1, x2, y2).
    img_height : int
        The height of the image. Used to set the top and bottom padding.
    img_width : int
        The width of the image. Used to set the left and right padding.
    target_aspect_ratio : float
        The target aspect ratio.

    Returns
    -------
    Tuple[int, int, int, int]
        The adjusted bounding box.
    """

    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    bbox_aspect_ratio = bbox_width / bbox_height

    if bbox_aspect_ratio > target_aspect_ratio:  # bbox is wider, adjust height
        new_bbox_height = int(bbox_width / target_aspect_ratio)
        diff = new_bbox_height - bbox_height
        top_pad = diff // 2
        bottom_pad = diff - top_pad

        if y2 + bottom_pad > img_height:  # if the bottom pad is too large
            bottom_pad = img_height - y2  # adjust the bottom pad
            top_pad = diff - bottom_pad  # and adjust the top pad
        elif y1 - top_pad < 0:
            top_pad = y1
            bottom_pad = diff - top_pad

        y1 -= top_pad
        y2 += bottom_pad
    else:  # bbox is taller, adjust width
        new_bbox_width = int(bbox_height * target_aspect_ratio)
        diff = new_bbox_width - bbox_width
        left_pad = diff // 2
        right_pad = diff - left_pad

        if x2 + right_pad > img_width:
            right_pad = img_width - x2
            left_pad = diff - right_pad
        elif x1 - left_pad < 0:
            left_pad = x1
            right_pad = diff - left_pad

        x1 -= left_pad
        x2 += right_pad

    return x1, y1, x2, y2


def crop_and_resize(
    image: Image.Image,
    mask: Image.Image,
    target_height: int,
    target_width: int,
    resize_method: Image.Resampling = Image.Resampling.LANCZOS,
) -> Tuple[Image.Image, Image.Image]:
    """
    Crop and resize the image and mask according to the bounding box of the mask.

    Parameters
    ----------
    image : Image.Image
        The input image.
    mask : Image.Image
        The input mask.
    target_height : int
        The target height of the cropped and resized image and mask.
    target_width : int
        The target width of the cropped and resized image and mask.
    resize_method : Image.Resampling, optional
        The resampling method for resizing the image and mask. Defaults to
        Image.Resampling.LANCZOS.

    Returns
    -------
    Tuple[Image.Image, Image.Image]
        The cropped and resized image and mask.
    """

    bbox = mask.getbbox()  # x1, y1, x2, y2
    bbox = _adjust_bounding_box(  # match the aspect ratio of the target h and w
        bbox=bbox,
        img_height=image.height,
        img_width=image.width,
        target_aspect_ratio=(target_width / target_height),
    )

    cropped_image = image.crop(bbox)
    cropped_mask = mask.crop(bbox)

    cropped_image = cropped_image.resize((target_width, target_height), resize_method)
    cropped_mask = cropped_mask.resize((target_width, target_height), resize_method)

    return cropped_image, cropped_mask


def preprocess_inputs(
    image: Image.Image,
    mask: Image.Image,
    pre_process_type: InpaintingPreProcessTypes,
    output_width: int,
    output_height: int,
) -> Tuple[Image.Image, Image.Image]:
    """
    Pre-process the input image and mask according to the specified pre-process type.

    Parameters
    ----------
    image : Image.Image
        The input image.
    mask : Image.Image
        The input mask.
    pre_process_type : InpaintingPreProcessTypes
        The pre-process type.
    output_width : int
        The width of the output image and mask.
    output_height : int
        The height of the output image and mask.

    Returns
    -------
    Tuple[Image.Image, Image.Image]
        The pre-processed image and mask.
    """

    if pre_process_type == InpaintingPreProcessTypes.RESIZE:
        image = image.resize((output_width, output_height), Image.Resampling.LANCZOS)
        mask = mask.resize((output_width, output_height), Image.Resampling.LANCZOS)
    elif pre_process_type == InpaintingPreProcessTypes.CROP_AND_RESIZE:
        image, mask = crop_and_resize(image, mask, output_height, output_width)

    return image, mask


def postprocess_outputs(  # pylint: disable=unused-argument
    image: Image.Image,
    mask: Image.Image,
    inpainted_image: Image.Image,
    pre_process_type: InpaintingPreProcessTypes,
    post_process_type: InpaintingPostProcessTypes,
    blending_type: Optional[InpaintingBlendingTypes] = None,
) -> Image.Image:
    """
    Post-process the inpainted image according to the specified post-process type. The
    function uses the pre-process type to determine the post-processing steps.

    Parameters
    ----------
    image : Image.Image
        The input image.
    mask : Image.Image
        The input mask.
    inpainted_image : Image.Image
        The inpainted image.
    pre_process_type : InpaintingPreProcessTypes
        The pre-process type.
    post_process_type : InpaintingPostProcessTypes
        The post-process type.
    blending_type : Optional[InpaintingBlendingTypes], optional
        The blending type. Defaults to None.

    Returns
    -------
    Image.Image
        The post-processed inpainted image.
    """

    # Directly replace the masked region with the inpainted image
    if post_process_type == InpaintingPostProcessTypes.DIRECT_REPLACE:
        if pre_process_type == InpaintingPreProcessTypes.CROP_AND_RESIZE:
            bbox = mask.getbbox()
            bbox = _adjust_bounding_box(
                bbox,
                img_height=image.height,
                img_width=image.width,
                target_aspect_ratio=(inpainted_image.width / inpainted_image.height),
            )
            mask_roi = mask.crop(bbox)

            resized_inpainted_image = inpainted_image.resize(
                size=(bbox[2] - bbox[0], bbox[3] - bbox[1]),
                resample=Image.Resampling.LANCZOS,
            )
            mask_roi = mask_roi.resize(
                size=(bbox[2] - bbox[0], bbox[3] - bbox[1]),
                resample=Image.Resampling.LANCZOS,
            )
            image.paste(resized_inpainted_image, box=bbox, mask=mask_roi)

            return image
        if pre_process_type == InpaintingPreProcessTypes.RESIZE:
            size = image.size

            resized_inpainted_image = inpainted_image.resize(
                size, resample=Image.Resampling.LANCZOS
            )
            resized_mask = mask.resize(size, resample=Image.Resampling.LANCZOS)
            image.paste(resized_inpainted_image, mask=resized_mask)

            return image

    if post_process_type == InpaintingPostProcessTypes.BLEND:
        logger.warning("Blending is not supported yet. Use direct replacement instead.")
        return image

    logger.warning("Invalid post-process type. Returning the original image.")
    return image
