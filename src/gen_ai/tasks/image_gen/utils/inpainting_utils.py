from typing import Optional, Tuple

from PIL import Image

from gen_ai.constants.inpainting_configuration_types import (
    InpaintingBlendingTypes,
    InpaintingPostProcessTypes,
    InpaintingPreProcessTypes,
)
from gen_ai.logger import logger
from gen_ai.tasks.image_gen.utils._bbox_utils import adjust_bounding_box
from gen_ai.tasks.image_gen.utils._blending import blend


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
    bbox = adjust_bounding_box(  # match the aspect ratio of the target h and w
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


def postprocess_outputs(
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
            bbox = adjust_bounding_box(
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
            image.paste(
                resized_inpainted_image,
                box=bbox,
                mask=mask_roi,
            )

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
        blended_image = blend(
            image1=image,
            image2=inpainted_image,
            mask=mask,
            blending_type=blending_type,
            pre_process_type=pre_process_type,
        )
        return blended_image

    logger.warning("Invalid post-process type. Returning the original image.")
    return image
