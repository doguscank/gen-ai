from typing import Tuple


def adjust_bounding_box(
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
