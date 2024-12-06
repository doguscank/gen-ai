from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from gen_ai.utils import pathify_strings


@pathify_strings
def create_video_from_images(
    images: List[Image.Image],
    output_dir: Path,
    fps: int = 30,
    filename: Optional[str] = None,
) -> None:
    """
    Create a video from a list of images.

    Parameters
    ----------
    images : List[Image.Image]
        The list of images to create the video from.
    output_dir : Path
        The output directory to save the video.
    fps : int, optional
        The frames per second of the video. The default value is 30.
    filename : str, optional
        The name of the video file. If not provided, the default name is `result.mp4`.

    Returns
    -------
    None
    """

    if not filename:
        filename = "result.mp4"

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    # Assuming all images are the same size
    frame_size = images[0].size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)

    for image in images:
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        video_writer.write(frame)

    video_writer.release()
