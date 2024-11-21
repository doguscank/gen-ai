from enum import Enum


class ImageGenTaskTypes(Enum):
    TEXT2IMG = "text2img"
    IMG2IMG = "img2img"
    INPAINTING = "inpainting"
