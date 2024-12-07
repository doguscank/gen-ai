from typing import List

from PIL import Image
from pydantic import Field, field_validator

from gen_ai.base.output import Output


class StableDiffusionXLOutput(Output):
    """Output for the Stable Diffusion task."""

    images: List[Image.Image] = Field(default_factory=list)

    @field_validator("images")
    def handle_single_image(cls, value):  # pylint: disable=no-self-argument
        if isinstance(value, Image.Image):
            return [value]
        return value
