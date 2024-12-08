from typing import List, Optional, Union

from PIL import Image
from pydantic import Field

from gen_ai.base.input import Input
from gen_ai.constants.task_types.florence_2_task_types import Florence2TaskTypes
from gen_ai.tasks.multitask.florence_2.input_validator import validate_prompt
from gen_ai.tasks.multitask.florence_2.utils import create_text_prompt

TEXT_PROMPT_REQUIRED_TASK_TYPES = [
    Florence2TaskTypes.CAPTION_TO_PHRASE_GROUNDING,
    Florence2TaskTypes.REFERRING_EXPRESSION_SEGMENTATION,
    Florence2TaskTypes.REGION_TO_SEGMENTATION,
    Florence2TaskTypes.OPEN_VOCABULARY_DETECTION,
    Florence2TaskTypes.REGION_TO_CATEGORY,
    Florence2TaskTypes.REGION_TO_DESCRIPTION,
]


class Florence2Input(Input):
    """
    Configuration class for Florence2.

    Parameters
    ----------
    image : PIL.Image.Image
        The image to use for the model.
    text_prompt : Union[str, List[str]]
        The text prompt to use for the model.
    task_prompt : Florence2TaskTypes
        The task prompt to use for the model.
    prompt : Optional[str], optional
        The prompt to use for the model. Defaults to None.
    max_new_tokens : int, optional
        The maximum number of new tokens to generate. Defaults to 1024.
    num_beams : int, optional
        The number of beams to use for generation. Defaults to 3.
    early_stopping : bool, optional
        Whether to stop early. Defaults to False.
    do_sample : bool, optional
        Whether to sample. Defaults to False.
    """

    image: Image.Image
    text_prompt: Optional[Union[str, List[str]]] = None  # List[str] is used for OVD
    task_prompt: Florence2TaskTypes
    prompt: Optional[str] = Field(None, init=False)
    max_new_tokens: int = 1024
    num_beams: int = 3
    early_stopping: bool = False
    do_sample: bool = False

    def model_post_init(self, __context) -> None:
        if self.task_prompt == Florence2TaskTypes.OPEN_VOCABULARY_DETECTION:
            if isinstance(self.text_prompt, list):
                self.text_prompt = create_text_prompt(self.text_prompt)
        else:
            if isinstance(self.text_prompt, list):
                raise ValueError(
                    "Text prompt must be a string for the selected task.\n"
                    f"Text prompt: {self.text_prompt}, Task: {self.task_prompt}"
                )

        if self.text_prompt is not None:
            self.prompt = self.task_prompt.value + self.text_prompt
        else:
            self.prompt = self.task_prompt.value

        if not validate_prompt(
            text_prompt=self.text_prompt, task_prompt=self.task_prompt
        ):
            raise ValueError(
                "Given prompt is invalid for selected task.\n"
                f"Prompt: {self.text_prompt}, Task: {self.task_prompt}"
            )
