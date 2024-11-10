import re

from gen_ai.constants.florence_2_task_types import Florence2TaskTypes


def _control_location_prompt(text_prompt: str) -> bool:
    """
    Control the location prompt.

    Parameters
    ----------
    text_prompt : str
        The text prompt to use.

    Returns
    -------
    bool
        Whether the location prompt is valid.
    """

    pattern = r"<loc_\d+>"
    return bool(re.search(pattern, text_prompt))


def control_prompt(text_prompt: str, task_prompt: Florence2TaskTypes) -> bool:
    """
    Control the prompt.

    Parameters
    ----------
    text_prompt : str
        The text prompt to use.
    task_prompt : Florence2TaskTypes
        The task prompt to use.

    Returns
    -------
    bool
        Whether the prompt is valid.
    """

    tasks_using_loc_prompts = [
        Florence2TaskTypes.REGION_TO_CATEGORY,
        Florence2TaskTypes.REGION_TO_DESCRIPTION,
        Florence2TaskTypes.REGION_TO_SEGMENTATION,
    ]

    if task_prompt in tasks_using_loc_prompts:
        return _control_location_prompt(text_prompt)

    return True
