from typing import List


def create_text_prompt(classes: List[str]) -> str:
    """
    Create a text prompt for the given classes.

    Parameters
    ----------
    classes : List[str]
        List of classes to create the prompt for.

    Returns
    -------
    str
        Text prompt for the given classes.
    """

    if len(classes) == 1:
        return classes[0]

    return " <and> ".join(classes)
