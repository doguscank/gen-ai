import re

from pydantic import Field

from gen_ai.base.output import Output

_RE_PATTERN = r"^(\w+\s+\d+\s+\d+\s+\d+)$"


class LlamaMeshOutput(Output):
    """
    Output class for LlamaMeshModel

    Attributes
    ----------
    response: str
        The response from the model
    obj_data: str
        Output from the model in OBJ format
    """

    response: str
    obj_data: str = Field(None, init=False, repr=False)

    def model_post_init(self, __context) -> None:
        matches = re.findall(_RE_PATTERN, self.response, re.MULTILINE)

        self.obj_data = "\n".join(matches)
