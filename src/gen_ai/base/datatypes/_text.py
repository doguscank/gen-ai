import abc

from pydantic import BaseModel, ConfigDict


class Text(abc.ABC, BaseModel):
    """
    Represents a text.

    Parameters
    ----------
    text : str
        The text.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    text: str

    @property
    def text(self) -> str:
        return self.text

    @property
    def length(self) -> int:
        return len(self.text)

    def __len__(self) -> int:
        return self.length
