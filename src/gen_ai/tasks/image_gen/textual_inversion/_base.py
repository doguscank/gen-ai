import abc
from pathlib import Path
from typing import List, Optional, Union

from gen_ai.utils import pathify_strings


class TextualInversionBase(abc.ABC):
    @pathify_strings
    def __init__(self, pipeline) -> None:
        """
        Base class for textual inversion tasks.

        Parameters
        ----------
        pipeline : str
            The pipeline to use for the textual inversion.
        """

        self._pipeline = pipeline

    @property
    def pipeline(self):
        return self._pipeline

    @abc.abstractmethod
    def _load_from_file(
        self,
        file_path: Union[Path, List[Path]],
        token: Union[str, List[str]],
    ) -> None:
        """
        Load the textual inversion file.
        """

    @abc.abstractmethod
    def _load_from_hf(self, hf_model_id: Union[str, List[str]]) -> None:
        """
        Load the textual inversion from Hugging Face.
        """

    @abc.abstractmethod
    def load(
        self,
        *,
        file_path: Optional[Union[Path, List[Path]]] = None,
        token: Optional[Union[str, List[str]]] = None,
        hf_model_id: Optional[Union[str, List[str]]] = None
    ) -> None:
        """
        Load the textual inversion.
        """
