from pathlib import Path
from typing import List, Optional, Union

from gen_ai.configs.defaults.image_gen import stable_diffusion_15 as sd_config
from gen_ai.logger import logger
from gen_ai.tasks.image_gen.textual_inversion._base import TextualInversionBase
from gen_ai.utils import check_if_hf_cache_exists, pathify_strings


class StableDiffusion15TextualInversion(TextualInversionBase):
    @pathify_strings
    def __init__(self, pipeline) -> None:
        """
        Textual inversion task using Stable Diffusion 15.

        Parameters
        ----------
        file_path : Path, optional
            Path to the textual inversion file.
        token : str, optional
            Token to be used for the textual inversion.
        hf_model_id : str, optional
            Hugging Face model ID to use for the textual inversion.
        """

        super().__init__(pipeline)

    def _load_from_file(
        self,
        file_path: Union[Path, List[Path]],
        token: Union[str, List[str]],
    ) -> None:
        """
        Load the textual inversion file.

        Parameters
        ----------
        file_path : Path, List[Path]
            Path to the textual inversion file.
        token : str, List[str]
            Token to be used for the textual inversion.

        Returns
        -------
        None
        """

        if isinstance(file_path, list) and isinstance(token, list):
            if len(file_path) != len(token):
                logger.error("The number of file paths and tokens should be the same.")
                return
        if isinstance(file_path, str) and isinstance(token, list):
            logger.error("The file path should be a list if the token is a list.")
            return

        if isinstance(file_path, list) and isinstance(token, str):
            token = [token] * len(file_path)

        self.pipeline.load_textual_inversion(file_path, token=token)

    def _load_from_hf(self, hf_model_id: Union[str, List[str]]) -> None:
        """
        Load the textual inversion from Hugging Face.

        Parameters
        ----------
        hf_model_id : str, List[str]
            Hugging Face model ID to use for the textual inversion.

        Returns
        -------
        None
        """

        self.pipeline.load_textual_inversion(
            hf_model_id,
            cache_dir=sd_config.CACHE_DIR,
            only_local_files=check_if_hf_cache_exists(
                cache_dir=sd_config.CACHE_DIR, model_id=hf_model_id
            ),
        )

    def load(
        self,
        *,
        file_path: Optional[Union[Path, List[Path]]] = None,
        token: Optional[Union[str, List[str]]] = None,
        hf_model_id: Optional[Union[str, List[str]]] = None
    ) -> None:
        """
        Load the textual inversion.

        Parameters
        ----------
        file_path : Path, optional
            Path to the textual inversion file.
        token : str, optional
            Token to be used for the textual inversion.
        hf_model_id : str, optional
            Hugging Face model ID to use for the textual inversion.

        Returns
        -------
        None
        """

        if not self.pipeline.check_model_ready():
            logger.error("The model is not ready.")
            return

        if file_path is not None and token is not None:
            self._load_from_file(file_path, token)
        elif hf_model_id is not None:
            self._load_from_hf(hf_model_id)
        else:
            logger.error(
                "Please provide either file path and token or Hugging Face model ID."
            )

    def unload_by_token(self, token: Union[str, List[str]]) -> None:
        """
        Unload the textual inversion by token.

        Parameters
        ----------
        token : str, List[str]
            Token to be unloaded.

        Returns
        -------
        None
        """

        if not self.pipeline.check_model_ready():
            logger.error("The model is not ready.")
            return

        self.pipeline.unload_textual_inversion(token)

    def unload_all(self) -> None:
        """
        Unload all textual inversions.
        """

        if not self.pipeline.check_model_ready():
            logger.error("The model is not ready.")
            return

        self.pipeline.unload_textual_inversion()
