from pathlib import Path
from typing import Optional

from llama_cpp import Llama

from gen_ai.base.model import Model
from gen_ai.logger import logger
from gen_ai.tasks.object_gen.llama_mesh.input import LlamaMeshInput
from gen_ai.tasks.object_gen.llama_mesh.model_config import LlamaMeshModelConfig
from gen_ai.tasks.object_gen.llama_mesh.output import LlamaMeshOutput
from gen_ai.utils.file_ops import save_obj_file


class LlamaMeshModel(Model):
    """
    LlamaMeshModel class is a wrapper class for the Llama model.
    """

    def __init__(self, *, config: Optional[LlamaMeshModelConfig] = None):
        """
        Initialize the LlamaMeshModel class.

        Parameters
        ----------
        config : Optional[LlamaMeshModelConfig], optional
            The configuration of the LlamaMeshModel class, by default None
        """

        self._pipeline = None
        self._model_config = config

        if self.model_config is not None:
            self._load_pipeline(
                repo_id=self.model_config.repo_id,
                filename=self.model_config.filename,
            )

    @property
    def model_config(self) -> LlamaMeshModelConfig:
        return self._model_config

    @property
    def pipeline(self) -> Llama:
        return self._pipeline

    def check_model_ready(self) -> bool:
        """
        Check if the model is ready to use.

        Returns
        -------
        bool
            True if the model is ready to use, False otherwise.
        """

        return self._pipeline is not None

    def _load_pipeline(
        self,
        repo_id: str,
        filename: str,
    ) -> None:
        """
        Load the pipeline from the given repo_id and filename.

        Parameters
        ----------
        repo_id : str
            The Hugginface model repo id.
        filename : str
            The GGUF filename to load the model from.

        Returns
        -------
        None
        """

        self._pipeline = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            cache_dir=self._model_config.cache_dir,
            verbose=self._model_config.verbose,
            n_gpu_layers=self._model_config.n_gpu_layers,
            n_ctx=self._model_config.n_ctx,
            flash_attn=self._model_config.flash_attn,
            chat_format=self._model_config.chat_format,
        )

        self._model_config.repo_id = repo_id
        self._model_config.filename = filename

    def update_pipeline(self, model_config: LlamaMeshModelConfig) -> None:
        """
        Update the pipeline with the given model configuration.

        Parameters
        ----------
        model_config : LlamaMeshModelConfig
            The model configuration to update the pipeline with.

        Returns
        -------
        None
        """

        self._load_pipeline(
            repo_id=model_config.repo_id, filename=model_config.filename
        )

        self._model_config = model_config

    def _load_model_hard_set(self) -> None:
        """Load the model with the hard set model configuration."""

        if self._model_config is not None:
            if (
                self._model_config.repo_id is not None
                and self._model_config.filename is not None
            ):
                self._load_pipeline(
                    repo_id=self._model_config.repo_id,
                    filename=self._model_config.filename,
                )

    def __call__(
        self, input: LlamaMeshInput, output_dir: Optional[Path] = None
    ) -> LlamaMeshOutput:
        """
        Generate the mesh output from the given input.

        Parameters
        ----------
        input : LlamaMeshInput
            The input to generate the mesh output from.
        output_dir : Optional[Path], optional
            The output directory to save the generated mesh output, by default None

        Returns
        -------
        LlamaMeshOutput
            The generated mesh output.
        """

        if not self.check_model_ready():
            logger.error("Model is not ready. Please load the model first.")
            return LlamaMeshOutput(output="")

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        response = self._pipeline.create_chat_completion(
            messages=input.messages,
            max_tokens=input.max_new_tokens,
            stop=input.stop,
            temperature=input.temperature,
            min_p=input.min_p,
            top_p=input.top_p,
            top_k=input.top_k,
            stream=input.stream,
            seed=input.seed,
        )

        if input.stream:
            model_output = ""
            for chunk in response:
                try:
                    print(chunk["choices"][0]["delta"]["content"], end="")
                    model_output += chunk["choices"][0]["delta"]["content"]
                except KeyError:
                    pass
        else:
            model_output = response["choices"][0]["message"]["content"]

        mesh_output = LlamaMeshOutput(output=model_output)

        if output_dir is not None:
            save_obj_file(
                obj_data=mesh_output.obj_data,
                output_dir=output_dir,
                default_file_name="llama_mesh",
                file_extension=".obj",
                auto_index=True,
            )

        return mesh_output
