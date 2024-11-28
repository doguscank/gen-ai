from typing import Optional

from llama_cpp import Llama

from gen_ai.base.model import Model
from gen_ai.logger import logger
from gen_ai.tasks.text_gen.qwen_qwq.input import QwenQwQInput
from gen_ai.tasks.text_gen.qwen_qwq.model_config import QwenQwQModelConfig
from gen_ai.tasks.text_gen.qwen_qwq.output import QwenQwQOutput


class QwenQwQModel(Model):
    """
    QwenQwQModel class is a wrapper class for the QwenQwQ model.
    """

    def __init__(self, *, config: Optional[QwenQwQModelConfig] = None):
        """
        Initialize the QwenQwQModel class.

        Parameters
        ----------
        config : Optional[QwenQwQModelConfig], optional
            The configuration of the QwenQwQModel class, by default None
        """

        self._pipeline = None
        self._model_config = config

        if self.model_config is not None:
            self._load_pipeline(
                repo_id=self.model_config.repo_id,
                filename=self.model_config.filename,
            )

    @property
    def model_config(self) -> QwenQwQModelConfig:
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
        )

        self._model_config.repo_id = repo_id
        self._model_config.filename = filename

    def update_pipeline(self, model_config: QwenQwQModelConfig) -> None:
        """
        Update the pipeline with the given model configuration.

        Parameters
        ----------
        model_config : QwenQwQModelConfig
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

    def __call__(self, input: QwenQwQInput) -> QwenQwQOutput:
        """
        Generate the text output from the given input.

        Parameters
        ----------
        input : QwenQwQInput
            The input to generate the text output.

        Returns
        -------
        QwenQwQOutput
            The generated text output.
        """

        if not self.check_model_ready():
            logger.error("Model is not ready. Please load the model first.")
            return QwenQwQOutput(response="")

        response = self._pipeline.create_chat_completion(
            messages=input.messages,
            max_tokens=input.max_new_tokens,
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
                if "content" in chunk["choices"][0]["delta"]:
                    print(chunk["choices"][0]["delta"]["content"], end="")
                    model_output += chunk["choices"][0]["delta"]["content"]
        else:
            model_output = response["choices"][0]["message"]["content"]

        output = QwenQwQOutput(response=model_output)

        return output
