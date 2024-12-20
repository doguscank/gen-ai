from pathlib import Path
from typing import Optional
from warnings import warn

import torch
from pydantic import Field

from gen_ai.base.model_config import ModelConfig
from gen_ai.configs.defaults import florence_2_cfg
from gen_ai.constants.task_types.florence_2_task_types import Florence2TaskTypes


class Florence2ModelConfig(ModelConfig):
    """
    Configuration class for Florence 2.

    Parameters
    ----------
    causal_lm_hf_model_id : str
        The identifier of the causal language model to use.
    processor_hf_model_id : str
        The identifier of the processor to use.
    causal_lm_model_path : Path, optional
        The path to the causal language model.
    processor_model_path : Path, optional
        The path to the processor.
    device : str, optional
        The device to run the model on. Defaults to "cuda".
    task_type : Florence2TaskTypes, optional
        The type of task to perform. Defaults to
        Florence2TaskTypes.OPEN_VOCABULARY_DETECTION.
    """

    causal_lm_hf_model_id: Optional[str] = None
    processor_hf_model_id: Optional[str] = None
    causal_lm_model_path: Optional[Path] = None
    processor_model_path: Optional[Path] = None
    device: str = "cuda"
    task_type: Florence2TaskTypes = florence_2_cfg.DEFAULT_TASK

    torch_dtype: Optional[torch.dtype] = Field(None, init=False)

    def model_post_init(self, __context) -> None:
        if self.causal_lm_hf_model_id is None and self.causal_lm_model_path is None:
            warn(
                "No causal language model provided. Using the default model "
                f"'{florence_2_cfg.FLORENCE2_CAUSAL_LM_MODEL_ID}'."
            )
            self.causal_lm_hf_model_id = florence_2_cfg.FLORENCE2_CAUSAL_LM_MODEL_ID

        if self.processor_hf_model_id is None and self.processor_model_path is None:
            warn(
                "No processor provided. Using the default processor "
                f"'{florence_2_cfg.FLORENCE2_PROCESSOR_MODEL_ID}'."
            )
            self.processor_hf_model_id = florence_2_cfg.FLORENCE2_PROCESSOR_MODEL_ID

        self.task_type = torch.float16 if torch.cuda.is_available() else torch.float32
