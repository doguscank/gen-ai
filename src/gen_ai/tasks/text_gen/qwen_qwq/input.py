from typing import Dict, List, Optional

from pydantic import Field

from gen_ai.base.input import Input
from gen_ai.configs.defaults.text_gen import qwen_qwq as qwen_qwq_cfg


class QwenQwQInput(Input):
    """
    QwenQwQInput is the input class for the QwenQwQModel.

    Attributes
    ----------
    prompt : str
        The prompt for the model.
    system_prompt : str, optional
        The system prompt for the model. Default is the default system prompt.
    additional_messages : List[Dict[str, str]], optional
        Additional messages to send to the model. Default is None.
    max_new_tokens : int
        The maximum number of tokens to generate. Default is 32768.
    temperature : float
        The temperature for sampling. Default is 0.9.
    min_p : float
        The minimum probability for sampling. Default is 0.1.
    top_p : float
        The top p for sampling. Default is 0.8.
    top_k : int
        The top k for sampling. Default is 20.
    stream : bool
        Whether to stream the output. Default is True.
    seed : int, optional
        The seed for sampling. For random sampling, set to -1. Default is -1.
    messages : List[str]
        The messages to send to the model. Created automatically.
    """

    prompt: str
    system_prompt: Optional[str] = qwen_qwq_cfg.DEFAULT_SYSTEM_PROMPT
    additional_messages: Optional[List[Dict[str, str]]] = None
    max_new_tokens: int = 32768
    temperature: float = 0.9
    min_p: float = 0.1
    top_p: float = 0.8
    top_k: int = 20
    stream: bool = True
    seed: Optional[int] = -1
    messages: Optional[List[str]] = Field(None, init=False, repr=False)

    def model_post_init(self, __context) -> None:
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.prompt},
        ]

        if self.additional_messages:
            self.messages.extend(self.additional_messages)
