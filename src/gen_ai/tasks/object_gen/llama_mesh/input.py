from typing import List, Optional

from pydantic import Field

from gen_ai.base.input import Input


class LlamaMeshInput(Input):
    """
    LlamaMeshInput is the input class for the LlamaMeshModel.

    Attributes
    ----------
    prompt : str
        The prompt for the model.
    max_new_tokens : int
        The maximum number of tokens to generate. Default is 8192.
    stop : list
        The list of tokens to stop generation at.
        Default is ["<|eos_id|>", "<|eot_id|>"].
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
    seed : Optional[int]
        The seed for sampling. For random sampling, set to -1.
        Default is -1.
    messages : List[str]
        The messages to send to the model. Created automatically.
    """

    prompt: str
    max_new_tokens: int = 8192
    stop: list = ["<|eos_id|>", "<|eot_id|>"]
    temperature: float = 0.9
    min_p: float = 0.1
    top_p: float = 0.8
    top_k: int = 20
    stream: bool = True
    seed: Optional[int] = -1
    messages: Optional[List[str]] = Field(None, init=False, repr=False)

    def model_post_init(self, __context) -> "LlamaMeshInput":
        self.messages = [{"role": "user", "content": self.prompt}]

        return self
