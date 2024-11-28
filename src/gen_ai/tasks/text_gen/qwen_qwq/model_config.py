from pathlib import Path
from typing import Optional

from gen_ai.base.model_config import ModelConfig


class QwenQwQModelConfig(ModelConfig):
    """
    Configuration for the QwenQwQModel.

    Attributes
    ----------
    repo_id : str
        The Huggingface repository id.
    filename : str
        The GGUF filename from the Huggingface repository.
    cache_dir : Optional[Path]
        The cache directory. Default is None.
    verbose : bool
        Whether to be verbose. Default is False.
    n_gpu_layers : Optional[int]
        The number of GPU layers to use. -1 means all layers. Default is 40.
    n_ctx : int
        The context size. Default and maximum is 8192 for Llama Mesh.
    flash_attn : bool
        Whether to use Flash Attention. Default is True.
    """

    repo_id: str
    filename: str
    cache_dir: Optional[Path] = None
    verbose: bool = False
    n_gpu_layers: Optional[int] = 40  # -1 means all layers
    n_ctx: int = 32768
    flash_attn: bool = True  # Use Flash Attention
