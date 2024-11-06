import torch


def free_gpu_cache() -> None:
    """Free the GPU cache by emptying it."""

    torch.cuda.empty_cache()
