import inspect
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, List, Union

from gen_ai.logger import logger


def pathify_strings(func: Callable):
    """
    Decorator to convert string arguments ending with "_path" or "_dir" to Path objects.

    Parameters
    ----------
    func : Callable
        The function to be decorated.
    """

    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        for name, value in bound_args.arguments.items():
            if isinstance(value, str) and ("_path" in name or "_dir" in name):
                bound_args.arguments[name] = Path(value)
        return func(*bound_args.args, **bound_args.kwargs)

    return wrapper


@contextmanager
def measure_time(desc=""):
    """
    Measure the time taken to execute a block of code.

    Parameters
    ----------
    desc : str
        The description of the block of code.
    """

    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        delta_time = end_time - start_time
        logger.info(f"Time taken in {desc} = {delta_time:.4f} seconds")


def check_if_hf_cache_exists(cache_dir: Path, model_id: Union[str, List[str]]) -> bool:
    """
    Check if the cache directory for the model exists.

    Parameters
    ----------
    cache_dir : Path
        The cache directory for the model.
    model_id : Union[str, List[str]]
        The model id.

    Returns
    -------
    bool
        True if the cache directory exists, False otherwise.
    """

    if isinstance(model_id, str):
        model_dirname = "models--" + model_id.replace("/", "--")
        model_cache_dir = cache_dir / model_dirname

        result = model_cache_dir.exists()
    else:
        result = all(check_if_hf_cache_exists(cache_dir, model) for model in model_id)

    return result
