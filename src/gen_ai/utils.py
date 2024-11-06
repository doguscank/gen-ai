import inspect
from pathlib import Path
from contextlib import contextmanager
import time
from gen_ai.logger import logger

def pathify_strings(func):
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
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        delta_time = end_time - start_time
        logger.info(f"Time taken in {desc} = {delta_time:.4f} seconds")

