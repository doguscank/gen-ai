from pathlib import Path
import inspect


def pathify_strings(func):
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        for name, value in bound_args.arguments.items():
            if isinstance(value, str) and ("_path" in name or "_dir" in name):
                bound_args.arguments[name] = Path(value)
        return func(*bound_args.args, **bound_args.kwargs)

    return wrapper
