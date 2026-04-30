from typing import Any

__all__ = ["main"]


def main(*args: Any, **kwargs: Any) -> Any:
    # Lazy import to avoid importing cub_pipeline.pipeline during package import.
    # This keeps public API compatibility (`from cub_pipeline import main`) while
    # preventing runpy warning when executing `python -m cub_pipeline.pipeline`.
    from .pipeline import main as _main

    return _main(*args, **kwargs)
