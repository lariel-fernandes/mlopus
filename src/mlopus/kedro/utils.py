import functools
import logging
from typing import TypeVar, Callable

logger = logging.getLogger(__name__)

T = TypeVar("T")
F = Callable[[...], T]


def log_error(exc: Exception | None, _logger: logging.Logger = logger) -> None:
    """Log exception and mark it as already logged."""
    if exc and not (_dict := getattr(exc, "__dict__", {})).get("logged"):
        _logger.error(repr(exc), exc_info=True)
        _dict["logged"] = True


def log_errors(_logger: logging.Logger = logger) -> Callable[[F], F]:
    """Decorator that catches exceptions, logs them and marks them as already logged."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                log_error(exc, _logger)
                raise exc

        return wrapper

    return decorator
