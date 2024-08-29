import inspect
from typing import Callable, TypeVar, Any, Dict, Reversible

T = TypeVar("T")  # Any type


def adjust_kwargs(func: callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter keyword arguments by omitting keys not in function signature."""
    return {k: kwargs[k] for k in inspect.signature(func).parameters}


def call_with_kwargs(func: Callable[[...], T], kwargs: Dict[str, Any]) -> T:
    """Call function with keyword arguments while omitting keys not in signature."""
    return func(**adjust_kwargs(func, kwargs))


def compose(funcs: Reversible[Callable[[T], T]]) -> Callable[[T], T]:
    """Compose function.

    Example:
        compose([f, g, h])(x)  # Returns: h(g(f(x)))
    """

    def _composite(x: T) -> T:
        for func in reversed(funcs):
            x = func(x)
        return x

    return _composite
