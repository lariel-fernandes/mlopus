import functools
import re
from typing import Callable


def escape_sql_single_quote(text: str) -> str:
    """Single quote escape for SQL strings."""
    return text.replace("'", "''")


def unscape_sql_single_quote(text: str) -> str:
    """Revert single quote escape in SQL string."""
    return text.replace("''", "'")


def retval_matches(pattern: re.Pattern, index: int = None):
    """Decorator to enforce that function returns a string in specified pattern."""

    def decorator(func: Callable[[...], str]) -> Callable[[...], str]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> str:
            retval = func(*args, **kwargs)

            if isinstance(retval, tuple) and index is not None:
                if not isinstance(subject := retval[index], str):
                    raise ValueError(f"Expected a string: {retval}")
            elif isinstance(retval, str):
                if index is not None:
                    raise ValueError(f"Expected a tuple: {retval}")
                subject = retval
            else:
                raise ValueError(f"Expected a string: {retval}")

            assert pattern.fullmatch(subject), f"Expected a string matching the pattern '{pattern}': {retval}"
            return retval

        return wrapper

    return decorator
