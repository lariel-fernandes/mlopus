from typing import Any


def is_empty(val: Any) -> bool:
    """Tell if value is None or an empty container."""
    if isinstance(val, bool):
        return False
    return not bool(val)
