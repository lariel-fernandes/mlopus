from typing import Any


def xor(*args: Any) -> bool:
    """XOR operator (exactly one must be True)"""
    return sum(bool(x) for x in args) == 1
