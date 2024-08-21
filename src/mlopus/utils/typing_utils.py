import typing
from typing import Any


def assert_isinstance(subject: Any, type_: type):
    """Assert subject is instance of type."""
    if not isinstance(subject, type_):
        raise TypeError(f"Expected an instance of {type_}: {subject}")


def assert_issubclass(subject: Any, type_: type):
    """Assert subject is subclass of type."""
    if not safe_issubclass(subject, type_):
        raise TypeError(f"Expected a subclass of {type_}: {subject}")


def as_type(subject: Any) -> type | None:
    """If subject is a type, return it as it is. If it's a typing alias, return its origin. Otherwise, return None."""
    if isinstance(subject, type):
        return subject

    if origin := typing.get_origin(subject):
        return origin

    return None


def safe_issubclass(subject: Any, bound: type) -> bool:
    """Replacement for `issubclass` that works with generic type aliases (e.g.: Foo[T]).

    Example:
        class Foo(Generic[T]): pass

        issubclass(Foo[int], Foo)  # Raises: TypeError

        is_subclass_or_origin(Foo[int], Foo)  # Returns: True
    """
    if isinstance(subject, type):
        return issubclass(subject, bound)

    if isinstance(origin := typing.get_origin(subject), type):
        return issubclass(origin, bound)

    return False
