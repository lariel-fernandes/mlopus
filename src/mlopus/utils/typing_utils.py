import types
import typing
from typing import Any, TypeVar, Type, Tuple, Iterator

T = TypeVar("T")  # Any type

NoneType = type(None)


def is_optional(annotation: type | Type[T]) -> bool:
    """Tell if typing annotation is an optional."""
    return (
        (origin := typing.get_origin(annotation)) is types.UnionType
        or origin is typing.Union
        and any(arg is NoneType for arg in typing.get_args(annotation))
    )


def assert_isinstance(subject: Any, type_: type):
    """Assert subject is instance of type."""
    if not isinstance(subject, type_):
        raise TypeError(f"Expected an instance of {type_}: {subject}")


def assert_issubclass(subject: Any, type_: type):
    """Assert subject is subclass of type."""
    if not safe_issubclass(subject, type_):
        raise TypeError(f"Expected a subclass of {type_}: {subject}")


def as_type(subject: Any, of: Type[T] | None = None, strict: bool = False) -> Type[T] | type | None:
    """Coerce subject to type."""

    if isinstance(subject, TypeVar) and (bound := subject.__bound__):
        subject = bound

    if origin := get_origin(subject):
        subject = origin

    if not isinstance(subject, type):
        if strict:
            raise TypeError(f"Cannot coerce to type: {subject}")
        return None

    if of is not None and not issubclass(subject, of):
        raise TypeError(f"Expected a subclass of {of}: {subject}")

    return subject


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


def get_origin(subject: type) -> type:
    """Get type origin from parameterized generic type (handles edge case for Pydantic v2)."""
    if pgm := getattr(subject, "__pydantic_generic_metadata__", None):
        return pgm["origin"]
    else:
        return typing.get_origin(subject)


def get_args(subject: type) -> Tuple[type, ...]:
    """Get type param args from parameterized generic type (handles edge case for Pydantic v2)."""
    if pgm := getattr(subject, "__pydantic_generic_metadata__", None):
        return pgm["args"]
    else:
        return typing.get_args(subject)


def iter_parameterized_bases(cls: type) -> Iterator[Tuple[type, Tuple[type, ...]]]:
    """Iterate pairs of (type_origin, type_param_args) for all parameterized generic types in the class bases."""
    for base in set(cls.__bases__).union([cls.__base__]):
        if base is not None:
            if args := get_args(base):
                yield get_origin(base), args
            yield from iter_parameterized_bases(base)
