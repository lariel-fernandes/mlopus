import types
import typing
from typing import Any, TypeVar, Type, Callable, Set

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


def get_type_param(subject: type, bound: Type[T], pos: int, strict: bool) -> Type[T] | None:
    """Extract type param at specified position."""
    try:
        param = typing.get_args(subject)[pos]
    except IndexError:
        if strict:
            raise TypeError(f"No type param at position #{pos} in type {subject}")
        return None

    if isinstance(param, TypeVar):
        param = param.__bound__ or param

    if not safe_issubclass(param, bound):
        raise TypeError(f"Expected a subclass of {bound}, found {param}")

    return param


def iter_bases(cls: type, recursive: bool = False) -> Set[type]:
    """Iterate all base types from a class."""
    for base in getattr(as_type(cls), "__orig_bases__", []):
        yield base
        if recursive:
            yield from iter_bases(base)


def find_bases(
    cls: type,
    filter_: Callable[[type], bool] | None = None,
    _as_type: Type[T] | None = None,
    min_: int | None = None,
    max_: int | None = None,
    recursive: bool = False,
) -> Set[Type[T]]:
    """Find base classes of `cls` that satisfy the filter."""
    bases = {b for b in iter_bases(cls, recursive) if filter_ is None or filter_(b)}

    if min_ is not None and len(bases) < min_ or max_ is not None and len(bases) > max_:
        raise TypeError(
            f"Expected at least {min_} and at most {max_} base classes of {cls} "
            f"matching filter {filter_}, found {len(bases)}: {bases}"
        )

    if _as_type is not None:
        bases = typing.cast(Set[_as_type], bases)

    return bases


def find_base(
    cls: type,
    filter_: Callable[[type], bool] | None = None,
    _as_type: Type[T] | None = None,
    recursive: bool = False,
) -> Type[T]:
    """Find the one base class of `cls` that satisfies the filter."""
    return list(find_bases(cls, filter_, _as_type, min_=1, max_=1, recursive=recursive))[0]
