import importlib
import typing
from typing import List, TypeVar, Type

import importlib_metadata

from mlopus.utils import typing_utils

T = TypeVar("T")  # Any type

EntryPoint = importlib_metadata.EntryPoint


def fq_name(type_: type | Type[T]) -> str:
    """Get fully qualified type name to be used with `find_type`."""
    return "%s:%s" % (type_.__module__, type_.__qualname__)


def find_attr(fq_name_: str, type_: Type[T] | None = None) -> T:
    """Find attribute by fully qualified name (e.g.: `package.module:Class.attribute`)."""
    if ":" in fq_name_:
        mod_path, attr_path = fq_name_.split(":")
    else:
        mod_path, attr_path = fq_name_, None

    cursor = importlib.import_module(mod_path)

    if attr_path:
        for part in attr_path.split("."):
            cursor = getattr(cursor, part)

    if type_ is not None:
        typing_utils.assert_isinstance(cursor, type_)

    return cursor


def find_type(fq_name_: str, type_: Type[T] | None = None) -> Type[T]:
    """Find type by fully qualified name (e.g.: `package.module:Class.InnerClass`)."""
    found = find_attr(fq_name_, type)

    if type_ is not None:
        typing_utils.assert_issubclass(found, type_)

    return typing.cast(Type[T], found)


def list_plugins(group: str) -> List[EntryPoint]:
    """List plugin objects in group."""
    return list(importlib_metadata.entry_points(group=group))


def load_plugin(group: str, name: str, type_: Type[T] | None = None) -> Type[T]:
    """Load named plugin from specified group."""
    if (n := len(plugins := importlib_metadata.entry_points(group=group, name=name))) != 1:
        raise RuntimeError(f"Expected exactly 1 plugin named '{name}' in group '{group}', fround {n}: {plugins}")

    val = list(plugins)[0].load()

    if type_ is not None:
        typing_utils.assert_issubclass(val, type_)

    return val
