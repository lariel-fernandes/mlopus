import functools
import inspect
from collections.abc import Mapping
from typing import Any, Type, TypeVar, Dict, Union

from pydantic import BaseModel as _BaseModelLatest  # Pyantic v1 or v2 BaseModel (whichever is installed)
from pydantic.v1 import (
    BaseModel as _BaseModelV1,  # Pydantic V1 BaseModel (not patched)
    create_model,
    fields,
    Field,
    validator,
    root_validator,
    ValidationError,
    types,
    validate_arguments as _validate_arguments,
)

from mlopus.utils import typing_utils, common

T = TypeVar("T")  # Any type

__all__ = [
    "types",
    "fields",
    "BaseModel",  # Pydantic V1 BaseModel (patched)
    "create_model",
    "Field",
    "validator",
    "root_validator",
    "ValidationError",
]

ConfigType = Union[Type[Any], Dict[str, Any], None]  # Class config type

P = TypeVar("P", _BaseModelV1, _BaseModelLatest)  # Any type of `BaseModel` (v1 or v2, patched or not)

ModelLike = Mapping | _BaseModelV1 | _BaseModelLatest  # Anything that can be parsed into a `BaseModel`


class BaseModel(_BaseModelV1):
    """Patch for pydantic BaseModel."""

    class Config:
        """Pydantic class config."""

        repr_empty: bool = True  # If `False`, skip fields with empty values in representation
        arbitrary_types_allowed = True  # Fixes: RuntimeError: no validator found for <class '...'>
        keep_untouched = (functools.cached_property,)  # Fixes: TypeError: cannot pickle '_thread.RLock' object

    def __repr__(self):
        """Representation skips fields with `repr=False`."""
        args = [
            f"{k}={v}"  # noqa
            for k, f in self.__fields__.items()
            if f.field_info.repr
            and not f.field_info.exclude
            and (not common.is_empty(v := getattr(self, k)) or self.Config.repr_empty)
        ]
        return "%s(%s)" % (self.__class__.__name__, ", ".join(args))

    def __str__(self):
        """String matches representation."""
        return repr(self)


class EmptyStrAsMissing(_BaseModelV1):
    """Mixin for BaseModel."""

    @root_validator(pre=True)  # noqa
    @classmethod
    def _handle_empty_str(cls, values: dict) -> dict:
        """Handles empty strings in input as missing values."""
        return {k: v for k, v in values.items() if v != ""}


class EmptyDictAsMissing(_BaseModelV1):
    """Mixin for BaseModel."""

    @root_validator(pre=True)  # noqa
    @classmethod
    def _handle_empty_dict(cls, values: dict) -> dict:
        """Handles empty dicts in input as missing values."""
        return {k: v for k, v in values.items() if v != {}}


class ExcludeEmptyMixin(_BaseModelV1):
    """Mixin for BaseModel."""

    def dict(self, **kwargs) -> dict:
        """Ignores empty fields when serializing to dict."""
        kwargs["exclude"] = kwargs.get("exclude") or {}
        for field in self.__fields__:
            if common.is_empty(getattr(self, field)):
                kwargs["exclude"][field] = True
        return super().dict(**kwargs)


class HashableMixin:
    """Mixin for BaseModel."""

    def __hash__(self):
        """Fixes: TypeError: unhashable type."""
        return id(self)


class SignatureMixin:
    """Mixin for BaseModel."""

    def __getattribute__(self, attr: str) -> Any:
        """Fixes: AttributeError: '__signature__' attribute of '...' is class-only."""
        if attr == "__signature__":
            return inspect.signature(self.__init__)
        return super().__getattribute__(attr)


class MappingMixin(_BaseModelV1, Mapping):
    """Mixin that allows passing BaseModel instances as kwargs with the '**' operator.

    Example:
        class Foo(MappingMixin):
            x: int = 1
            y: int = 2

        foo = Foo()

        dict(**foo, z=3)  # Returns: {"x": 1, "y": 2, "z": 3}
    """

    def __init__(self, *args, **kwargs):
        # Fix for `RuntimeError(Could not convert dictionary to <class>)` in `pydantic.validate_arguments`
        # when the function expects a `Mapping` and receives a pydantic object with the trait `MappingMixin`.
        if not kwargs and len(args) == 1 and isinstance(arg := args[0], dict):
            kwargs = arg
        super().__init__(**kwargs)

    def __iter__(self):
        return iter(self.__fields__)

    def __getitem__(self, __key):
        return getattr(self, __key)

    def __len__(self):
        return len(self.__fields__)


def create_model_from_data(
    name: str, data: dict, __base__: Type[P] | None = None, use_defaults: bool = True, **kwargs
) -> Type[P]:
    """Infer pydantic model from data."""
    _fields = {}

    for key, value in data.items():
        if isinstance(value, dict):
            type_ = create_model_from_data(key.capitalize(), value, **kwargs)
            default = type_.parse_obj(value)
        elif value is None:
            type_, default = Any, None
        else:
            type_, default = type(value), value

        _fields[key] = (type_, default if use_defaults else Field())

    return create_model(name, **_fields, **kwargs, __base__=__base__)


def create_obj_from_data(
    name: str, data: dict, __base__: Type[P] | None = None, use_defaults_in_model: bool = False, **kwargs
) -> P:
    """Infer pydantic model from data and parse it."""
    model = create_model_from_data(name, data, **kwargs, __base__=__base__, use_defaults=use_defaults_in_model)
    return model.parse_obj(data)


def force_set_attr(obj, key: str, val: Any):
    """Low-level attribute set on object (bypasses validations)."""
    object.__setattr__(obj, key, val)


def is_model_cls(type_: type) -> bool:
    """Check if type is pydantic base model."""
    return typing_utils.safe_issubclass(type_, _BaseModelV1) or typing_utils.safe_issubclass(type_, _BaseModelLatest)


def is_model_obj(obj: Any) -> bool:
    """Check if object is instance of pydantic base model."""
    return is_model_cls(type(obj))


def as_model_cls(type_: type) -> Type[P] | None:
    """If type is pydantic base model, return it. Else return None."""
    return type_ if is_model_cls(type_) else None


def as_model_obj(obj: Any) -> P | None:
    """If object is instance of pydantic base model, return it. Else return None."""
    return obj if is_model_obj(obj) else None


def parse_config(config: Type[Any] | Dict[str, Any] | None) -> Type[Any]:
    """Parse argument into model config class."""
    if isinstance(config := config or {}, dict):
        config = type("Config", (), config)
    return config


def validate_arguments(_func: callable = None, *, config: ConfigType = None):
    """Patch of `validate_arguments` that allows skipping the return type validation.

    Return type validation is turned off by default when the function's
    return type is a string alias to a type that hasn't been defined yet.
    """
    if _func is None:
        return functools.partial(validate_arguments, config=config)

    config = parse_config(config)

    if not getattr(config, "validate_return", not isinstance(_func.__annotations__.get("return"), str)):
        _func.__annotations__.pop("return", None)

    return _validate_arguments(config=config)(_func)
