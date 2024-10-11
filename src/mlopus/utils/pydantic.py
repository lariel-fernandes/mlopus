import functools
import inspect
from collections.abc import Mapping
from typing import Any, Type, TypeVar, Dict

from pydantic import (
    BaseModel as _BaseModel,
    create_model,
    fields,
    Field,
    field_validator,
    model_validator,
    ValidationError,
    types,
    validate_call,
)
from typing_extensions import Self

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
    "field_validator",
    "model_validator",
    "ValidationError",
]

P = TypeVar("P", bound=_BaseModel)  # Any type of `BaseModel` (patched or not)

ModelLike = Mapping | _BaseModel  # Anything that can be parsed into a `BaseModel`


def root_validator(*args, **kwargs):
    if not kwargs and len(args) == 1:
        return root_validator()(args[0])
    kwargs.pop("allow_reuse", None)
    kwargs.setdefault("mode", "before" if kwargs.pop("pre", False) else "after")
    return model_validator(*args, **kwargs)


def validator(field: str, *args, **kwargs):
    kwargs.pop("allow_reuse", None)
    kwargs.setdefault("mode", "before" if kwargs.pop("pre", False) else "after")
    return field_validator(field, *args, **kwargs)


class BaseModel(_BaseModel):
    """Patch for pydantic BaseModel."""

    class Config:
        """Pydantic class config."""

        coerce_numbers_to_str = True  # Fixes ValidationError when `str` is expected and `int` is passed
        repr_empty: bool = True  # If `False`, skip fields with empty values in representation
        arbitrary_types_allowed = True  # Fixes: RuntimeError: no validator found for <class '...'>
        ignored_types = (functools.cached_property,)  # Fixes: TypeError: cannot pickle '_thread.RLock' object
        protected_namespaces = ()  # Fixes: UserWarning: Field "model_*" has conflict with protected namespace "model_"

    def __repr__(self):
        """Representation skips fields if:
        - Field conf has `repr=False` or `exclude=True`.
        - Field value is empty and class conf has `repr_empty=False`.
        """
        args = [
            f"{k}={v}"  # noqa
            for k, f in self.model_fields.items()
            if f.repr and not f.exclude and (not common.is_empty(v := getattr(self, k)) or self.Config.repr_empty)
        ]
        return "%s(%s)" % (self.__class__.__name__, ", ".join(args))

    def __str__(self):
        """String matches representation."""
        return repr(self)

    def dict(self, *args, **kwargs):
        """Replace deprecated `dict` with `model_dump`."""
        return self.model_dump(*args, **kwargs)

    @classmethod
    def parse_obj(cls, obj: Any) -> Self:
        """Replace deprecated `parse_obj` with `model_validate`."""
        return cls.model_validate(obj)


class EmptyStrAsMissing(_BaseModel):
    """Mixin for BaseModel."""

    @root_validator(pre=True)  # noqa
    @classmethod
    def _handle_empty_str(cls, values: dict) -> dict:
        """Handles empty strings in input as missing values."""
        return {k: v for k, v in values.items() if v != ""}


class EmptyDictAsMissing(_BaseModel):
    """Mixin for BaseModel."""

    @root_validator(pre=True)  # noqa
    @classmethod
    def _handle_empty_dict(cls, values: dict) -> dict:
        """Handles empty dicts in input as missing values."""
        return {k: v for k, v in values.items() if v != {}}


class ExcludeEmptyMixin(_BaseModel):
    """Mixin for BaseModel."""

    def model_dump(self, **kwargs) -> dict:
        """Ignores empty fields when serializing to dict."""
        exclude = kwargs.get("exclude") or set()

        for field in self.model_fields:
            if common.is_empty(getattr(self, field)):
                if isinstance(exclude, dict):
                    exclude[field] = True
                else:
                    exclude.add(field)

        if isinstance(exclude, dict):
            exclude = set(k for k, v in exclude.items() if v)

        return super().model_dump(**kwargs | {"exclude": exclude})


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


class MappingMixin(_BaseModel, Mapping):
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
        return iter(self.model_fields)

    def __getitem__(self, __key):
        return getattr(self, __key)

    def __len__(self):
        return len(self.model_fields)


class BaseParamsMixin(_BaseModel):
    """Mixin for BaseModel that stores a mapping of parameterized generic bases and their respective type args."""

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        cls.__parameterized_bases__ = dict(typing_utils.iter_parameterized_bases(cls))

    @classmethod
    def _find_base_param(cls, of_base: type, at_pos: int, as_type_of: Type[T] | None = None) -> Type[T]:
        for base, params in cls.__parameterized_bases__.items():
            if issubclass(base, of_base):
                break
        else:
            raise TypeError(f"Cannot find parameterized base of type {of_base}")
        return typing_utils.as_type(params[at_pos], of=as_type_of, strict=True)


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
    return typing_utils.safe_issubclass(type_, _BaseModel)


def is_model_obj(obj: Any) -> bool:
    """Check if object is instance of pydantic base model."""
    return is_model_cls(type(obj))


def as_model_cls(type_: type) -> Type[P] | None:
    """If type is pydantic base model, return it. Else return None."""
    return type_ if is_model_cls(type_) else None


def as_model_obj(obj: Any) -> P | None:
    """If object is instance of pydantic base model, return it. Else return None."""
    return obj if is_model_obj(obj) else None


def validate_arguments(_func: callable = None, *, config: Dict[str, Any] = None):
    """Patch of `validate_arguments` that allows skipping the return type validation.

    Return type validation is turned off by default when the function's
    return type is a string alias to a type that hasn't been defined yet.
    """
    config = config or {}

    if _func is None:
        return functools.partial(validate_arguments, config=config)

    if not config.get("validate_return", not isinstance(_func.__annotations__.get("return"), str)):
        _func.__annotations__.pop("return", None)

    return validate_call(config=config)(_func)
