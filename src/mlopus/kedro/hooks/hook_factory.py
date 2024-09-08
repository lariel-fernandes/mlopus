from abc import abstractmethod, ABC
from typing import Callable, Any, Mapping
from typing import TypeVar, Generic

from mlopus.utils import pydantic, dicts, string_utils

H = TypeVar("H")  # Type of hook


class HookFactory(ABC, Generic[H]):
    """Base class for hook factories."""

    @abstractmethod
    def __call__(self, config: Mapping) -> H:
        """Use config to build hook."""


class AnonymousHookFactory(pydantic.BaseModel, HookFactory[H], Generic[H]):
    """Builds hook from config using an arbitrary function."""

    func: Callable[[Mapping], H]

    def __call__(self, config: Mapping) -> H:
        """Builds hook from config using an arbitrary function."""
        return self.func(config)


def hook_factory(func: Callable[[Mapping], H]) -> AnonymousHookFactory[H]:
    """Shortcut to turn a function into a hook factory."""
    return AnonymousHookFactory(func=func)


class HookWithFactory(pydantic.BaseModel, pydantic.SignatureMixin, pydantic.HashableMixin):
    """Trait for Hook classes. Exposes a shortcut to get the hook factory."""

    @classmethod
    def _default_hook_name(cls) -> str:
        """Default name of this hook."""
        return string_utils.camel_to_snake(cls.__name__)

    @classmethod
    def _default_config_path(cls) -> str:
        """Default config key-path for obtaining the params that configure this hook."""
        return f"parameters.{cls._default_hook_name()}"

    @classmethod
    def factory(cls, config_path: str | None = None, **extra_params: Any) -> HookFactory:
        """Get hook factory.

        :param config_path: | Path to hook settings in Kedro config.
                            | Defaults to ``parameters.<hook_class_in_snake_case>``

        :param extra_params: Keyword arguments to override the settings from
                             :paramref:`config_path` (merged recursively).
        """
        path = (config_path or cls._default_config_path()).split(".")
        return hook_factory(
            lambda config: cls.parse_obj(dicts.deep_merge(dicts.get_nested(config, path), extra_params))
        )
