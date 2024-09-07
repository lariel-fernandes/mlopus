from typing import Dict, Any, Type, TypeVar, List

from mlopus.utils import import_utils, dicts
from .api.base import BaseMlflowApi

API = TypeVar("API", bound=BaseMlflowApi)
"""Type of :class:`BaseMlflowApi`"""

PLUGIN_GROUP = "mlopus.mlflow_api_providers"


def list_api_plugins() -> List[import_utils.EntryPoint]:
    """List all API plugins available in this environment."""
    return import_utils.list_plugins(PLUGIN_GROUP)


def get_api(
    plugin: str | None = None,
    cls: Type[API] | str | None = None,
    conf: Dict[str, Any] | None = None,
) -> BaseMlflowApi | API:
    """Load MLflow API class or plugin with specified configuration.

    The default API class is :class:`~mlopus.mlflow.providers.mlflow.MlflowApi`
    (registered under the plugin name `mlflow`).

    :param plugin: | Plugin name from group `mlopus.mlflow_api_providers`.
                   | Incompatible with :paramref:`cls`.

    :param cls: | A type that implements :class:`BaseMlflowApi` or a fully qualified class name of such a type
                  (e.g.: `package.module:Class`).
                | Incompatible with :paramref:`plugin`.

    :param conf: | A `dict` of keyword arguments for the resolved API class.
                 | See :func:`api_conf_schema`

    :return: API instance.
    """
    return _get_api_cls(plugin, cls).parse_obj(conf or {})


def api_conf_schema(
    plugin: str | None = None,
    cls: Type[API] | str | None = None,
) -> dicts.AnyDict:
    """Get configuration schema for MLflow API class or plugin.

    :param plugin: | See :paramref:`get_api.plugin`.

    :param cls: | See :paramref:`get_api.cls`.
    """
    return _get_api_cls(plugin, cls).schema()


def _get_api_cls(
    plugin: str | None = None,
    cls: Type[API] | str | None = None,
) -> Type[API]:
    assert None in (plugin, cls), "`plugin` and `cls` are mutually excluding."

    if isinstance(cls, str):
        cls = import_utils.find_type(cls, BaseMlflowApi)
    elif cls is None:
        cls = import_utils.load_plugin(PLUGIN_GROUP, plugin or "mlflow", BaseMlflowApi)

    return cls
