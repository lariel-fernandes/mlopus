from typing import Dict, Any, Type, TypeVar, List

from mlopus.utils import import_utils, dicts
from .api.base import BaseMlflowApi

A = TypeVar("A", bound=BaseMlflowApi)

PLUGIN_GROUP = "mlopus.mlflow_api_providers"


def list_api_plugins() -> List[import_utils.EntryPoint]:
    """Get list of all API plugins available in this environment."""
    return import_utils.list_plugins(PLUGIN_GROUP)


def get_api(
    plugin: str | None = None,
    cls: Type[A] | str | None = None,
    conf: Dict[str, Any] | None = None,
) -> BaseMlflowApi | A:
    """Load MLflow API class or plugin with specified configuration.

    The default API class is `mlopus.mlflow.providers.mlflow.MlflowApi`,
    which manages communication with an open source MLflow server
    (assuming no artifacts proxy and database is server-managed).
    """
    return _get_api_cls(plugin, cls).parse_obj(conf or {})


def api_conf_schema(
    plugin: str | None = None,
    cls: Type[A] | str | None = None,
) -> dicts.AnyDict:
    """Get configuration schema for MLflow API class or plugin."""
    return _get_api_cls(plugin, cls).schema()


def _get_api_cls(
    plugin: str | None = None,
    cls: Type[A] | str | None = None,
) -> Type[A]:
    assert None in (plugin, cls), "`plugin` and `cls` are mutually excluding."

    if isinstance(cls, str):
        cls = import_utils.find_type(cls, BaseMlflowApi)
    elif cls is None:
        cls = import_utils.load_plugin(PLUGIN_GROUP, plugin or "mlflow", BaseMlflowApi)

    return cls
