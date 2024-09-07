"""This module is based on the interface :class:`~mlopus.mlflow.BaseMlflowApi`,
which may be implemented in order to work with different MLflow backends/providers
in the scope of experiment tracking and model registry.

Built-in implementations can be found under the module :mod:`mlopus.mlflow.providers`
and are also available under the plugin group `mlopus.mlflow_api_providers`.

Third-party implementations may also be added to that group in order to expand funcionality.
"""

from . import providers
from .api.common import schema, exceptions
from .api.base import BaseMlflowApi
from .api.run import RunApi
from .api.exp import ExpApi
from .api.model import ModelApi
from .api.mv import ModelVersionApi
from .utils import get_api, list_api_plugins, api_conf_schema
from .traits import MlflowRunMixin, MlflowApiMixin, MlflowRunManager

RunStatus = schema.RunStatus

__all__ = [
    "get_api",
    "list_api_plugins",
    "api_conf_schema",
    "RunStatus",
    "exceptions",
    "providers",
    "MlflowRunMixin",
    "MlflowApiMixin",
    "MlflowRunManager",
    "BaseMlflowApi",
    "ExpApi",
    "RunApi",
    "ModelApi",
    "ModelVersionApi",
]
