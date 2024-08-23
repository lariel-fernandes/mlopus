"""This module offers tools for configuring and using standardized, plugin-based MLflow APIs.

Plugins must implement the interface `mlopus.mlflow.BaseMlflowApi`
and be included in the entry-points group `mlopus.mlflow_api_providers`.

While each plugin may offer access to a different MLflow-like backend/provider,
all plugins are meant to be thread-safe and independent of env vars/global vars,
so multiple API instances can coexist in the same program if necessary.

The default plugin, aliased in the entry-points as `mlflow`,
handles communication to open-source MLflow servers (assuming
no artifacts proxy and server-managed SQL database).

Another built-in alternative is the minimal-dependency `generic` plugin,
which works exclusively with the local cache and does not implement any client-server communication.
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
