import functools
import logging
from typing import Dict, Any

import mlopus.mlflow
from mlopus.utils import pydantic, dicts
from .api.base import BaseMlflowApi
from .api.common import schema
from .api.run import RunApi
from .utils import get_api

logger = logging.getLogger(__name__)


class MlflowApiMixin(pydantic.BaseModel):
    """Mixin for pydantic classes that hold a reference to a Mlflow API instance.

    The API is instantiated by the utility `mlopus.mlflow.get_api()` on object initialization.

    Example:

    .. code-block:: python

        class Foo(MlflowMixinApi):
            pass

        foo = Foo(
            api={"plugin": "...", "conf": {...}}  # kwargs for `mlopus.mlflow.get_api()`
        )

        foo.api  # BaseMlflowApi
    """

    mlflow_api: BaseMlflowApi = pydantic.Field(exclude=True, default=None)

    @pydantic.validator("mlflow_api", pre=True)  # noqa
    @classmethod
    def _load_mlflow_api(cls, value: BaseMlflowApi | Dict[str, Any] | None) -> BaseMlflowApi:
        return value if isinstance(value, BaseMlflowApi) else get_api(**value or {})

    def using(self, mlflow_api: BaseMlflowApi) -> "MlflowApiMixin":
        """Get a copy of this object that uses the specified MLflow API."""
        return self.copy(update={"mlflow_api": mlflow_api})


class ExpConf(pydantic.BaseModel):
    """Experiment specification for `MlflowRunManager`."""

    name: str = pydantic.Field(description="Used when getting or creating the experiment.")


class RunConf(pydantic.BaseModel, pydantic.EmptyStrAsMissing):
    """Run specification for `MlflowRunManager`."""

    id: str | None = pydantic.Field(default=None, description="Run ID for resuming a previous run.")
    name: str | None = pydantic.Field(default=None, description="Run name for starting a new run.")
    tags: dicts.AnyDict = pydantic.Field(
        default_factory=dict,
        description="Run tags for starting a new run or finding an ongoing one.",
    )


class MlflowRunManager(MlflowApiMixin):
    """A pydantic object that holds a reference to an ongoing MLflow Run API.

    1. If `run.id` is given, that run is resumed.
    2. Otherwise, an ongoing run with `run.tags` in `exp.name` is searched for.
    3. If none can be found, a new one is started in `exp.name` with `run.tags`.
    """

    mlflow_api: BaseMlflowApi = pydantic.Field(exclude=True, alias="api")

    exp: ExpConf = pydantic.Field(
        default_factory=ExpConf,
        description="Experiment specification, used when getting or creating the experiment.",
    )

    run_conf: RunConf = pydantic.Field(
        alias="run",
        default_factory=RunConf,
        description="Run specification, used for finding or creating run, then replaced with the actual run API handle",
    )

    @functools.cached_property
    def run(self) -> RunApi:
        """API handle for the ongoing MLflow Run."""
        return self._resolve_run()

    def _resolve_run(self) -> RunApi:
        """Resume, find or start run and return API handle."""
        if run_id := self.run_conf.id:
            if (run := self.mlflow_api.get_run(run_id)).status != mlopus.mlflow.RunStatus.RUNNING:
                logger.info("MLflow Run URL: %s", run.resume().url)
            return run

        assert self.run_conf.tags, "Cannot locate shared run or start a new one without tags."

        query = {
            **{"tags.%s" % ".".join(k): v for k, v in dicts.flatten(self.run_conf.tags).items()},
            "status": schema.RunStatus.RUNNING,
            "exp.id": (exp := self.mlflow_api.get_or_create_exp(self.exp.name)).id,
        }

        for run in self.mlflow_api.find_runs(query, sorting=[("start_time", -1)]):
            return run.resume()

        run = self.mlflow_api.start_run(exp, self.run_conf.name, self.run_conf.tags)
        logger.info("MLflow Run URL: %s", run.url)
        return run


class MlflowRunMixin(pydantic.BaseModel):
    """Mixin for pydantic classes that hold a reference to a RunAPI.

    If a run ID is provided, that run is resumed.
    Otherwise, an active run is searched with the given tags in the given experiment.
    If no result is found, a new run is started which would match that search criteria.

    Example:

    .. code-block:: python

        class Foo(MlflowRunMixin):
            pass

        foo = Foo(mlflow={
            "api": {...},  # kwargs for `mlopus.mlflow.get_api()`
            "exp": {"name": ...},
            "run": {"name": ..., "tags": ..., "id": ...},
        })

        foo.mlflow.api  # BaseMlflowApi
        foo.mlflow.run  # RunApi in RUNNING status (either resumed, found or started)
    """

    run_manager: MlflowRunManager | None = pydantic.Field(exclude=True, alias="mlflow")
