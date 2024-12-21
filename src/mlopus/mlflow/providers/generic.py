import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple, Mapping, Iterator

from mlopus.mlflow.api.base import BaseMlflowApi
from mlopus.mlflow.api.common import schema
from mlopus.utils import pydantic, mongo, iter_utils, urls

logger = logging.getLogger(__name__)

# Entity types used in MLOpus
E = schema.Experiment
R = schema.Run
M = schema.Model
V = schema.ModelVersion


class GenericMlflowApi(BaseMlflowApi):
    """MLflow API provider without any implementation of client-server exchange, meant for offline mode only.


    **Plugin name:** `generic`

    **Requires extras:** `None`

    **Default cache dir:** `None` (no fall back, must be provided)

    Example 1: Using cached metadata and artifacts

    .. code-block:: python

        # At build time
        api = mlopus.mlflow.get_api(plugin="mlflow", ...)
        api.get_model(...).get_version(...).export("build/mlflow-cache")  # Export metadata and artifacts

        # At runtime (no internet access required)
        api = mlopus.mlflow.get_api(plugin="generic", conf={"cache_dir": "build/mlflow-cache"})
        api.get_model(...).get_version(...).load(...)

    Example 2: Using cached metadata and pulling artifacts at runtime

    .. code-block:: python

        # At build time
        api = mlopus.mlflow.get_api(plugin="mlflow", ...)
        api.get_model(...).get_version(...).export_meta("build/mlflow-cache")  # Export metadata only

        # At runtime (no access to MLFlow server required)
        api = mlopus.mlflow.get_api(plugin="generic", conf={"cache_dir": "build/mlflow-cache", "pull_artifacts_in_offline_mode": True})
        api.get_model(...).get_version(...).load(...)  # Triggers artifact pull
    """

    cache_dir: Path

    offline_mode: bool = True

    # =======================================================================================================
    # === Pydantic validators ===============================================================================

    @pydantic.validator("offline_mode")  # noqa
    @classmethod
    def _valid_offline_mode(cls, value: bool) -> bool:
        """Assert offline mode."""
        assert value, f"{cls.__name__} must be used with `offline_mode=True`."
        return value

    # =======================================================================================================
    # === Implementations of abstract methods from `BaseMlflowApi` ==========================================

    def _impl_default_cache_dir(self) -> Path:
        """Assert cache dir is provided."""
        raise AssertionError(f"{self.__class__.__name__} requires `cache_dir` to be specified.")

    def _impl_get_exp_url(self, exp_id: str) -> urls.Url:
        """Get Experiment URL."""
        raise NotImplementedError()

    def _impl_get_run_url(self, run_id: str, exp_id: str) -> urls.Url:
        """Get Run URL."""
        raise NotImplementedError()

    def _impl_get_model_url(self, name: str) -> urls.Url:
        """Get URL to registered model."""
        raise NotImplementedError()

    def _impl_get_mv_url(self, name: str, version: str) -> urls.Url:
        """Get model version URL."""
        raise NotImplementedError()

    def _impl_fetch_exp(self, exp_id: str) -> E:
        """Get Experiment by ID."""
        raise NotImplementedError()

    def _impl_fetch_run(self, run_id: str) -> R:
        """Get Run by ID."""
        raise NotImplementedError()

    def _impl_fetch_model(self, name: str) -> M:
        """Get registered Model by name."""
        raise NotImplementedError()

    def _impl_fetch_mv(self, name_and_version: Tuple[str, str]) -> V:
        """Get ModelVersion by name and version."""
        raise NotImplementedError()

    def _impl_find_experiments(
        self, query: mongo.Query, sorting: mongo.Sorting
    ) -> Tuple[mongo.Query, mongo.Sorting, iter_utils.Paginator[E]]:
        """Push down MongoDB query where possible and return query remainder with results iterator."""
        raise NotImplementedError()

    def _impl_find_runs(
        self, query: mongo.Query, sorting: mongo.Sorting
    ) -> Tuple[mongo.Query, mongo.Sorting, iter_utils.Paginator[R]]:
        """Push down MongoDB query where possible and return query remainder with results iterator."""
        raise NotImplementedError()

    def _impl_find_models(
        self, query: mongo.Query, sorting: mongo.Sorting
    ) -> Tuple[mongo.Query, mongo.Sorting, iter_utils.Paginator[M]]:
        """Push down MongoDB query where possible and return query remainder with results iterator."""
        raise NotImplementedError()

    def _impl_find_mv(
        self, query: mongo.Query, sorting: mongo.Sorting
    ) -> Tuple[mongo.Query, mongo.Sorting, iter_utils.Paginator[V]]:
        """Push down MongoDB query where possible and return query remainder with results iterator."""
        raise NotImplementedError()

    def _impl_find_child_runs(self, run: R) -> Iterator[R]:
        """Find child runs."""
        raise NotImplementedError()

    def _impl_create_exp(self, name: str, tags: Mapping) -> E:
        """Create experiment and return its metadata."""
        raise NotImplementedError()

    def _impl_create_model(self, name: str, tags: Mapping) -> M:
        """Create registered model and return its metadata."""
        raise NotImplementedError()

    def _impl_create_run(
        self, exp_id: str, name: str | None, repo: urls.Url | None, parent_run_id: str | None = None
    ) -> str:
        """Create run."""
        raise NotImplementedError()

    def _impl_set_run_status(self, run_id: str, status: schema.RunStatus):
        """Set Run status."""
        raise NotImplementedError()

    def _impl_set_run_end_time(self, run_id: str, end_time: datetime):
        """Set Run end time."""
        raise NotImplementedError()

    def _impl_update_exp_tags(self, exp_id: str, tags: Mapping):
        """Update Exp tags."""
        raise NotImplementedError()

    def _impl_update_run_tags(self, run_id: str, tags: Mapping):
        """Update Run tags."""
        raise NotImplementedError()

    def _impl_update_model_tags(self, name: str, tags: Mapping):
        """Update Model tags."""
        raise NotImplementedError()

    def _impl_update_mv_tags(self, name: str, version: str, tags: Mapping):
        """Update Exp tags."""
        raise NotImplementedError()

    def _impl_log_run_params(self, run_id: str, params: Mapping):
        """Log run params."""
        raise NotImplementedError()

    def _impl_log_run_metrics(self, run_id: str, metrics: Mapping):
        """Log run metrics."""
        raise NotImplementedError()

    def _impl_register_mv(self, model: M, run: R, path_in_run: str, version: str | None, tags: Mapping) -> V:
        """Register model version."""
        raise NotImplementedError()
