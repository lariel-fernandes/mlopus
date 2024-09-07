import typing
from pathlib import Path
from typing import Callable, TypeVar, Iterator, Mapping

from mlopus.utils import dicts, pydantic, mongo, urls
from . import entity, contract
from .common import schema, decorators, transfer
from .mv import ModelVersionApi

A = TypeVar("A")  # Any type

M = schema.ModelVersion

ModelIdentifier = contract.ModelIdentifier


class RunApi(schema.Run, entity.EntityApi):
    """Run metadata with MLflow API handle."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from .exp import ExpApi

        self.exp: ExpApi = ExpApi(**self.exp)

    def using(self, api: contract.MlflowApiContract) -> "RunApi":
        super().using(api)
        self.exp.using(api)
        return self

    def _get_latest_data(self) -> schema.Run:
        """Get latest data for this entity. Used for self update after methods with the `require_update` decorator."""
        return self.api.get_run(self)

    def __enter__(self):
        """Executed upon entering a `with` block."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Executed upon exiting a `with` block."""
        self.end_run(succeeded := exc_type is None)
        return succeeded

    @property
    def url(self) -> str:
        """The URL to this experiment run."""
        return self.api.get_run_url(self, self.exp)

    @pydantic.validate_arguments
    def clean_cached_artifact(self, path_in_run: str = "") -> "RunApi":
        """Clean cached artifact for this run.

        :param path_in_run: | See :paramref:`~mlopus.mlflow.BaseMlflowApi.clean_cached_run_artifact.path_in_run`
        """
        self.api.clean_cached_run_artifact(self, path_in_run)
        return self

    @pydantic.validate_arguments
    def list_artifacts(self, path_in_run: str = "") -> transfer.LsResult:
        """List artifacts in this run's artifact repo.

        :param path_in_run: | See :paramref:`~mlopus.mlflow.BaseMlflowApi.list_run_artifacts.path_in_run`
        """
        return self.api.list_run_artifacts(self, path_in_run)

    @pydantic.validate_arguments
    def cache_artifact(self, path_in_run: str = "") -> Path:
        """Pull run artifact from MLflow server to local cache.

        :param path_in_run: | See :paramref:`~mlopus.mlflow.BaseMlflowApi.cache_run_artifact.path_in_run`
        """
        return self.api.cache_run_artifact(self, path_in_run)

    @pydantic.validate_arguments
    def export_artifact(self, target: Path, path_in_run: str = "") -> Path:
        """Export run artifact cache to target.

        See also:
            - :meth:`mlopus.mlflow.BaseMlflowApi.export_run_artifact`

        :param target: Cache export path.
        :param path_in_run: | See :paramref:`~mlopus.mlflow.BaseMlflowApi.export_run_artifact.path_in_run`
        """
        return self.api.export_run_artifact(self, target, path_in_run)

    @pydantic.validate_arguments
    def get_artifact(self, path_in_run: str = "") -> Path:
        """Get local path to run artifact.

        See also:
            - :meth:`mlopus.mlflow.BaseMlflowApi.get_run_artifact`

        :param path_in_run: | See :paramref:`~mlopus.mlflow.BaseMlflowApi.get_run_artifact.path_in_run`
        """
        return self.api.get_run_artifact(self, path_in_run)

    @pydantic.validate_arguments
    def place_artifact(
        self, target: Path, path_in_run: str = "", overwrite: bool = False, link: bool = True
    ) -> "RunApi":
        """Place run artifact on target path.

        See also:
            - :meth:`mlopus.mlflow.BaseMlflowApi.place_run_artifact`

        :param target: Target path.
        :param path_in_run: | See :paramref:`~mlopus.mlflow.BaseMlflowApi.place_run_artifact.path_in_run`
        :param overwrite: | See :paramref:`~mlopus.mlflow.BaseMlflowApi.place_run_artifact.overwrite`
        :param link: | See :paramref:`~mlopus.mlflow.BaseMlflowApi.place_run_artifact.link`
        """
        self.api.place_run_artifact(self, target, path_in_run, overwrite, link)
        return self

    @pydantic.validate_arguments
    def load_artifact(self, loader: Callable[[Path], A], path_in_run: str = "") -> A:
        """Load run artifact.

        See also:
            - :meth:`mlopus.mlflow.BaseMlflowApi.load_run_artifact`

        :param loader: Loader callback.
        :param path_in_run: | See :paramref:`~mlopus.mlflow.BaseMlflowApi.load_run_artifact.path_in_run`
        """
        return self.api.load_run_artifact(self, loader, path_in_run)

    @pydantic.validate_arguments
    def log_artifact(
        self,
        source: Path | Callable[[Path], None],
        path_in_run: str | None = None,
        keep_the_source: bool | None = None,
        allow_duplication: bool | None = None,
        use_cache: bool | None = None,
    ) -> "RunApi":
        """Publish artifact file or dir to this experiment run.

        See also:
            - :meth:`mlopus.mlflow.BaseMlflowApi.log_run_artifact`

        :param source: | See :paramref:`~mlopus.mlflow.BaseMlflowApi.log_run_artifact.source`
        :param path_in_run: | See :paramref:`~mlopus.mlflow.BaseMlflowApi.log_run_artifact.path_in_run`
        :param keep_the_source: | See :paramref:`~mlopus.mlflow.BaseMlflowApi.log_run_artifact.keep_the_source`
        :param allow_duplication: | See :paramref:`~mlopus.mlflow.BaseMlflowApi.log_run_artifact.allow_duplication`
        :param use_cache: | See :paramref:`~mlopus.mlflow.BaseMlflowApi.log_run_artifact.use_cache`
        """
        self.api.log_run_artifact(self, source, path_in_run, keep_the_source, allow_duplication, use_cache)
        return self

    @pydantic.validate_arguments
    def log_model_version(
        self,
        model: ModelIdentifier,
        source: Path | Callable[[Path], None],
        path_in_run: str | None = None,
        keep_the_source: bool | None = None,
        allow_duplication: bool = False,
        use_cache: bool | None = None,
        version: str | None = None,
        tags: Mapping | None = None,
    ) -> ModelVersionApi:
        """Publish artifact file or dir as model version inside this experiment run.

        :param model: | Model name or object.

        :param source: | See :paramref:`~mlopus.mlflow.BaseMlflowApi.log_run_artifact.source`

        :param path_in_run: | See :paramref:`~mlopus.mlflow.BaseMlflowApi.log_model_version.path_in_run`

        :param keep_the_source: | See :paramref:`~mlopus.mlflow.BaseMlflowApi.log_run_artifact.keep_the_source`

        :param allow_duplication: | See :paramref:`~mlopus.mlflow.BaseMlflowApi.log_run_artifact.allow_duplication`

        :param use_cache: | See :paramref:`~mlopus.mlflow.BaseMlflowApi.log_run_artifact.use_cache`

        :param version: | See :paramref:`~mlopus.mlflow.BaseMlflowApi.log_model_version.version`

        :param tags: | Model version tags.
                     | See :class:`schema.ModelVersion.tags`

        :return: New model version metadata with API handle.
        """
        mv = self.api.log_model_version(
            model, self, source, path_in_run, keep_the_source, allow_duplication, use_cache, version, tags
        )
        return typing.cast(ModelVersionApi, mv)

    @pydantic.validate_arguments
    def find_model_versions(
        self, query: mongo.Query | None = None, sorting: mongo.Sorting | None = None
    ) -> Iterator[ModelVersionApi]:
        """Search model versions belonging to this run with query in MongoDB query language.

        :param query: Query in MongoDB query language.
        :param sorting: Sorting criteria (e.g.: `[("asc_field", 1), ("desc_field", -1)]`).
        """
        results = self.api.find_model_versions(dicts.set_reserved_key(query, key="run.id", val=self.id), sorting)
        return typing.cast(Iterator[ModelVersionApi], results)

    def cache_meta(self) -> "RunApi":
        """Fetch latest metadata of this run and save to cache."""
        return self._use_values_from(self.api.cache_run_meta(self))

    def export_meta(self, target: Path) -> "RunApi":
        """Export metadata cache for this run.

        :param target: Cache export path..
        """
        return self._use_values_from(self.api.export_run_meta(self, target))

    @pydantic.validate_arguments
    def create_child(
        self,
        name: str | None = None,
        tags: Mapping | None = None,
        repo: str | urls.Url | None = None,
    ) -> "RunApi":
        """Declare a new child run to be used later.

        :param name: See :attr:`schema.Run.name`.
        :param tags: See :attr:`schema.Run.tags`.
        :param repo: See :paramref:`~mlopus.mlflow.BaseMlflowApi.create_run.repo`.
        """
        return typing.cast(RunApi, self.api.create_run(self.exp, name, tags, repo, self))

    @pydantic.validate_arguments
    def start_child(
        self,
        name: str | None = None,
        tags: Mapping | None = None,
        repo: str | urls.Url | None = None,
    ) -> "RunApi":
        """Start a new child run.

        :param name: See :attr:`schema.Run.name`.
        :param tags: See :attr:`schema.Run.tags`.
        :param repo: See :paramref:`~mlopus.mlflow.BaseMlflowApi.create_run.repo`.
        """
        return typing.cast(RunApi, self.api.start_run(self.exp, name, tags, repo, self))

    @property
    def children(self) -> Iterator["RunApi"]:
        """Child runs."""
        results = self.api.find_child_runs(parent=self)
        return typing.cast(Iterator[RunApi], results)

    def resume(self) -> "RunApi":
        """Resume this experiment run."""
        return self._use_values_from(self.api.resume_run(self))

    def end_run(self, succeeded: bool = True) -> "RunApi":
        """End experiment run."""
        return self._use_values_from(self.api.end_run(self, succeeded))

    @decorators.require_update
    def set_tags(self, tags: Mapping) -> "RunApi":
        """Set tags on this run.

        :param tags: See :attr:`schema.Run.tags`.
        """
        self.api.set_tags_on_run(self, tags)
        return self

    @decorators.require_update
    def log_params(self, params: Mapping) -> "RunApi":
        """Log params to this run.

        :param params: See :attr:`schema.Run.params`.
        """
        self.api.log_params(self, params)
        return self

    @decorators.require_update
    def log_metrics(self, metrics: Mapping) -> "RunApi":
        """Log metrics to this experiment run.

        :param metrics: See :attr:`schema.Run.metrics`.
        """
        self.api.log_metrics(self, metrics)
        return self
