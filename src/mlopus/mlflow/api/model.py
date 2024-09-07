import typing
from pathlib import Path
from typing import Callable, Iterator, Mapping

from mlopus.utils import dicts, pydantic, mongo
from . import entity, contract
from .common import schema, decorators
from .mv import ModelVersionApi

V = schema.ModelVersion

RunIdentifier = contract.RunIdentifier


class ModelApi(schema.Model, entity.EntityApi):
    """Registered model metadata with MLflow API handle."""

    def _get_latest_data(self) -> schema.Model:
        """Get latest data for this entity. Used for self update after methods with the `require_update` decorator."""
        return self.api.get_model(self)

    @property
    def url(self) -> str:
        """This model's URL."""
        return self.api.get_model_url(self)

    def cache_meta(self) -> "ModelApi":
        """Fetch latest metadata for this model and save it to cache."""
        return self._use_values_from(self.api.cache_model_meta(self))

    def export_meta(self, target: Path) -> "ModelApi":
        """Export model metadata cache to target.

        :param target: Cache export path.
        """
        return self._use_values_from(self.api.export_model_meta(self, target))

    @decorators.require_update
    def set_tags(self, tags: Mapping) -> "ModelApi":
        """Set tags on this model.

        :param tags: See :attr:`schema.Model.tags`.
        """
        self.api.set_tags_on_model(self, tags)
        return self

    @pydantic.validate_arguments
    def get_version(self, version: str) -> ModelVersionApi:
        """Get ModelVersion API by version identifier.

        :param version: Version identifier.
        """
        return typing.cast(ModelVersionApi, self.api.get_model_version((self.name, version)))

    @pydantic.validate_arguments
    def find_versions(
        self, query: mongo.Query | None = None, sorting: mongo.Sorting | None = None
    ) -> Iterator[ModelVersionApi]:
        """Search versions of this model with query in MongoDB query language.

        :param query: Query in MongoDB query language.
        :param sorting: Sorting criteria (e.g.: `[("asc_field", 1), ("desc_field", -1)]`).
        """
        results = self.api.find_model_versions(dicts.set_reserved_key(query, key="model.name", val=self.name), sorting)
        return typing.cast(Iterator[ModelVersionApi], results)

    @pydantic.validate_arguments
    def log_version(
        self,
        run: RunIdentifier,
        source: Path | Callable[[Path], None],
        path_in_run: str | None = None,
        keep_the_source: bool | None = None,
        allow_duplication: bool | None = None,
        use_cache: bool | None = None,
        version: str | None = None,
        tags: Mapping | None = None,
    ) -> ModelVersionApi:
        """Publish artifact file or dir as model version inside the specified experiment run.

        :param run: | Run ID or object.

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
        from .run import RunApi

        mv = self.api.log_model_version(
            self, run, source, path_in_run, keep_the_source, allow_duplication, use_cache, version, tags
        )

        if isinstance(run, RunApi):
            mv.run = run  # inject live run object so the mv gets updates regarding the run status

        return typing.cast(ModelVersionApi, mv)
