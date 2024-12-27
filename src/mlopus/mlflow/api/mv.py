from pathlib import Path
from typing import Callable, TypeVar, Mapping

from mlopus.utils import pydantic
from . import entity, contract
from .common import schema, decorators, transfer

A = TypeVar("A")  # Any type


class ModelVersionApi(schema.ModelVersion, entity.EntityApi):
    """Model version metadata with MLflow API handle."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from .run import RunApi
        from .model import ModelApi

        self.run: RunApi = RunApi(**self.run)
        self.model: ModelApi = ModelApi(**self.model)

    def using(self, api: contract.MlflowApiContract) -> "ModelVersionApi":
        super().using(api)
        self.run.using(api)
        self.model.using(api)
        return self

    def _get_latest_data(self) -> schema.ModelVersion:
        """Get latest data for this entity. Used for self update after methods with the `require_update` decorator."""
        return self.api.get_model_version(self)

    @property
    def url(self) -> str:
        """Get model version URL."""
        return self.api.get_model_version_url(self)

    @pydantic.validate_arguments
    def clean_cached_artifact(self) -> "ModelVersionApi":
        """Clean cached artifact for this model version."""
        self.api.clean_cached_model_artifact(self)
        return self

    def cache(self):
        """Cache metadata and artifact for this model version."""
        self.cache_meta()
        self.cache_artifact()

    def export(self, target: Path):
        """Export metadata and artifact cache of this model version to target path.

        :param target: Cache export path.
        """
        self.export_meta(target)
        self.export_artifact(target)

    def cache_meta(self) -> "ModelVersionApi":
        """Fetch latest metadata for this model version and save it to cache."""
        return self._use_values_from(self.api.cache_model_version_meta(self))

    def export_meta(self, target: Path) -> "ModelVersionApi":
        """Export model version metadata cache to target.

        :param target: Cache export path.
        """
        return self._use_values_from(self.api.export_model_version_meta(self, target))

    @decorators.require_update
    def set_tags(self, tags: Mapping) -> "ModelVersionApi":
        """Set tags on this model version.

        :param tags: See :attr:`schema.Model.tags`.
        """
        self.api.set_tags_on_model_version(self, tags)
        return self

    def cache_artifact(self) -> Path:
        """Pull artifact of this model version from MLflow server to local cache."""
        return self.api.cache_model_artifact(self)

    @pydantic.validate_arguments
    def export_artifact(self, target: Path) -> Path:
        """Export model version artifact cache to target.

        See also:
            - :meth:`mlopus.mlflow.BaseMlflowApi.export_model_artifact`

        :param target: Cache export path.
        """
        return self.api.export_model_artifact(self, target)

    @pydantic.validate_arguments
    def list_artifacts(self, path_suffix: str = "") -> transfer.LsResult:
        """List artifacts in this model version.

        :param path_suffix: Plain relative path inside model artifact dir (e.g.: `a/b/c`).
        """
        return self.api.list_model_artifact(self, path_suffix)

    def get_artifact(self) -> Path:
        """Get local path to model artifact.

        See also:
            - :meth:`mlopus.mlflow.BaseMlflowApi.get_model_artifact`
        """
        return self.api.get_model_artifact(self)

    @pydantic.validate_arguments
    def place_artifact(self, target: Path, overwrite: bool = False, link: bool = True) -> "ModelVersionApi":
        """Place model version artifact on target path.

        See also:
            - :meth:`mlopus.mlflow.BaseMlflowApi.place_model_artifact`

        :param target: Target path.
        :param overwrite: Overwrite target path if exists.
        :param link: Use symbolic link instead of copy.
        """
        self.api.place_model_artifact(self, target, overwrite, link)
        return self

    @pydantic.validate_arguments
    def load_artifact(self, loader: Callable[[Path], A]) -> A:
        """Load model version artifact.

        See also:
            - :meth:`mlopus.mlflow.BaseMlflowApi.load_model_artifact`

        :param loader: Loader callback.
        """
        return self.api.load_model_artifact(self, loader)
