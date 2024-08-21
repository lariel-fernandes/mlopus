from pathlib import Path
from typing import Callable, TypeVar, Mapping

from mlopus.utils import pydantic
from . import entity, contract
from .common import schema, decorators, transfer

A = TypeVar("A")  # Any type


class ModelVersionApi(schema.ModelVersion, entity.EntityApi):
    """Shortcut to API calls with specified model version."""

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

    def cache(self):
        """Cache metadata and artifact for this model version."""
        self.cache_meta()
        self.cache_artifact()

    def export(self, target: Path):
        """Export metadata and artifact cache of this model version to target path."""
        self.export_meta(target)
        self.export_artifact(target)

    def cache_meta(self) -> "ModelVersionApi":
        """Fetch latest metadata for this model version and save it to cache."""
        return self._use_values_from(self.api.cache_model_version_meta(self))

    def export_meta(self, target: Path) -> "ModelVersionApi":
        """Export metadata cache for this model version."""
        return self._use_values_from(self.api.export_model_version_meta(self, target))

    @decorators.require_update
    def set_tags(self, tags: Mapping) -> "ModelVersionApi":
        """Set tags on this model version."""
        self.api.set_tags_on_model_version(self, tags)
        return self

    def cache_artifact(self) -> Path:
        """Pull artifact of this model version from MLflow server to local cache."""
        return self.api.cache_model_artifact(self)

    @pydantic.validate_arguments
    def export_artifact(self, target: Path) -> Path:
        """Export model version artifact cache to target path."""
        return self.api.export_model_artifact(self, target)

    @pydantic.validate_arguments
    def list_artifacts(self, path_suffix: str = "") -> transfer.LsResult:
        """List artifacts in this model version."""
        return self.api.list_run_artifacts(self.run, self.path_in_run + "/" + path_suffix.strip("/"))

    def get_artifact(self) -> Path:
        """Get local path to artifact of this model version."""
        return self.api.get_model_artifact(self)

    @pydantic.validate_arguments
    def place_artifact(self, target: Path, overwrite: bool = False, link: bool = True):
        """Place artifact of this model version on target path."""
        self.api.place_model_artifact(self, target, overwrite, link)

    @pydantic.validate_arguments
    def load_artifact(self, loader: Callable[[Path], A]) -> A:
        """Load artifact of this model version."""
        return self.api.load_model_artifact(self, loader)
