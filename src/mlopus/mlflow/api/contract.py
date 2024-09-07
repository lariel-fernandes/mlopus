from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar, Callable, Iterator, Tuple, Mapping

from mlopus.utils import pydantic, mongo, urls
from .common import schema, transfer

A = TypeVar("A")  # Any type

# Types used by MLOpus
E = schema.Experiment
R = schema.Run
M = schema.Model
V = schema.ModelVersion

# Identifier types
ExpIdentifier = E | str  # Exp entity or exp ID
RunIdentifier = R | str  # Run entity or run ID
ModelIdentifier = M | str  # Model entity or model name
ModelVersionIdentifier = V | Tuple[str, str]  # ModelVersion entity or tuple of (name, version) string


class MlflowApiContract(pydantic.BaseModel, ABC):
    """Declaration of all standard public methods for MLflow API classes."""

    @abstractmethod
    def clean_all_cache(self):
        """Clean all cached metadata and artifacts."""

    @abstractmethod
    def clean_temp_artifacts(self):
        """Clean temporary artifacts."""

    @abstractmethod
    def clean_cached_run_artifact(self, run: RunIdentifier, path_in_run: str = ""):
        """Clean cached artifacts for specified run."""

    @abstractmethod
    def clean_cached_model_artifact(self, model_version: ModelVersionIdentifier):
        """Clean cached artifacts for specified model version."""

    @abstractmethod
    def list_run_artifacts(self, run: RunIdentifier, path_in_run: str = "") -> transfer.LsResult:
        """List run artifacts in repo."""

    @abstractmethod
    def list_model_artifact(self, model_version: ModelVersionIdentifier, path_suffix: str = "") -> transfer.LsResult:
        """List model version artifacts in repo."""

    @abstractmethod
    def cache_run_artifact(self, run: RunIdentifier, path_in_run: str = "") -> Path:
        """Pull run artifacts from MLflow server to local cache."""

    @abstractmethod
    def cache_model_artifact(self, model_version: ModelVersionIdentifier) -> Path:
        """Pull model version artifacts from MLflow server to local cache."""

    @abstractmethod
    def get_run_artifact(self, run: RunIdentifier, path_in_run: str = "") -> Path:
        """Get local path to run artifacts."""

    @abstractmethod
    def get_model_artifact(self, model_version: ModelVersionIdentifier) -> Path:
        """Get local path to model artifacts."""

    @abstractmethod
    def place_run_artifact(
        self,
        run: RunIdentifier,
        target: Path,
        path_in_run: str = "",
        overwrite: bool = False,
        link: bool = True,
    ):
        """Place run artifacts on target path.

        On online mode: Data is synced with the MLflow server.
        On offline mode: No guarantee that the data is current.
        """

    @abstractmethod
    def place_model_artifact(
        self,
        model_version: ModelVersionIdentifier,
        target: Path,
        overwrite: bool = False,
        link: bool = True,
    ):
        """Place model version artifacts on target path.

        On online mode: Data is synced with the MLflow server.
        On offline mode: No guarantee that the data is current.
        """

    @abstractmethod
    def export_run_artifact(
        self,
        run: RunIdentifier,
        target: Path,
        path_in_run: str = "",
    ) -> Path:
        """Export run artifact cache to target path."""

    @abstractmethod
    def export_model_artifact(
        self,
        model_version: ModelVersionIdentifier,
        target: Path,
    ) -> Path:
        """Export model version artifact cache to target path."""

    @abstractmethod
    def load_run_artifact(self, run: RunIdentifier, loader: Callable[[Path], A], path_in_run: str = "") -> A:
        """Load run artifacts."""

    @abstractmethod
    def load_model_artifact(self, model_version: ModelVersionIdentifier, loader: Callable[[Path], A]) -> A:
        """Load model artifacts."""

    @abstractmethod
    def log_run_artifact(
        self,
        run: RunIdentifier,
        source: Path | Callable[[Path], None],
        path_in_run: str | None = None,
        keep_the_source: bool | None = None,
        allow_duplication: bool | None = None,
        use_cache: bool | None = None,
    ):
        """Publish artifact file or dir to experiment run."""

    @abstractmethod
    def log_model_version(
        self,
        model: ModelIdentifier,
        run: RunIdentifier,
        source: Path | Callable[[Path], None],
        path_in_run: str | None = None,
        keep_the_source: bool | None = None,
        allow_duplication: bool = False,
        use_cache: bool | None = None,
        version: str | None = None,
        tags: Mapping | None = None,
    ) -> V:
        """Publish artifact file or dir as model version inside the specified experiment run."""

    @abstractmethod
    def get_exp_url(self, exp: ExpIdentifier) -> str:
        """Get Experiment URL."""

    @abstractmethod
    def get_run_url(self, run: RunIdentifier, exp: ExpIdentifier | None = None) -> str:
        """Get Run URL."""

    @abstractmethod
    def get_model_url(self, model: ModelIdentifier) -> str:
        """Get URL to registered model."""

    @abstractmethod
    def get_model_version_url(self, model_version: ModelVersionIdentifier) -> str:
        """Get model version URL."""

    @abstractmethod
    def get_exp(self, exp: ExpIdentifier, **cache_opts: bool) -> E:
        """Get Experiment API by ID."""

    @abstractmethod
    def get_run(self, run: RunIdentifier, **cache_opts: bool) -> R:
        """Get Run API by ID."""

    @abstractmethod
    def get_model(self, model: ModelIdentifier, **cache_opts: bool) -> M:
        """Get Experiment API by ID."""

    @abstractmethod
    def get_model_version(self, model_version: ModelVersionIdentifier, **cache_opts: bool) -> V:
        """Get ModelVersion metadata by ID."""

    @abstractmethod
    def find_exps(self, query: mongo.Query | None = None, sorting: mongo.Sorting | None = None) -> Iterator[E]:
        """Search experiments with query in MongoDB query language."""

    @abstractmethod
    def find_runs(self, query: mongo.Query | None = None, sorting: mongo.Sorting | None = None) -> Iterator[R]:
        """Search runs with query in MongoDB query language."""

    @abstractmethod
    def find_models(self, query: mongo.Query | None = None, sorting: mongo.Sorting | None = None) -> Iterator[M]:
        """Search registered models with query in MongoDB query language."""

    @abstractmethod
    def find_model_versions(
        self, query: mongo.Query | None = None, sorting: mongo.Sorting | None = None
    ) -> Iterator[V]:
        """Search runs with query in MongoDB query language."""

    @pydantic.validate_arguments
    def find_child_runs(self, parent: RunIdentifier) -> Iterator[R]:
        """Find child runs."""

    @abstractmethod
    def cache_exp_meta(self, exp: ExpIdentifier) -> E:
        """Get latest Experiment metadata and save to local cache."""

    @abstractmethod
    def cache_run_meta(self, run: RunIdentifier) -> R:
        """Get latest Run metadata and save to local cache."""

    @abstractmethod
    def cache_model_meta(self, model: ModelIdentifier) -> M:
        """Get latest Model metadata and save to local cache."""

    @abstractmethod
    def cache_model_version_meta(self, model_version: ModelVersionIdentifier) -> V:
        """Get latest model version metadata and save to local cache."""

    @abstractmethod
    def export_exp_meta(self, exp: ExpIdentifier, target: Path) -> E:
        """Export experiment metadata cache to target."""

    @abstractmethod
    def export_run_meta(self, run: RunIdentifier, target: Path) -> R:
        """Export run metadata cache to target."""

    @abstractmethod
    def export_model_meta(self, model: ModelIdentifier, target: Path) -> M:
        """Export model metadata cache to target."""

    @abstractmethod
    def export_model_version_meta(self, mv: ModelVersionIdentifier, target: Path) -> V:
        """Export model version metadata cache to target."""

    @abstractmethod
    def create_exp(self, name: str, tags: Mapping | None = None) -> E:
        """Create Experiment and return its API."""

    @abstractmethod
    def get_or_create_exp(self, name: str) -> E:
        """Get or create Experiment and return its API."""

    @abstractmethod
    def create_model(self, name: str, tags: Mapping | None = None) -> M:
        """Create registered model and return its API."""

    @abstractmethod
    def create_run(
        self,
        exp: ExpIdentifier,
        name: str | None = None,
        tags: Mapping | None = None,
        repo: str | urls.Url | None = None,
        parent: RunIdentifier | None = None,
    ) -> R:
        """Declare a new experiment run to be used later."""

    @abstractmethod
    def start_run(
        self,
        exp: ExpIdentifier,
        name: str | None = None,
        tags: Mapping | None = None,
        repo: str | urls.Url | None = None,
        parent: RunIdentifier | None = None,
    ) -> R:
        """Start a new experiment run."""

    @abstractmethod
    def resume_run(self, run: RunIdentifier) -> R:
        """Resume a previous experiment run."""

    @abstractmethod
    def end_run(self, run: RunIdentifier, succeeded: bool = True) -> R:
        """End experiment run."""

    @abstractmethod
    def set_tags_on_exp(self, exp: ExpIdentifier, tags: Mapping):
        """Set tags on experiment."""

    @abstractmethod
    def set_tags_on_run(self, run: RunIdentifier, tags: Mapping):
        """Set tags on experiment run."""

    @abstractmethod
    def set_tags_on_model(self, model: ModelIdentifier, tags: Mapping):
        """Set tags on registered model."""

    @abstractmethod
    def set_tags_on_model_version(self, model_version: ModelVersionIdentifier, tags: Mapping):
        """Set tags on model version."""

    @abstractmethod
    def log_params(self, run: RunIdentifier, params: Mapping):
        """Log params to experiment run."""

    @abstractmethod
    def log_metrics(self, run: RunIdentifier, metrics: Mapping):
        """Log metrics to experiment run."""
