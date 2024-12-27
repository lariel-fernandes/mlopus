import contextlib
import logging
import os.path
import tempfile
import typing
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Type, TypeVar, Callable, Iterator, Tuple, Mapping

from mlopus.utils import pydantic, paths, urls, mongo, string_utils, iter_utils
from . import contract
from .common import schema, decorators, serde, exceptions, patterns, transfer
from .exp import ExpApi
from .model import ModelApi
from .mv import ModelVersionApi
from .run import RunApi

logger = logging.getLogger(__name__)

A = TypeVar("A")  # Any type

# Entity types
E = schema.Experiment
R = schema.Run
M = schema.Model
V = schema.ModelVersion
T = TypeVar("T", bound=schema.BaseEntity)

# Identifier types
ExpIdentifier = contract.ExpIdentifier | ExpApi
RunIdentifier = contract.RunIdentifier | RunApi
ModelIdentifier = contract.ModelIdentifier | ModelApi
ModelVersionIdentifier = contract.ModelVersionIdentifier | ModelVersionApi


class BaseMlflowApi(contract.MlflowApiContract, ABC, frozen=True):
    """Base class for API clients that use "MLflow-like" backends for experiment tracking and model registry.

    Important:
        Implementations of this interface are meant to be thread-safe and independent of env vars/globals,
        so multiple API instances can coexist in the same program if necessary.
    """

    cache_dir: Path | None = pydantic.Field(
        default=None,
        description=(
            "Root path for cached artifacts and metadata. "
            "If not specified, then a default is determined by the respective API plugin."
        ),
    )

    offline_mode: bool = pydantic.Field(
        default=False,
        description=(
            "If `True`, block all operations that attempt communication "
            "with the MLflow server (i.e.: only use cached metadata). "
            "Artifacts are still accessible if they are cached or if "
            ":attr:`pull_artifacts_in_offline_mode` is `True`."
        ),
    )

    pull_artifacts_in_offline_mode: bool = pydantic.Field(
        default=False,
        description=(
            "If `True`, allow pulling artifacts from storage to cache in offline mode. "
            "Useful if caching metadata only and pulling artifacts on demand "
            "(the artifact's URL must be known beforehand, e.g. by caching the metadata of its parent entity). "
        ),
    )

    temp_artifacts_dir: Path = pydantic.Field(
        default=None,
        description=(
            "Path for temporary artifacts that are stored by artifact dumpers before being published and preserved "
            "after a publish error (e.g.: an upload interruption). Defaults to a path inside the local cache."
        ),
    )

    cache_local_artifacts: bool = pydantic.Field(
        default=False,
        description=(
            "Use local cache even if the run artifacts repository is in the local file system. "
            "May be used for testing cache without connecting to a remote MLflow server."
            "Not recommended in production because of unecessary duplicated disk usage. "
        ),
    )

    always_pull_artifacts: bool = pydantic.Field(
        default=False,
        description=(
            "When accessing a cached artifact file or dir, re-sync it with the remote artifacts repository, even "
            "on a cache hit. Prevents accessing stale data if the remote artifact has been changed in the meantime. "
            "The default data transfer utility (based on rclone) is pretty efficient for syncing directories, but "
            "enabling this option may still add some overhead of calculating checksums if they contain many files."
        ),
    )

    file_transfer: transfer.FileTransfer = pydantic.Field(
        repr=False,
        default_factory=transfer.FileTransfer,
        description=(
            "Utility for uploading/downloading artifact files or dirs. Also used for listing files. Based on "
            "RClone by default. Users may replace this with a different implementation when subclassing the API."
        ),
    )

    entity_serializer: serde.EntitySerializer = pydantic.Field(
        repr=False,
        default_factory=serde.EntitySerializer,
        description=(
            "Utility for (de)serializing entity metadata (i.e.: exp, runs, models, versions)."
            "Users may replace this with a different implementation when subclassing the API."
        ),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Apply default to cache dir, expand and resolve
        if paths.is_cwd(cache := self.cache_dir or ""):
            cache = self._impl_default_cache_dir()
        pydantic.force_set_attr(self, key="cache_dir", val=cache.expanduser().resolve())

        # Apply default to temp artifacts dir, expand and resolve
        if paths.is_cwd(tmp := self.temp_artifacts_dir or ""):
            tmp = self._artifacts_cache.joinpath("temp")
        pydantic.force_set_attr(self, key="temp_artifacts_dir", val=tmp.expanduser().resolve())

    @property
    def in_offline_mode(self) -> "BaseMlflowApi":
        """Get an offline copy of this API."""
        return self.model_copy(update={"offline_mode": True})

    # =======================================================================================================
    # === Metadata cache locators ===========================================================================

    @property
    def _metadata_cache(self) -> Path:
        return self.cache_dir.joinpath("metadata")

    @property
    def _exp_cache(self) -> Path:
        return self._metadata_cache.joinpath("exp")

    @property
    def _run_cache(self) -> Path:
        return self._metadata_cache.joinpath("run")

    @property
    def _model_cache(self) -> Path:
        return self._metadata_cache.joinpath("model")

    @property
    def _mv_cache(self) -> Path:
        return self._metadata_cache.joinpath("mv")

    def _get_exp_cache(self, exp_id: str) -> Path:
        return self._exp_cache.joinpath(exp_id)

    def _get_run_cache(self, run_id: str) -> Path:
        return self._run_cache.joinpath(run_id)

    def _get_model_cache(self, name: str) -> Path:
        return self._model_cache.joinpath(patterns.encode_model_name(name))

    def _get_mv_cache(self, name: str, version: str) -> Path:
        return self._mv_cache.joinpath(patterns.encode_model_name(name), version)

    # =======================================================================================================
    # === Artifact cache locators ===========================================================================

    @property
    def _artifacts_cache(self) -> Path:
        return self.cache_dir.joinpath("artifacts")

    @property
    def _run_artifacts_cache(self) -> Path:
        return self._artifacts_cache.joinpath("runs")

    def _get_run_artifact_cache_path(
        self, run: RunIdentifier, path_in_run: str = "", allow_base_resolve: bool = True
    ) -> Path:
        path_in_run = self._valid_path_in_run(path_in_run, allow_empty=allow_base_resolve)
        return self._run_artifacts_cache.joinpath(self._coerce_run_id(run), path_in_run)

    def _get_temp_artifacts_dir(self) -> Path:
        return Path(tempfile.mkdtemp(dir=paths.ensure_is_dir(self.temp_artifacts_dir)))

    # =======================================================================================================
    # === Cache protection ==================================================================================

    @contextlib.contextmanager
    def _lock_run_artifact(self, run_id: str, path_in_run: str, allow_base_resolve: bool = True) -> Path:
        path_in_run = self._valid_path_in_run(path_in_run, allow_empty=allow_base_resolve)
        with paths.dir_lock(self._get_run_artifact_cache_path(run_id)) as path:
            yield path.joinpath(path_in_run)

    # =======================================================================================================
    # === Cache cleanup =====================================================================================

    def _clean_temp_artifacts(self):
        paths.ensure_non_existing(self.temp_artifacts_dir, force=True)

    def _clean_all_meta_cache(self):
        for path in (self._exp_cache, self._run_cache, self._model_cache, self._mv_cache):
            with paths.dir_lock(path):
                paths.ensure_empty_dir(path, force=True)

    def _clean_run_artifact(self, run_id: str, path_in_run: str = ""):
        with self._lock_run_artifact(run_id, path_in_run) as path:
            paths.ensure_non_existing(path, force=True)

    def _clean_all_runs_artifacts(self):
        with paths.dir_lock(self._run_artifacts_cache) as path:
            paths.ensure_non_existing(path, force=True)

    def _clean_all_cache(self):
        self._clean_all_meta_cache()
        self._clean_all_runs_artifacts()

    # =======================================================================================================
    # === Metadata Getters ==================================================================================

    @classmethod
    def _meta_fetcher(cls, fetcher: Callable[[A], T], args: A) -> Callable[[], T]:
        return lambda: fetcher(args)

    def _meta_cache_reader(self, path: Path, type_: Type[T]) -> Callable[[], T]:
        return lambda: self.entity_serializer.load(type_, path)

    def _meta_cache_writer(self, lock_dir: Path, path: Path, type_: Type[T]) -> Callable[[T], None]:
        def _write_meta_cache(meta: T):
            assert isinstance(meta, type_)
            with paths.dir_lock(lock_dir):
                self.entity_serializer.save(meta, path)

        return _write_meta_cache

    def _get_meta(
        self,
        fetcher: Callable[[], T],
        cache_reader: Callable[[], T],
        cache_writer: Callable[[T], None],
        force_cache_refresh: bool = False,
    ) -> T:
        """Get metadata."""
        if force_cache_refresh:
            if self.offline_mode:
                raise RuntimeError("Cannot refresh cache on offline mode.")
            cache_writer(meta := fetcher())
        elif not self.offline_mode:
            meta = fetcher()
        else:
            meta = cache_reader()

        return meta

    def _get_exp(self, exp_id: str, **cache_opts: bool) -> E:
        """Get Experiment metadata."""
        return self._get_meta(
            self._meta_fetcher(self._fetch_exp, exp_id),
            self._meta_cache_reader(cache := self._get_exp_cache(exp_id), E),
            self._meta_cache_writer(self._exp_cache, cache, E),
            **cache_opts,
        )

    def _get_run(self, run_id: str, **cache_opts: bool) -> R:
        """Get Run metadata."""
        return self._get_meta(
            self._meta_fetcher(self._fetch_run, run_id),
            self._meta_cache_reader(cache := self._get_run_cache(run_id), R),
            lambda run: [
                self._meta_cache_writer(self._run_cache, cache, R)(run),
                self._meta_cache_writer(self._exp_cache, self._get_exp_cache(run.exp.id), E)(run.exp),
            ][0],
            **cache_opts,
        )

    def _get_model(self, name: str, **cache_opts: bool) -> M:
        """Get Model metadata."""
        return self._get_meta(
            self._meta_fetcher(self._fetch_model, name),
            self._meta_cache_reader(cache := self._get_model_cache(name), M),
            self._meta_cache_writer(self._model_cache, cache, M),
            **cache_opts,
        )

    def _get_mv(self, name_and_version: Tuple[str, str], **cache_opts: bool) -> V:
        """Get ModelVersion metadata."""
        return self._get_meta(
            self._meta_fetcher(self._fetch_mv, name_and_version),
            self._meta_cache_reader(cache := self._get_mv_cache(*name_and_version), V),
            lambda mv: [
                self._meta_cache_writer(self._mv_cache, cache, V)(mv),
                self._meta_cache_writer(self._model_cache, self._get_model_cache(mv.model.name), M)(mv.model),
            ][0],
            **cache_opts,
        )

    @decorators.online
    def _fetch_exp(self, exp_id: str) -> E:
        return self._impl_fetch_exp(exp_id)

    @decorators.online
    def _fetch_run(self, run_id: str) -> R:
        return self._impl_fetch_run(run_id)

    @decorators.online
    def _fetch_model(self, name: str) -> M:
        return self._impl_fetch_model(name)

    @decorators.online
    def _fetch_mv(self, name_and_version: Tuple[str, str]) -> V:
        return self._impl_fetch_mv(name_and_version)

    def _export_meta(self, meta: T, cache: Path, target: Path):
        paths.ensure_only_parents(target := target / cache.relative_to(self.cache_dir), force=True)
        self.entity_serializer.save(meta, target)

    # =======================================================================================================
    # === Metadata Finders ==================================================================================

    def _find_experiments(self, query: mongo.Query, sorting: mongo.Sorting) -> Iterator[E]:
        return self._find_meta(self._exp_cache, E, query, sorting, self._impl_find_experiments)

    def _find_runs(self, query: mongo.Query, sorting: mongo.Sorting) -> Iterator[R]:
        return self._find_meta(self._run_cache, R, query, sorting, self._impl_find_runs)

    def _find_models(self, query: mongo.Query, sorting: mongo.Sorting) -> Iterator[M]:
        return self._find_meta(self._model_cache, M, query, sorting, self._impl_find_models)

    def _find_mv(self, query: mongo.Query, sorting: mongo.Sorting) -> Iterator[V]:
        return self._find_meta(self._mv_cache, V, query, sorting, self._impl_find_mv)

    def _find_meta(
        self,
        cache: Path,
        type_: Type[T],
        query: mongo.Query,
        sorting: mongo.Sorting,
        finder: Callable[[mongo.Query, mongo.Sorting], Tuple[mongo.Query, mongo.Sorting, iter_utils.Paginator[T]]],
    ) -> Iterator[T]:
        if self.offline_mode:
            logger.warning("Metadata search in offline mode may yield incomplete or stale results.")
            cached = (self.entity_serializer.load(type_, x) for x in paths.iter_files(cache))
            paginator = iter_utils.Paginator[T].single_page(cached)
        else:
            query, sorting, paginator = finder(query, sorting)

        if sorting:
            paginator = paginator.collapse()

        if query:
            paginator = paginator.map_pages(
                lambda results: list(
                    mongo.find_all(
                        results,
                        query=query,
                        sorting=sorting,
                        to_doc=lambda obj: obj.dict(),
                        from_doc=type_.parse_obj,
                    )
                )
            )

        for page in paginator:
            for result in page:
                yield result

    # =======================================================================================================
    # === Artifact getters ==================================================================================

    def _list_run_artifacts(self, run: RunIdentifier, path_in_run: str = "") -> transfer.LsResult:
        path_in_run = self._valid_path_in_run(path_in_run, allow_empty=True)

        if self.offline_mode and not self.pull_artifacts_in_offline_mode:
            logger.warning("Listing run artifacts in offline mode may yield incomplete or stale results.")
            subject = self._get_run_artifact_cache_path(run, path_in_run)
        elif urls.is_local(subject := urls.urljoin(self._coerce_run(run).repo, path_in_run)):
            subject = subject.path

        return self.file_transfer.ls(subject)

    def _pull_run_artifact(self, run: RunIdentifier, path_in_run: str) -> Path:
        """Pull artifact from run repo to local cache, unless repo is already local."""
        if self.offline_mode and not self.pull_artifacts_in_offline_mode:
            raise RuntimeError("Artifact pull is disabled.")

        path_in_run = self._valid_path_in_run(path_in_run, allow_empty=True)

        with self._lock_run_artifact((run := self._coerce_run(run)).id, path_in_run) as target:
            if urls.is_local(url := run.repo_url):
                source = Path(url.path) / path_in_run

                if self.cache_local_artifacts:
                    paths.place_path(source, target, mode="copy", overwrite=True)
                else:
                    logger.info("Run artifacts repo is local, nothing to pull.")
                    return source
            else:
                self.file_transfer.pull_files(urls.urljoin(run.repo_url, path_in_run), target)

            return target

    def _get_run_artifact(self, run: RunIdentifier, path_in_run: str) -> Path:
        """Get path to local run artifact, may trigger a cache pull."""
        path_in_run = self._valid_path_in_run(path_in_run, allow_empty=True)
        cache = self._get_run_artifact_cache_path(run, path_in_run)

        if (not self.offline_mode or self.pull_artifacts_in_offline_mode) and (
            not cache.exists() or self.always_pull_artifacts
        ):
            return self._pull_run_artifact(run, path_in_run)

        if not cache.exists():
            raise FileNotFoundError(cache)

        return cache

    def _place_run_artifact(
        self,
        run: RunIdentifier,
        path_in_run: str,
        target: Path,
        link: bool,
        overwrite: bool,
    ):
        """Place local run artifact on target path, may trigger a cache pull.

        The resulting files are always write-protected, but directories are not.
        """
        mode = typing.cast(paths.PathOperation, "link" if link else "copy")

        if (src := self._get_run_artifact(run, path_in_run)).is_dir() and link:
            paths.ensure_only_parents(target, force=overwrite)

            for dirpath, _, filenames in os.walk(src):  # Recursively create symbolic links for files
                relpath = Path(dirpath).relative_to(src)
                for filename in filenames:
                    paths.place_path(Path(dirpath) / filename, target / relpath / filename, mode, overwrite)
        else:
            paths.place_path(src, target, mode, overwrite)

        if target.is_dir() and not link:  # Recursively fix permissions of copied directories
            target.chmod(paths.Mode.rwx)

            for dirpath, dirnames, _ in os.walk(target):
                for dirname in dirnames:
                    Path(dirpath, dirname).chmod(paths.Mode.rwx)

    # =======================================================================================================
    # === Arguments pre-processing ==========================================================================

    def _coerce_exp(self, exp: ExpIdentifier) -> E:
        match exp:
            case E():
                return exp
            case str():
                return self._get_exp(exp)
            case _:
                raise TypeError("Expected Experiment, ExpApi or experiment ID as string.")

    @classmethod
    @string_utils.retval_matches(patterns.EXP_ID)
    def _coerce_exp_id(cls, exp: ExpIdentifier) -> str:
        match exp:
            case E():
                return exp.id
            case str():
                return exp
            case _:
                raise TypeError("Expected Experiment, ExpApi or experiment ID as string.")

    def _coerce_run(self, run: RunIdentifier) -> R:
        match run:
            case R():
                return run
            case str():
                return self._get_run(run)
            case _:
                raise TypeError("Expected Run, RunApi or run ID as string.")

    @classmethod
    @string_utils.retval_matches(patterns.RUN_ID)
    def _coerce_run_id(cls, run: RunIdentifier) -> str:
        match run:
            case R():
                return run.id
            case str():
                return run
            case _:
                raise TypeError("Expected Run, RunApi or run ID as string.")

    def _coerce_model(self, model: ModelIdentifier) -> M:
        match model:
            case M():
                return model
            case str():
                return self._get_model(model)
            case _:
                raise TypeError("Expected Model, ModelApi or model name as string.")

    @classmethod
    @string_utils.retval_matches(patterns.MODEL_NAME)
    def _coerce_model_name(cls, model: ModelIdentifier) -> str:
        match model:
            case M():
                return model.name
            case str():
                return model
            case _:
                raise TypeError("Expected Model, ModelApi or model name as string.")

    def _coerce_mv(self, mv: ModelVersionIdentifier) -> V:
        match mv:
            case V():
                return mv
            case (str(), str()):
                return self._get_mv(*mv)
            case _:
                raise TypeError("Expected ModelVersion or tuple or (name, version) strings.")

    @classmethod
    @string_utils.retval_matches(patterns.MODEL_NAME, index=0)
    @string_utils.retval_matches(patterns.MODEL_VERSION, index=1)
    def _coerce_mv_tuple(cls, mv: ModelVersionIdentifier) -> Tuple[str, str]:
        match mv:
            case V():
                return mv.model.name, mv.version
            case (str(), str()):
                return mv
            case _:
                raise TypeError("Expected ModelVersion or tuple or (name, version) strings.")

    @classmethod
    def _valid_path_in_run(cls, path_in_run: str, allow_empty: bool = False) -> str:
        """Validate the `path_in_run` for an artifact or model.

        - Cannot be empty, unless specified
        - Slashes are trimmed
        - Cannot do path climbing (e.g.: "../")
        - Cannot do current path referencing (e.g.: "./")

        Valid path example: "a/b/c"
        """
        if (path_in_run := str(path_in_run).strip("/")) and os.path.abspath(path := "/root/" + path_in_run) == path:
            return path_in_run
        if allow_empty and path_in_run == "":
            return path_in_run
        raise paths.IllegalPath(f"`path_in_run={path_in_run}`")

    # =======================================================================================================
    # === Experiment tracking ===============================================================================

    @decorators.online
    def _create_exp(self, name: str, tags: Mapping) -> E:
        """Create experiment and return its metadata."""
        return self._impl_create_exp(name, tags)

    @decorators.online
    def _create_model(self, name: str, tags: Mapping) -> M:
        """Create registered model and return its metadata."""
        return self._impl_create_model(self._coerce_model_name(name), tags)

    @decorators.online
    def _create_run(
        self,
        exp: ExpIdentifier,
        name: str | None,
        repo: urls.Url | str | None,
        tags: Mapping,
        status: schema.RunStatus,
        parent: RunIdentifier | None,
    ) -> R:
        """Create run with start at current UTC time."""
        if repo is not None:
            repo = urls.parse_url(repo, resolve_if_local=True)
        run_id = self._impl_create_run(
            self._coerce_exp_id(exp), name, repo, parent_run_id=self._coerce_run_id(parent) if parent else None
        )
        self._update_run_tags(run_id, tags)
        self._set_run_status(run_id, status)
        return self._get_run(run_id)

    @decorators.online
    def _set_run_status(self, run: RunIdentifier, status: schema.RunStatus):
        self._impl_set_run_status(self._coerce_run_id(run), status)

    @decorators.online
    def _set_run_end_time(self, run: RunIdentifier, end_time: datetime):
        self._impl_set_run_end_time(self._coerce_run_id(run), end_time)

    @decorators.online
    def _update_exp_tags(self, exp: ExpIdentifier, tags: Mapping):
        self._impl_update_exp_tags(self._coerce_exp_id(exp), tags)

    @decorators.online
    def _update_run_tags(self, run: RunIdentifier, tags: Mapping):
        self._impl_update_run_tags(self._coerce_run_id(run), tags)

    @decorators.online
    def _update_model_tags(self, model: ModelIdentifier, tags: Mapping):
        self._impl_update_model_tags(self._coerce_model_name(model), tags)

    @decorators.online
    def _update_mv_tags(self, mv: ModelVersionIdentifier, tags: Mapping):
        self._impl_update_mv_tags(*self._coerce_mv_tuple(mv), tags=tags)

    @decorators.online
    def _log_run_params(self, run: RunIdentifier, params: Mapping):
        self._impl_log_run_params(self._coerce_run_id(run), params)

    @decorators.online
    def _log_run_metrics(self, run: RunIdentifier, metrics: Mapping):
        self._impl_log_run_metrics(self._coerce_run_id(run), metrics)

    # =======================================================================================================
    # === Model registry ====================================================================================

    @decorators.online
    def _register_mv(
        self, model: ModelIdentifier, run: RunIdentifier, path_in_run: str, version: str | None, tags: Mapping
    ) -> V:
        path_in_run = self._valid_path_in_run(path_in_run)
        return self._impl_register_mv(self._coerce_model(model), self._coerce_run(run), path_in_run, version, tags)

    # =======================================================================================================
    # === Abstract Methods ==================================================================================

    @abstractmethod
    def _impl_default_cache_dir(self) -> Path:
        """Get default cache dir based on the current MLflow API settings."""

    @abstractmethod
    def _impl_get_exp_url(self, exp_id: str) -> urls.Url:
        """Get Experiment URL."""

    @abstractmethod
    def _impl_get_run_url(self, run_id: str, exp_id: str) -> urls.Url:
        """Get Run URL."""

    @abstractmethod
    def _impl_get_model_url(self, name: str) -> urls.Url:
        """Get URL to registered model."""

    @abstractmethod
    def _impl_get_mv_url(self, name: str, version: str) -> urls.Url:
        """Get model version URL."""

    @abstractmethod
    def _impl_fetch_exp(self, exp_id: str) -> E:
        """Get Experiment by ID."""

    @abstractmethod
    def _impl_fetch_run(self, run_id: str) -> R:
        """Get Run by ID."""

    @abstractmethod
    def _impl_fetch_model(self, name: str) -> M:
        """Get registered Model by name."""

    @abstractmethod
    def _impl_fetch_mv(self, name_and_version: Tuple[str, str]) -> V:
        """Get ModelVersion by name and version."""

    @abstractmethod
    def _impl_find_experiments(
        self, query: mongo.Query, sorting: mongo.Sorting
    ) -> Tuple[mongo.Query, mongo.Sorting, iter_utils.Paginator[E]]:
        """Push down MongoDB query where possible and return query remainder with results iterator."""

    @abstractmethod
    def _impl_find_runs(
        self, query: mongo.Query, sorting: mongo.Sorting
    ) -> Tuple[mongo.Query, mongo.Sorting, iter_utils.Paginator[R]]:
        """Push down MongoDB query where possible and return query remainder with results iterator."""

    @abstractmethod
    def _impl_find_models(
        self, query: mongo.Query, sorting: mongo.Sorting
    ) -> Tuple[mongo.Query, mongo.Sorting, iter_utils.Paginator[M]]:
        """Push down MongoDB query where possible and return query remainder with results iterator."""

    @abstractmethod
    def _impl_find_mv(
        self, query: mongo.Query, sorting: mongo.Sorting
    ) -> Tuple[mongo.Query, mongo.Sorting, iter_utils.Paginator[V]]:
        """Push down MongoDB query where possible and return query remainder with results iterator."""

    @abstractmethod
    def _impl_find_child_runs(self, run: R) -> Iterator[R]:
        """Find child runs."""

    @abstractmethod
    def _impl_create_exp(self, name: str, tags: Mapping) -> E:
        """Create experiment and return its metadata."""

    @abstractmethod
    def _impl_create_model(self, name: str, tags: Mapping) -> M:
        """Create registered model and return its metadata."""

    @abstractmethod
    def _impl_create_run(
        self, exp_id: str, name: str | None, repo: urls.Url | None, parent_run_id: str | None = None
    ) -> str:
        """Create experiment run."""

    @abstractmethod
    def _impl_set_run_status(self, run_id: str, status: schema.RunStatus):
        """Set Run status."""

    @abstractmethod
    def _impl_set_run_end_time(self, run_id: str, end_time: datetime):
        """Set Run end time."""

    @abstractmethod
    def _impl_update_exp_tags(self, exp_id: str, tags: Mapping):
        """Update Exp tags."""

    @abstractmethod
    def _impl_update_run_tags(self, run_id: str, tags: Mapping):
        """Update Run tags."""

    @abstractmethod
    def _impl_update_model_tags(self, name: str, tags: Mapping):
        """Update Model tags."""

    @abstractmethod
    def _impl_update_mv_tags(self, name: str, version: str, tags: Mapping):
        """Update Exp tags."""

    @abstractmethod
    def _impl_log_run_params(self, run_id: str, params: Mapping):
        """Log run params."""

    @abstractmethod
    def _impl_log_run_metrics(self, run_id: str, metrics: Mapping):
        """Log run metrics."""

    @abstractmethod
    def _impl_register_mv(self, model: M, run: R, path_in_run: str, version: str | None, tags: Mapping) -> V:
        """Register model version."""

    # =======================================================================================================
    # === Public Methods ====================================================================================

    def clean_all_cache(self):
        """Clean all cached metadata and artifacts."""
        self._clean_all_cache()

    def clean_temp_artifacts(self):
        """Clean temporary artifacts."""
        self._clean_temp_artifacts()

    @pydantic.validate_arguments
    def clean_cached_run_artifact(self, run: RunIdentifier, path_in_run: str = ""):
        """Clean cached artifact for specified run.

        :param run: Run ID or object.
        :param path_in_run: Plain relative path inside run artifacts (e.g.: `a/b/c`)
        """
        self._clean_run_artifact(self._coerce_run_id(run), path_in_run)

    @pydantic.validate_arguments
    def clean_cached_model_artifact(self, model_version: ModelVersionIdentifier):
        """Clean cached artifact for specified model version.

        :param model_version: Model version object or `(name, version)` tuple.
        """
        mv = self._coerce_mv(model_version)
        self.clean_cached_run_artifact(mv.run, mv.path_in_run)

    @pydantic.validate_arguments
    def list_run_artifacts(self, run: RunIdentifier, path_in_run: str = "") -> transfer.LsResult:
        """List run artifacts in repo.

        :param run: Run ID or object.
        :param path_in_run: Plain relative path inside run artifacts (e.g.: `a/b/c`)
        """
        return self._list_run_artifacts(run, path_in_run)

    @pydantic.validate_arguments
    def list_model_artifact(self, model_version: ModelVersionIdentifier, path_suffix: str = "") -> transfer.LsResult:
        """List model version artifacts in repo.

        :param model_version: Model version object or `(name, version)` tuple.
        :param path_suffix: Plain relative path inside model artifact dir (e.g.: `a/b/c`).
        """
        return self.list_run_artifacts(
            run=(mv := self._coerce_mv(model_version)).run,
            path_in_run=mv.path_in_run + "/" + path_suffix.strip("/"),
        )

    @pydantic.validate_arguments
    def cache_run_artifact(self, run: RunIdentifier, path_in_run: str = "") -> Path:
        """Pull run artifact from MLflow server to local cache.

        :param run: Run ID or object.
        :param path_in_run: Plain relative path inside run artifacts (e.g.: `a/b/c`)
        """
        return self._pull_run_artifact(run, path_in_run)

    @pydantic.validate_arguments
    def cache_model_artifact(self, model_version: ModelVersionIdentifier) -> Path:
        """Pull model version artifact from MLflow server to local cache.

        :param model_version: Model version object or `(name, version)` tuple.
        """
        mv = self._coerce_mv(model_version)
        return self.cache_run_artifact(mv.run, mv.path_in_run)

    @pydantic.validate_arguments
    def get_run_artifact(self, run: RunIdentifier, path_in_run: str = "") -> Path:
        """Get local path to run artifact.

        Triggers a cache pull on a cache miss or if :attr:`always_pull_artifacts`.

        :param run: Run ID or object.
        :param path_in_run: Plain relative path inside run artifacts (e.g.: `a/b/c`)
        """
        return self._get_run_artifact(self._coerce_run_id(run), path_in_run)

    @pydantic.validate_arguments
    def get_model_artifact(self, model_version: ModelVersionIdentifier) -> Path:
        """Get local path to model artifact.

        Triggers a cache pull on a cache miss or if :attr:`always_pull_artifacts`.

        :param model_version: Model version object or `(name, version)` tuple.
        """
        mv = self._coerce_mv(model_version)
        return self.get_run_artifact(mv.run, mv.path_in_run)

    @pydantic.validate_arguments
    def place_run_artifact(
        self,
        run: RunIdentifier,
        target: Path,
        path_in_run: str = "",
        overwrite: bool = False,
        link: bool = True,
    ):
        """Place run artifact on target path.

        Triggers a cache pull on a cache miss or if :attr:`always_pull_artifacts`.
        The resulting files are always write-protected, but directories are not.

        :param run: Run ID or object.
        :param target: Target path.
        :param path_in_run: Plain relative path inside run artifacts (e.g.: `a/b/c`)
        :param overwrite: Overwrite target path if exists.
        :param link: Use symbolic link instead of copy.
        """
        self._place_run_artifact(self._coerce_run_id(run), path_in_run, target, link, overwrite)

    @pydantic.validate_arguments
    def place_model_artifact(
        self,
        model_version: ModelVersionIdentifier,
        target: Path,
        overwrite: bool = False,
        link: bool = True,
    ):
        """Place model version artifact on target path.

        Triggers a cache pull on a cache miss or if :attr:`always_pull_artifacts`.

        :param model_version: Model version object or `(name, version)` tuple.
        :param target: Target path.
        :param overwrite: Overwrite target path if exists.
        :param link: Use symbolic link instead of copy.
        """
        mv = self._coerce_mv(model_version)
        self.place_run_artifact(mv.run, target, mv.path_in_run, overwrite, link)

    @pydantic.validate_arguments
    def export_run_artifact(
        self,
        run: RunIdentifier,
        target: Path,
        path_in_run: str = "",
    ) -> Path:
        """Export run artifact cache to target path while keeping the original cache structure.

        The target path can then be used as cache dir by the `generic` MLflow API in offline mode.

        :param run: Run ID or object.
        :param target: Cache export path.
        :param path_in_run: Plain relative path inside run artifacts (e.g.: `a/b/c`)
        """
        if paths.is_sub_dir(target, self.cache_dir) or paths.is_sub_dir(self.cache_dir, target):
            raise paths.IllegalPath(f"Cannot export cache to itself, its subdirs or parents: {target}")
        cache = self._get_run_artifact(run, path_in_run)
        target = self.model_copy(update={"cache_dir": target})._get_run_artifact_cache_path(run, path_in_run)
        paths.place_path(cache, target, mode="copy", overwrite=True)
        paths.rchmod(target, paths.Mode.rwx)  # Exported caches are not write-protected
        return target

    @pydantic.validate_arguments
    def export_model_artifact(
        self,
        model_version: ModelVersionIdentifier,
        target: Path,
    ) -> Path:
        """Export model version artifact cache to target path while keeping the original cache structure.

        The target path can then be used as cache dir by the `generic` MLflow API in offline mode.

        :param model_version: Model version object or `(name, version)` tuple.
        :param target: Cache export path.
        """
        mv = self._coerce_mv(model_version)
        return self.export_run_artifact(mv.run, target, mv.path_in_run)

    @pydantic.validate_arguments
    def load_run_artifact(self, run: RunIdentifier, loader: Callable[[Path], A], path_in_run: str = "") -> A:
        """Load run artifact.

        Triggers a cache pull on a cache miss or if :attr:`always_pull_artifacts`.

        :param run: Run ID or object.
        :param loader: Loader callback.
        :param path_in_run: Plain relative path inside run artifacts (e.g.: `a/b/c`)
        """
        return loader(self._get_run_artifact(self._coerce_run_id(run), path_in_run))

    @pydantic.validate_arguments
    def load_model_artifact(self, model_version: ModelVersionIdentifier, loader: Callable[[Path], A]) -> A:
        """Load model version artifact.

        Triggers a cache pull on a cache miss or if :attr:`always_pull_artifacts`.

        :param model_version: Model version object or `(name, version)` tuple.
        :param loader: Loader callback.
        """
        mv = self._coerce_mv(model_version)
        logger.info("Loading model: %s v%s", mv.model.name, mv.version)
        return self.load_run_artifact(mv.run, loader, mv.path_in_run)

    @decorators.online
    @pydantic.validate_arguments
    def log_run_artifact(
        self,
        run: RunIdentifier,
        source: Path | Callable[[Path], None],
        path_in_run: str | None = None,
        keep_the_source: bool | None = None,
        allow_duplication: bool | None = None,
        use_cache: bool | None = None,
    ):
        """Publish artifact file or dir to experiment run.

        The flags :paramref:`keep_the_source`, :paramref:`allow_duplication` and :paramref:`use_cache` are
        experimental and may conflict with one another. It is recommended to leave them unspecified, so this
        method will do a best-effort to use cache if it makes sense to, keep the source files if it makes
        sense to (possibly as a symbolic link) and avoid duplicated disk usage when possible.

        :param run: | Run ID or object.

        :param source: | Path to artifact file or dir, or a dumper callback.
                       | If it's a callback and the upload is interrupted, the temporary artifact is kept.

        :param path_in_run: Plain relative path inside run artifacts (e.g.: `a/b/c`)

            - If `source` is a `Path`: Defaults to file or dir name.
            - If `source` is a callback: No default available.

        :param keep_the_source:
            - If `source` is a `Path`: Keep that file or dir (defaults to `True`).
            - If `source` is a callback: Keep the temporary artifact, even after a successful upload (defaults to `False`).

        :param allow_duplication: | If `False`, a `source` file or dir may be replaced with a symbolic link to the local cache in order to avoid duplicated disk usage.
                                  | Defaults to `True` if :paramref:`keep_the_source` is `True` and the run artifacts repo is local.

        :param use_cache: | If `True`, keep artifact in local cache after publishing.
                          | Defaults to `True` if the run artifacts repo is remote.
        """
        tmp = None

        if using_dumper := callable(source):
            logger.debug("Using temporary artifact path: %s", tmp := self._get_temp_artifacts_dir())
            source(source := tmp.joinpath("artifact"))

        if (source := Path(source).expanduser().resolve()).is_relative_to(self._run_artifacts_cache):
            raise paths.IllegalPath(f"Source path points to artifact cache: {source}")

        if keep_the_source is None:
            keep_the_source = False if using_dumper else True  # noqa: SIM211

        try:
            if using_dumper:
                assert path_in_run, "When using an artifact dumper, `path_in_run` must be specified."
            else:
                path_in_run = path_in_run or source.name
            path_in_run = self._valid_path_in_run(path_in_run, allow_empty=False)

            run = self._coerce_run(run)
            target = urls.urljoin(run.repo_url, path_in_run)

            if repo_is_local := urls.is_local(target):
                if use_cache is None:
                    use_cache = False

                if allow_duplication is None:
                    allow_duplication = True if keep_the_source or use_cache else False  # noqa: SIM211,SIM210

                if keep_the_source:
                    if allow_duplication:
                        mode = "copy"
                    else:
                        raise RuntimeError("Cannot keep the source without duplication when artifacts repo is local")
                else:
                    mode = "move"

                paths.place_path(
                    source,
                    target.path,
                    mode=typing.cast(paths.PathOperation, mode),
                    overwrite=True,
                    move_abs_links=False,
                )
            else:
                if use_cache is None:
                    use_cache = True

                if allow_duplication is None:
                    allow_duplication = False

                self.file_transfer.push_files(source, target)
        except BaseException as exc:
            raise exceptions.FailedToPublishArtifact(source) from exc

        logger.debug(f"Artifact successfully published to '{target}'")

        if use_cache:
            with self._lock_run_artifact(run.id, path_in_run, allow_base_resolve=False) as cache:
                if repo_is_local:
                    if allow_duplication:
                        paths.place_path(target.path, cache, mode="copy", overwrite=True)
                    else:
                        raise RuntimeError("Cannot cache artifact without duplication when run artifacts repo is local")
                elif keep_the_source:
                    if allow_duplication:
                        paths.place_path(source, cache, mode="copy", overwrite=True)
                    else:
                        logger.warning("Keeping the `source` as a symbolic link to the cached artifact")
                        logger.debug(f"{source} -> {cache}")
                        paths.place_path(source, cache, mode="move", overwrite=True)
                        paths.place_path(cache, source, mode="link", overwrite=True)
                else:
                    paths.place_path(source, cache, mode="move", overwrite=True)

        if not keep_the_source:
            paths.ensure_non_existing(source, force=True)
            if tmp is not None:
                paths.ensure_non_existing(tmp, force=True)

    @pydantic.validate_arguments
    def log_model_version(
        self,
        model: ModelIdentifier,
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

        :param model: | Model name or object.

        :param run: | Run ID or object.

        :param source: | See :paramref:`log_run_artifact.source`

        :param path_in_run: | Plain relative path inside run artifacts (e.g.: `a/b/c`).
                            | Defaults to model name.

        :param keep_the_source: | See :paramref:`log_run_artifact.keep_the_source`

        :param allow_duplication: | See :paramref:`log_run_artifact.allow_duplication`

        :param use_cache: | See :paramref:`log_run_artifact.use_cache`

        :param version: | Arbitrary model version
                        | (not supported by all backends).

        :param tags: | Model version tags.
                     | See :class:`schema.ModelVersion.tags`

        :return: New model version metadata with API handle.
        """
        logger.info("Logging version of model '%s'", model_name := self._coerce_model_name(model))
        path_in_run = path_in_run or patterns.encode_model_name(model_name)
        self.log_run_artifact(run, source, path_in_run, keep_the_source, allow_duplication, use_cache)
        return ModelVersionApi(**self._register_mv(model, run, path_in_run, version, tags or {})).using(self)

    @pydantic.validate_arguments
    def get_exp_url(self, exp: ExpIdentifier) -> str:
        """Get Experiment URL.

        :param exp: Exp ID or object.
        """
        return str(self._impl_get_exp_url(self._coerce_exp_id(exp)))

    @pydantic.validate_arguments
    def get_run_url(self, run: RunIdentifier, exp: ExpIdentifier | None = None) -> str:
        """Get Run URL.

        :param run: Run ID or object.
        :param exp: Exp ID or object.

        Caveats:
            - :paramref:`exp` must be specified on :attr:`~BaseMlflowApi.offline_mode`
              if :paramref:`run` is an ID and the run metadata is not in cache.
        """
        exp = self._coerce_run(run).exp if exp is None else exp
        return str(self._impl_get_run_url(self._coerce_run_id(run), self._coerce_exp_id(exp)))

    @pydantic.validate_arguments
    def get_model_url(self, model: ModelIdentifier) -> str:
        """Get URL to registered model.

        :param model: Model name or object.
        """
        return str(self._impl_get_model_url(self._coerce_model_name(model)))

    @pydantic.validate_arguments
    def get_model_version_url(self, model_version: ModelVersionIdentifier) -> str:
        """Get model version URL.

        :param model_version: Model version object or `(name, version)` tuple.
        """
        return str(self._impl_get_mv_url(*self._coerce_mv_tuple(model_version)))

    @pydantic.validate_arguments
    def get_exp(self, exp: ExpIdentifier, **cache_opts: bool) -> ExpApi:
        """Get Experiment API by ID.

        :param exp: Exp ID or object.
        """
        return ExpApi(**self._get_exp(self._coerce_exp_id(exp), **cache_opts)).using(self)

    @pydantic.validate_arguments
    def get_run(self, run: RunIdentifier, **cache_opts: bool) -> RunApi:
        """Get Run API by ID.

        :param run: Run ID or object.
        """
        return RunApi(**self._get_run(self._coerce_run_id(run), **cache_opts)).using(self)

    @pydantic.validate_arguments
    def get_model(self, model: ModelIdentifier, **cache_opts: bool) -> ModelApi:
        """Get Model API by name.

        :param model: Model name or object.
        """
        return ModelApi(**self._get_model(self._coerce_model_name(model), **cache_opts)).using(self)

    @pydantic.validate_arguments
    def get_model_version(self, model_version: ModelVersionIdentifier, **cache_opts: bool) -> ModelVersionApi:
        """Get ModelVersion API by name and version.

        :param model_version: Model version object or `(name, version)` tuple.
        """
        return ModelVersionApi(**self._get_mv(self._coerce_mv_tuple(model_version), **cache_opts)).using(self)

    @pydantic.validate_arguments
    def find_exps(self, query: mongo.Query | None = None, sorting: mongo.Sorting | None = None) -> Iterator[ExpApi]:
        """Search experiments with query in MongoDB query language.

        :param query: Query in MongoDB query language.
        :param sorting: Sorting criteria (e.g.: `[("asc_field", 1), ("desc_field", -1)]`).
        """
        return (ExpApi(**x).using(self) for x in self._find_experiments(query or {}, sorting or []))

    @pydantic.validate_arguments
    def find_runs(self, query: mongo.Query | None = None, sorting: mongo.Sorting | None = None) -> Iterator[RunApi]:
        """Search runs with query in MongoDB query language.

        :param query: Query in MongoDB query language.
        :param sorting: Sorting criteria (e.g.: `[("asc_field", 1), ("desc_field", -1)]`).
        """
        return (RunApi(**x).using(self) for x in self._find_runs(query or {}, sorting or []))

    @pydantic.validate_arguments
    def find_models(self, query: mongo.Query | None = None, sorting: mongo.Sorting | None = None) -> Iterator[ModelApi]:
        """Search registered models with query in MongoDB query language.

        :param query: Query in MongoDB query language.
        :param sorting: Sorting criteria (e.g.: `[("asc_field", 1), ("desc_field", -1)]`).
        """
        return (ModelApi(**x).using(self) for x in self._find_models(query or {}, sorting or []))

    @pydantic.validate_arguments
    def find_model_versions(
        self, query: mongo.Query | None = None, sorting: mongo.Sorting | None = None
    ) -> Iterator[ModelVersionApi]:
        """Search model versions with query in MongoDB query language.

        :param query: Query in MongoDB query language.
        :param sorting: Sorting criteria (e.g.: `[("asc_field", 1), ("desc_field", -1)]`).
        """
        return (ModelVersionApi(**x).using(self) for x in self._find_mv(query or {}, sorting or []))

    @pydantic.validate_arguments
    def find_child_runs(self, parent: RunIdentifier) -> Iterator[RunApi]:
        """Find child runs.

        :param parent: Run ID or object.
        """
        return (RunApi(**x).using(self) for x in self._impl_find_child_runs(self._coerce_run(parent)))

    @pydantic.validate_arguments
    def cache_exp_meta(self, exp: ExpIdentifier) -> ExpApi:
        """Get latest Experiment metadata and save to local cache.

        :param exp: Experiment ID or object.
        """
        return self.get_exp(exp, force_cache_refresh=True)

    @pydantic.validate_arguments
    def cache_run_meta(self, run: RunIdentifier) -> RunApi:
        """Get latest Run metadata and save to local cache.

        :param run: Run ID or object.
        """
        return self.get_run(run, force_cache_refresh=True)

    @pydantic.validate_arguments
    def cache_model_meta(self, model: ModelIdentifier) -> ModelApi:
        """Get latest Model metadata and save to local cache.

        :param model: Model name or object.
        """
        return self.get_model(model, force_cache_refresh=True)

    @pydantic.validate_arguments
    def cache_model_version_meta(self, model_version: ModelVersionIdentifier) -> ModelVersionApi:
        """Get latest model version metadata and save to local cache.

        :param model_version: Model version object or `(name, version)` tuple.
        """
        return self.get_model_version(model_version, force_cache_refresh=True)

    @pydantic.validate_arguments
    def export_exp_meta(self, exp: ExpIdentifier, target: Path) -> ExpApi:
        """Export experiment metadata cache to target.

        :param exp: Experiment ID or object.
        :param target: Cache export path.
        """
        self._export_meta(exp := self.get_exp(id_ := self._coerce_exp_id(exp)), self._get_exp_cache(id_), target)
        return exp

    @pydantic.validate_arguments
    def export_run_meta(self, run: RunIdentifier, target: Path) -> RunApi:
        """Export run metadata cache to target.

        :param run: Run ID or object.
        :param target: Cache export path.
        """
        self._export_meta(run := self.get_run(id_ := self._coerce_run_id(run)), self._get_run_cache(id_), target)
        self.export_exp_meta(run.exp, target)
        return run

    @pydantic.validate_arguments
    def export_model_meta(self, model: ModelIdentifier, target: Path) -> ModelApi:
        """Export model metadata cache to target.

        :param model: Model name or object.
        :param target: Cache export path.
        """
        name = self._coerce_model_name(model)
        self._export_meta(model := self.get_model(name), self._get_model_cache(name), target)
        return model

    @pydantic.validate_arguments
    def export_model_version_meta(self, mv: ModelVersionIdentifier, target: Path) -> ModelVersionApi:
        """Export model version metadata cache to target.

        :param mv: Model version object or `(name, version)` tuple.
        :param target: Cache export path.
        """
        tup = self._coerce_mv_tuple(mv)
        self._export_meta(mv := self.get_model_version(tup), self._get_mv_cache(*tup), target)
        self.export_model_meta(mv.model, target)
        return mv

    @pydantic.validate_arguments
    def create_exp(self, name: str, tags: Mapping | None = None) -> ExpApi:
        """Create Experiment and return its API.

        :param name: See :attr:`schema.Experiment.name`.
        :param tags: See :attr:`schema.Experiment.tags`.
        """
        return ExpApi(**self._create_exp(name, tags or {})).using(self)

    @pydantic.validate_arguments
    def get_or_create_exp(self, name: str) -> ExpApi:
        """Get or create Experiment and return its API.

        :param name: See :attr:`schema.Experiment.name`.
        """
        for exp in self._find_experiments({"name": name}, []):
            break
        else:
            exp = self._create_exp(name, tags={})

        return ExpApi(**exp).using(self)

    @pydantic.validate_arguments
    def create_model(self, name: str, tags: Mapping | None = None) -> ModelApi:
        """Create registered model and return its API.

        :param name: See :attr:`schema.Model.name`.
        :param tags: See :attr:`schema.Model.tags`.
        """
        return ModelApi(**self._create_model(name, tags or {})).using(self)

    @pydantic.validate_arguments
    def get_or_create_model(self, name: str) -> ModelApi:
        """Get or create registered Model and return its API.

        :param name: See :attr:`schema.Model.name`.
        """
        for model in self._find_models({"name": name}, []):
            break
        else:
            model = self._create_model(name, tags={})

        return ModelApi(**model).using(self)

    @pydantic.validate_arguments
    def create_run(
        self,
        exp: ExpIdentifier,
        name: str | None = None,
        tags: Mapping | None = None,
        repo: str | urls.Url | None = None,
        parent: RunIdentifier | None = None,
    ) -> RunApi:
        """Declare a new experiment run to be used later.

        :param exp: Experiment ID or object.
        :param name: See :attr:`schema.Run.name`.
        :param tags: See :attr:`schema.Run.tags`.
        :param repo: (Experimental) Cloud storage URL to be used as alternative run artifacts repository.
        :param parent: Parent run ID or object.
        """
        return RunApi(**self._create_run(exp, name, repo, tags or {}, schema.RunStatus.SCHEDULED, parent)).using(self)

    @pydantic.validate_arguments
    def start_run(
        self,
        exp: ExpIdentifier,
        name: str | None = None,
        tags: Mapping | None = None,
        repo: str | urls.Url | None = None,
        parent: RunIdentifier | None = None,
    ) -> RunApi:
        """Start a new experiment run.

        :param exp: Experiment ID or object.
        :param name: See :attr:`schema.Run.name`.
        :param tags: See :attr:`schema.Run.tags`.
        :param repo: (Experimental) Cloud storage URL to be used as alternative run artifacts repository.
        :param parent: Parent run ID or object.
        """
        return RunApi(**self._create_run(exp, name, repo, tags or {}, schema.RunStatus.RUNNING, parent)).using(self)

    @pydantic.validate_arguments
    def resume_run(self, run: RunIdentifier) -> RunApi:
        """Resume a previous experiment run.

        :param run: Run ID or object.
        """
        self._set_run_status(run_id := self._coerce_run_id(run), schema.RunStatus.RUNNING)
        return self.get_run(run_id)

    @pydantic.validate_arguments
    def end_run(self, run: RunIdentifier, succeeded: bool = True) -> RunApi:
        """End experiment run.

        :param run: Run ID or object.
        :param succeeded: Whether the run was successful.
        """
        status = schema.RunStatus.FINISHED if succeeded else schema.RunStatus.FAILED
        self._set_run_status(run_id := self._coerce_run_id(run), status)
        self._set_run_end_time(run_id, datetime.now())
        return self.get_run(run_id)

    @pydantic.validate_arguments
    def set_tags_on_exp(self, exp: ExpIdentifier, tags: Mapping):
        """Set tags on experiment.

        :param exp: Experiment ID or object.
        :param tags: See :attr:`schema.Experiment.tags`.
        """
        self._update_exp_tags(exp, tags)

    @pydantic.validate_arguments
    def set_tags_on_run(self, run: RunIdentifier, tags: Mapping):
        """Set tags on experiment run.

        :param run: Run ID or object.
        :param tags: See :attr:`schema.Run.tags`.
        """
        self._update_run_tags(run, tags)

    @pydantic.validate_arguments
    def set_tags_on_model(self, model: ModelIdentifier, tags: Mapping):
        """Set tags on registered model.

        :param model: Model name or object.
        :param tags: See :attr:`schema.Model.tags`.
        """
        self._update_model_tags(model, tags)

    @pydantic.validate_arguments
    def set_tags_on_model_version(self, model_version: ModelVersionIdentifier, tags: Mapping):
        """Set tags on model version.

        :param model_version: Model version object or `(name, version)` tuple.
        :param tags: See :attr:`schema.Model.tags`.
        """
        self._update_mv_tags(model_version, tags)

    @pydantic.validate_arguments
    def log_params(self, run: RunIdentifier, params: Mapping):
        """Log params to experiment run.

        :param run: Run ID or object.
        :param params: See :attr:`schema.Run.params`.
        """
        self._log_run_params(run, params)

    @pydantic.validate_arguments
    def log_metrics(self, run: RunIdentifier, metrics: Mapping):
        """Log metrics to experiment run.

        :param run: Run ID or object.
        :param metrics: See :attr:`schema.Run.metrics`.
        """
        self._log_run_metrics(run, metrics)
