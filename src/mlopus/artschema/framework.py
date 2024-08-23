import functools
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar, Generic, Type, Callable

from mlopus.utils import pydantic, paths, typing_utils, json_utils

logger = logging.getLogger(__name__)

A = TypeVar("A", bound=object)  # Type of artifact


class Dumper(pydantic.BaseModel, ABC, Generic[A]):
    """Base class for artifact dumpers."""

    class Config(pydantic.BaseModel.Config):
        """Class-level config."""

        dumper_conf_file: str = "dumper_conf.json"

    # =======================================================================================================
    # === Abstract methods ==================================================================================

    @abstractmethod
    def _dump(self, path: Path, artifact: A) -> None:
        """Save artifact to `path` as file or dir."""

    @abstractmethod
    def _verify(self, path: Path) -> None:
        """Verify the `path` where the artifact was dumped."""

    # =======================================================================================================
    # === Public methods ====================================================================================

    def dump(self, path: Path, artifact: A, overwrite: bool = False) -> None:
        """Save artifact to `path` as file or dir."""
        self._dump(artifact=self._pre_dump(artifact), path=(path := paths.ensure_only_parents(path, force=overwrite)))
        self.verify(path)
        self.maybe_save_conf(path, strict=True)

    def verify(self, path: Path) -> None:
        """Verify the `path` where the artifact was dumped."""
        if path.exists():
            self._verify(path)
        else:
            raise FileNotFoundError(path)

    def maybe_save_conf(self, path: Path, strict: bool):
        """If path is a dir, save a file with dumper conf."""
        if path.is_file():
            logger.warning("Artifact dump is not a directory, dumper conf file will not be saved.")
        elif path.is_dir():
            if (conf_path := path / self.Config.dumper_conf_file).exists():
                if strict:
                    raise FileExistsError(conf_path)
            else:
                self._save_conf_to(conf_path)
        else:
            raise FileNotFoundError(path)

    # =======================================================================================================
    # === Private methods ===================================================================================

    def _pre_dump(self, artifact: A | dict) -> A:
        if isinstance(artifact, dict) and (model := pydantic.as_model_cls(self._artifact_type)):
            artifact = model.parse_obj(artifact)
        return artifact

    def _save_conf_to(self, path: Path):
        path.write_text(self._serialize_conf())

    def _serialize_conf(self) -> str:
        return json_utils.dumps(self.dict())

    # =======================================================================================================
    # === Type param inference ==============================================================================

    @property
    def _artifact_type(self) -> Type[A]:
        """Artifact type used by this schema."""
        return self._get_artifact_type()

    @classmethod
    def _get_artifact_type(cls) -> Type[A]:
        """Infer artifact type used by this schema."""
        base = typing_utils.find_base(cls, lambda b: typing_utils.safe_issubclass(b, Dumper))
        return typing_utils.get_type_param(base, object, pos=0, strict=True)


class _DummyDumper(Dumper[object]):
    """Dummy dumper with verification bypass and no dumping logic."""

    def _dump(self, path: Path, artifact: object) -> None:
        raise NotImplementedError()

    def _verify(self, path: Path) -> None:
        pass


D = TypeVar("D", bound=Dumper)  # Type of dumper


class Loader(pydantic.BaseModel, ABC, Generic[A, D]):
    """Base class for artifact loaders."""

    # =======================================================================================================
    # === Abstract methods ==================================================================================

    @abstractmethod
    def _load(self, path: Path, dumper: D) -> A | dict:
        """Load artifact from `path`."""

    # =======================================================================================================
    # === Public methods ====================================================================================

    def load(self, path: Path, dry_run: bool = False) -> A | Path:
        """Load artifact from `path`."""
        (dumper := self._load_dumper(path)).verify(path)

        if dry_run:
            return path

        return self._post_load(self._load(path, dumper))

    # =======================================================================================================
    # === Private methods ===================================================================================

    def _post_load(self, artifact: A | dict) -> A:
        if isinstance(artifact, dict) and (model := pydantic.as_model_cls(self._artifact_type)):
            artifact = model.parse_obj(artifact)
        return artifact

    def _load_dumper(self, path: Path) -> D:
        if (dumper_conf_file := path / self._dumper_type.Config.dumper_conf_file).exists():
            dumper_conf = json_utils.loads(dumper_conf_file.read_text())
        else:
            dumper_conf = {}

        try:
            return self._dumper_type.parse_obj(dumper_conf)
        except pydantic.ValidationError as exc:
            logger.error(
                "Could not parse dumper with type '%s' (an anonymous pydantic class will be used instead): %s",
                *(self._dumper_type, exc),
            )
            return pydantic.create_obj_from_data(name="AnonymousDumper", data=dumper_conf, __base__=_DummyDumper)

    # =======================================================================================================
    # === Type param inference ==============================================================================

    @property
    def _artifact_type(self) -> Type[A]:
        """Artifact type used by this schema."""
        return self._get_artifact_type()

    @property
    def _dumper_type(self) -> Type[D]:
        """Dumper class used by this schema."""
        return self._get_dumper_type()

    @classmethod
    def _get_artifact_type(cls) -> Type[A]:
        """Infer artifact type used by this schema."""
        base = typing_utils.find_base(cls, lambda b: typing_utils.safe_issubclass(b, Loader))
        return typing_utils.get_type_param(base, object, pos=0, strict=True)

    @classmethod
    def _get_dumper_type(cls) -> Type[D]:
        """Infer dumper class used by this schema."""
        base = typing_utils.find_base(cls, lambda b: typing_utils.safe_issubclass(b, Loader))
        return typing_utils.get_type_param(base, Dumper, pos=1, strict=True)


class _DummyLoader(Loader[object, _DummyDumper]):
    """Dummy loader with no loading logic."""

    def _load(self, path: Path, dumper: _DummyDumper) -> object | dict:
        return path


L = TypeVar("L", bound=Loader)  # Type of loader


class Schema(pydantic.BaseModel, Generic[A, D, L]):
    """Base class for artifact schemas.

    Example:

    .. code-block:: python

        class Artifact(pydantic.BaseModel):
            some_data: dict[str, str]


        class Dumper(mlopus.artschema.Dumper[Artifact]):
            encoding: str = "UTF-8"

            def _dump(self, path: Path, artifact: A) -> None:
                # save `artifact.some_data` inside `path` using `self.encoding`


        class Loader(mlopus.artschema.Loader[Artifact, Dumper]):
            max_files: int | None = None

            def _load(self, path: Path, dumper: Dumper) -> Artifact:
                # load instance of `Artifact` from `path` using `self.max_files` and `dumper.encoding`


        class Schema(mlopus.artschema.Schema[Artifact, Dumper, Loader]):
            pass  # No methods needed here, but the type params are important!


        # Instantiate
        artifact = Artifact(some_data={...})

        # Dump
        dumper = Schema().get_dumper(artifact, encoding="...")
        dumper(path)

        # Load
        loader = Schema().get_loader(max_files=3)
        loader(path)  # Returns: Artifact

        # Combine with MlflowApi
        with mlopus.mlflow \\
            .get_api(...) \\
            .get_exp(...) \\
            .start_run(...):

            run.log_artifact(dumper, path_in_run="foobar")

        run.load_artifact(loader, path_in_run="foobar")
        # Same applies when using `log_model_version` and `load_model_artifact`
    """

    # =======================================================================================================
    # === Public methods ====================================================================================

    def get_dumper(
        self, artifact: A | dict | Path, dumper: D | dict | None = None, **dumper_kwargs
    ) -> Callable[[Path], None] | Path:
        """Get dumper callback for artifact data.

        - If artifact is a `Path`, it will be just verified and returned as it is.
        - If artifact is a `dict` and the target type is a pydantic model, it will be parsed before being dumped.
        """
        assert dumper is None or not dumper_kwargs, "`dumper` and `dumper_kwargs` are not compatible."

        if not isinstance(dumper, self.Dumper):
            dumper = self.Dumper.parse_obj(dumper_kwargs or dumper or {})

        if isinstance(artifact, Path):
            dumper.verify(path := artifact)
            dumper.maybe_save_conf(path, strict=False)
            return path

        return functools.partial(dumper.dump, artifact=artifact)

    def get_loader(
        self, loader: L | dict | None = None, dry_run: bool = False, **loader_kwargs
    ) -> Callable[[Path], A | Path]:
        """Get loader callback for artifact file(s).

        - If `dry_run=True`, the callback will just verify the given `Path` and return it as it is.
        - If the loaded artifact is a `dict` and the target type is a pydantic model, the callback will parse it.
        """
        assert loader is None or not loader_kwargs, "`loader` and `loader_kwargs` are not compatible."

        if not isinstance(loader, self.Loader):
            loader = self.Loader.parse_obj(loader_kwargs or loader or {})

        return functools.partial(loader.load, dry_run=dry_run)

    # =======================================================================================================
    # === Type param inference ==============================================================================

    @property
    def Artifact(self) -> Type[A]:  # noqa
        """Artifact type used by this schema."""
        return self._get_artifact_type()

    @property
    def Dumper(self) -> Type[D]:  # noqa
        """Dumper class used by this schema."""
        return self._get_dumper_type()

    @property
    def Loader(self) -> Type[L]:  # noqa
        """Loader class used by this schema."""
        return self._get_loader_type()

    @classmethod
    def _get_artifact_type(cls) -> Type[A]:
        """Infer artifact type used by this schema."""
        base = typing_utils.find_base(cls, lambda b: typing_utils.safe_issubclass(b, Schema))
        return typing_utils.get_type_param(base, object, pos=0, strict=True)

    @classmethod
    def _get_dumper_type(cls) -> Type[D]:
        """Infer dumper class used by this schema."""
        base = typing_utils.find_base(cls, lambda b: typing_utils.safe_issubclass(b, Schema))
        return typing_utils.get_type_param(base, Dumper, pos=1, strict=True)

    @classmethod
    def _get_loader_type(cls) -> Type[L]:
        """Infer loader class used by this schema."""
        base = typing_utils.find_base(cls, lambda b: typing_utils.safe_issubclass(b, Schema))
        return typing_utils.get_type_param(base, Loader, pos=2, strict=True)


class _DummySchema(Schema[object, _DummyDumper, _DummyLoader]):
    """Schema with verification bypass and no dump/load logic."""
