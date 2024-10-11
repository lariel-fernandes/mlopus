import functools
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar, Generic, Type, Callable

from mlopus.utils import pydantic, paths, json_utils

logger = logging.getLogger(__name__)

A = TypeVar("A", bound=object)
"""Type of artifact"""


class Dumper(pydantic.BaseModel, pydantic.BaseParamsMixin, ABC, Generic[A]):
    """Base class for artifact dumpers."""

    class Config(pydantic.BaseModel.Config):
        """Class-level config."""

        dumper_conf_file: str = "dumper_conf.json"

    # =======================================================================================================
    # === Abstract methods ==================================================================================

    @abstractmethod
    def _dump(self, path: Path, artifact: A) -> None:
        """Save artifact to :paramref:`path` as file or dir.

        :param path: When this method is called, it is guaranteed that this
                     path doesn't exist yet and that it's parent is a dir.

        :param artifact: An instance of :attr:`.Artifact`.
        """

    @abstractmethod
    def _verify(self, path: Path) -> None:
        """Verify the :paramref:`path`.

        :param path: Path where an instance of :attr:`.Artifact` is
                     supposed to have been dumped, downloaded or placed.

        :raises AssertionError: Unless :paramref:`path` is a file
                                or dir in the expected structure.
        """

    # =======================================================================================================
    # === Public methods ====================================================================================

    def dump(self, path: Path | str, artifact: A | dict, overwrite: bool = False) -> None:
        """Save artifact to :paramref:`path` as file or dir.

        If possible, also saves a file with this dumper's conf.

        :param path: Target path.
        :param artifact:
            - An instance of :attr:`.Artifact`.
            - A `dict` that can be parsed into an :attr:`.Artifact` (in case :attr:`.Artifact` is a Pydantic model)

        :param overwrite: Overwrite the :paramref:`path` if exists.
        """
        self._dump(artifact=self._pre_dump(artifact), path=(path := paths.ensure_only_parents(path, force=overwrite)))
        self.verify(path)
        self.maybe_save_conf(path, strict=True)

    def verify(self, path: Path | str) -> None:
        """Verify the :paramref:`path`.

        :param path: Path where an instance of :attr:`.Artifact` is
                     supposed to have been dumped, downloaded or placed.

        :raises AssertionError: Unless :paramref:`path` is a file
                                or dir in the expected structure.
        """
        if (path := Path(path)).exists():
            self._verify(path)
        else:
            raise FileNotFoundError(path)

    def maybe_save_conf(self, path: Path, strict: bool):
        """If :paramref:`path` is a dir, save a file with this dumper's conf.

        :param path:
        :param strict:
        :raises FileNotFoundError: If :paramref:`path` doesn't exist.
        :raises FileExistsError: If :paramref:`strict` is `True` and a dumper
                                 conf file already exists in :paramref:`path`.
        """
        conf = self.dict()

        if path.is_file():
            if conf:  # Do not warn if there's no conf to be saved
                logger.warning("Artifact dump is not a directory, dumper conf file will not be saved.")
        elif path.is_dir():
            if (conf_path := path / self.Config.dumper_conf_file).exists():
                if strict:
                    raise FileExistsError(conf_path)
            elif conf:  # Do not save if there's no conf to be saved
                conf_path.write_text(json_utils.dumps(conf))
        else:
            raise FileNotFoundError(path)

    # =======================================================================================================
    # === Private methods ===================================================================================

    def _pre_dump(self, artifact: A | dict) -> A:
        if isinstance(artifact, dict) and (model := pydantic.as_model_cls(self.Artifact)):
            artifact = model.model_validate(artifact)
        return artifact

    # =======================================================================================================
    # === Type param inference ==============================================================================

    @property
    def Artifact(self) -> Type[A]:  # noqa
        """Artifact type used by this dumper.

        :return: Type of :attr:`~mlopus.artschema.framework.A`
        """
        return self._get_artifact_type()

    @classmethod
    def _get_artifact_type(cls) -> Type[A]:
        """Infer artifact type used by this dumper."""
        return cls._find_base_param(of_base=Dumper, at_pos=0, as_type_of=object)


class _DummyDumper(Dumper[object]):
    """Dummy dumper with verification bypass and no dumping logic."""

    def _dump(self, path: Path, artifact: object) -> None:
        raise NotImplementedError()

    def _verify(self, path: Path) -> None:
        pass


D = TypeVar("D", bound=Dumper)
"""Type of :class:`Dumper`"""


class Loader(pydantic.BaseModel, pydantic.BaseParamsMixin, ABC, Generic[A, D]):
    """Base class for artifact loaders."""

    # =======================================================================================================
    # === Abstract methods ==================================================================================

    @abstractmethod
    def _load(self, path: Path, dumper: D) -> A | dict:
        """Load artifact from :paramref:`path`.

        :param path: Path to artifact file or dir.
        :param dumper:
            - If :paramref:`path` is a dir containing a dumper conf file, this param will be an instance
              of :attr:`.Dumper` equivalent to the one that was originally used to save the artifact.
            - Otherwise, it will be a :attr:`.Dumper` initialized with empty params.

        :return:
            - An instance of :attr:`.Artifact`.
            - A `dict` that can be parsed into an :attr:`.Artifact` (in case :attr:`.Artifact` is a Pydantic model)
        """

    # =======================================================================================================
    # === Public methods ====================================================================================

    def load(self, path: Path | str, dry_run: bool = False) -> A | Path:
        """Load artifact from :paramref:`path`.

        As a side effect, this will use a :attr:`.Dumper` to :meth:`~Dumper.verify` the :paramref:`path`.
        If :paramref:`path` is a dir containing a dumper conf file, the used :attr:`.Dumper` is equivalent to the one
        that was originally used to save the artifact. Otherwise, it's a :attr:`.Dumper` initialized with empty params.

        :param path: Path to artifact file or dir.
        :param dry_run: Just verify the artifact path.
        :return:
            - If :paramref:`dry_run` is `True`, the same :paramref:`path`.
            - Otherwise, an instance of :attr:`.Artifact`.
        """
        (dumper := self._load_dumper(path := Path(path))).verify(path)

        if dry_run:
            return path

        return self._post_load(self._load(path, dumper))

    # =======================================================================================================
    # === Private methods ===================================================================================

    def _post_load(self, artifact: A | dict) -> A:
        if isinstance(artifact, dict) and (model := pydantic.as_model_cls(self.Artifact)):
            artifact = model.model_validate(artifact)
        return artifact

    def _load_dumper(self, path: Path) -> D:
        if (dumper_conf_file := path / self.Dumper.Config.dumper_conf_file).exists():
            dumper_conf = json_utils.loads(dumper_conf_file.read_text())
        else:
            dumper_conf = {}

        try:
            return self.Dumper.parse_obj(dumper_conf)
        except pydantic.ValidationError as exc:
            logger.error(
                "Could not parse dumper with type '%s' (an anonymous pydantic class will be used instead): %s",
                *(self.Dumper, exc),
            )
            return pydantic.create_obj_from_data(name="AnonymousDumper", data=dumper_conf, __base__=_DummyDumper)

    # =======================================================================================================
    # === Type param inference ==============================================================================

    @property
    def Artifact(self) -> Type[A]:  # noqa
        """Artifact type used by this loader.

        :return: Type of :attr:`~mlopus.artschema.framework.A`
        """
        return self._get_artifact_type()

    @property
    def Dumper(self) -> Type[D]:  # noqa
        """Dumper type used by this loader.

        :return: Type of :attr:`~mlopus.artschema.framework.D`
        """
        return self._get_dumper_type()

    @classmethod
    def _get_artifact_type(cls) -> Type[A]:
        """Infer artifact type used by this loader."""
        return cls._find_base_param(of_base=Loader, at_pos=0, as_type_of=object)

    @classmethod
    def _get_dumper_type(cls) -> Type[D]:
        """Infer dumper class used by this loader."""
        return cls._find_base_param(of_base=Loader, at_pos=1, as_type_of=Dumper)


class _DummyLoader(Loader[object, _DummyDumper]):
    """Dummy loader with no loading logic."""

    def _load(self, path: Path, dumper: _DummyDumper) -> object | dict:
        return path


L = TypeVar("L", bound=Loader)
"""Type of :class:`Loader`"""


class Schema(pydantic.BaseModel, pydantic.BaseParamsMixin, Generic[A, D, L]):
    """Base class for artifact schemas.

    Serves for putting together the types of :class:`Artifact`, :class:`Dumper` and :class:`Loader`.

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
        """Get a dumper callback.

        :param artifact:

            - An instance of :attr:`.Artifact`
            - A `Path` to a file or dir containing a pre-dumped :attr:`.Artifact`
            - A `dict` that can be parsed into an :attr:`.Artifact` (in case :attr:`.Artifact` is a Pydantic model)

        :param dumper: Custom :attr:`.Dumper` configuration. Defaults to an empty `dict`.

            - An instance of :attr:`.Dumper`
            - A `dict` that can be parsed into a :attr:`.Dumper`

        :param dumper_kwargs: | Keyword arguments for instantiating a :attr:`.Dumper`.
                              | Incompatible with the :paramref:`dumper` param.

        :return:

            - If :paramref:`artifact` is a `Path`: The same `Path` after being verified by the configured :attr:`.Dumper`.
            - Otherwise: A callback that accepts a `Path` and uses the configured :attr:`.Dumper` to dump the provided :attr:`.Artifact` on it.
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
        """Get a loader callback.

        :param loader: Custom :attr:`.Loader` configuration. Defaults to an empty `dict`.

            - An instance of :attr:`.Loader`
            - A `dict` that can be parsed into a :attr:`.Loader`

        :param loader_kwargs: | Keyword arguments for instantiating a :attr:`.Loader`.
                              | Incompatible with the :paramref:`loader` param.

        :param dry_run: | See :paramref:`Loader.load.dry_run`.

        :return:

            - If :paramref:`dry_run` is `True`: A callback that accepts a `Path`, verifies it and returns it.
            - Otherwise: A callback that accepts a `Path` and uses the configured :attr:`.Loader` to load and return an :attr:`.Artifact`
        """
        assert loader is None or not loader_kwargs, "`loader` and `loader_kwargs` are not compatible."

        if not isinstance(loader, self.Loader):
            loader = self.Loader.parse_obj(loader_kwargs or loader or {})

        return functools.partial(loader.load, dry_run=dry_run)

    # =======================================================================================================
    # === Type param inference ==============================================================================

    @property
    def Artifact(self) -> Type[A]:  # noqa
        """Artifact type used by this schema.

        :return: Type of :attr:`~mlopus.artschema.framework.A`
        """
        return self._get_artifact_type()

    @property
    def Dumper(self) -> Type[D]:  # noqa
        """:class:`Dumper` type used by this schema.

        :return: Type of :attr:`~mlopus.artschema.framework.D`
        """
        return self._get_dumper_type()

    @property
    def Loader(self) -> Type[L]:  # noqa
        """:class:`Loader` type used by this schema.

        :return: Type of :attr:`~mlopus.artschema.framework.L`
        """
        return self._get_loader_type()

    @classmethod
    def _get_artifact_type(cls) -> Type[A]:
        """Infer artifact type used by this schema."""
        return cls._find_base_param(of_base=Schema, at_pos=0, as_type_of=object)

    @classmethod
    def _get_dumper_type(cls) -> Type[D]:
        """Infer dumper class used by this schema."""
        return cls._find_base_param(of_base=Schema, at_pos=1, as_type_of=Dumper)

    @classmethod
    def _get_loader_type(cls) -> Type[L]:
        """Infer loader class used by this schema."""
        return cls._find_base_param(of_base=Schema, at_pos=2, as_type_of=Loader)


class _DummySchema(Schema[object, _DummyDumper, _DummyLoader]):
    """Schema with verification bypass and no dump/load logic."""
