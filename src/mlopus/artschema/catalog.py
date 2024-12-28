import logging
from pathlib import Path
from typing import Dict, Iterator, Tuple, TypeVar, Type, Generic, Any

from mlopus.mlflow.api.base import BaseMlflowApi
from mlopus.utils import pydantic, typing_utils
from . import helpers, framework
from .specs import LoadArtifactSpec

logger = logging.getLogger(__name__)

T = TypeVar("T")  # Type of artifact


class ArtifactsCatalog(pydantic.BaseModel):
    """Base class for artifact catalogs.

    Useful for type-safe loading/downloading/exporting
    artifacts based on parsed application settings.

    Example settings:

    .. code-block:: yaml

        foo:
            schema: package.module:Schema  # Schema specified explicitly by fully qualified class name
            subject:
                run_id: 12345678
                path_in_run: foo
        bar:
            schema: default  # Schema obtained by alias from model version tags or parent model tags
            subject:
                model_name: foo
                model_version: 3

    Example usage:

    .. code-block:: python

        # Load the YAML settings above
        artifact_specs: dict = ...

        # Declare an artifact catalog
        class ArtifactsCatalog(mlopus.artschema.ArtifactsCatalog):
            foo: FooArtifact
            bar: BarArtifact

        # Cache all artifacts and metadata and verify their files using the specified schemas
        ArtifactsCatalog.download(mlflow_api, artifact_specs)

        # Load all artifacts using the specified schemas
        artifacts_catalog = ArtifactsCatalog.load(mlflow_api, artifact_specs)

        artifacts_catalog.foo  # `FooArtifact`
        artifacts_catalog.bar  # `BarArtifact`

    In the example above, `artifact_specs` is implicitly parsed into a mapping of `str` to :class:`LoadArtifactSpec`,
    while the :attr:`~LoadArtifactSpec.subject` values of `foo` and `bar` are parsed into
    :class:`~mlopus.artschema.RunArtifact` and :class:`~mlopus.artschema.ModelVersionArtifact`, respectively.
    """

    @classmethod
    @pydantic.validate_arguments
    def load(
        cls,
        mlflow_api: BaseMlflowApi,
        artifact_specs: Dict[str, LoadArtifactSpec],
    ) -> "ArtifactsCatalog":
        """Load artifacts from specs using their respective schemas.

        See also:
            - :meth:`LoadArtifactSpec.load`

        :param mlflow_api:
        :param artifact_specs:
        """
        return cls.parse_obj(cls._load(mlflow_api, artifact_specs))

    @classmethod
    @pydantic.validate_arguments
    def download(
        cls,
        mlflow_api: BaseMlflowApi,
        artifact_specs: Dict[str, LoadArtifactSpec],
        verify: bool = True,
    ) -> Dict[str, Path]:
        """Cache artifacts and metadata and verify the files against the schemas.

        See also:
            - :meth:`LoadArtifactSpec.download`

        :param mlflow_api:
        :param artifact_specs:
        :param verify: | Use schemas for verification after download.
                       | See :meth:`~mlopus.artschema.Dumper.verify`.
        """
        paths = {}

        for name, spec in cls._iter_specs(artifact_specs):
            logger.debug("Downloading artifact '%s'", name)
            paths[name] = (spec := artifact_specs[name].using(mlflow_api)).download()
            spec.load(dry_run=True) if verify else None

        return paths

    @classmethod
    @pydantic.validate_arguments
    def export(
        cls,
        mlflow_api: BaseMlflowApi,
        artifact_specs: Dict[str, LoadArtifactSpec],
        target: Path | str,
        verify: bool = True,
    ) -> None:
        """Export artifacts and metadata caches while preserving cache structure.

        See also:
            - :meth:`LoadArtifactSpec.export`

        :param mlflow_api:
        :param artifact_specs:
        :param target: Cache export target path.
        :param verify: | Use schemas for verification after export.
                       | See :meth:`~mlopus.artschema.Dumper.verify`.
        """

        for key in cls.model_fields:
            try:
                (spec := artifact_specs[key]).using(mlflow_api).export(Path(target))

                if verify:
                    target_api = mlflow_api.in_offline_mode.model_copy(update={"cache_dir": target})
                    spec.using(target_api).load(dry_run=True)
            except BaseException as exc:
                raise RuntimeError(
                    f"Failed to export artifact files and/or associated metadata for field '{key}' of catalog '{cls}'"
                ) from exc

    @classmethod
    def verify(
        cls,
        mlflow_api: BaseMlflowApi,
        artifact_specs: Dict[str, LoadArtifactSpec],
    ) -> None:
        """Validate the artifact specs against this catalog and Python environment.

        .. versionadded:: 1.4

        - Assert that the Python package requirements are met for each artifact schema in the specs.
        - Assert that the return type of each artifact schema matches the expected type in the catalog.

        :param mlflow_api:
        :param artifact_specs:
        """
        for key in cls.model_fields:
            try:
                spec = artifact_specs[key].using(mlflow_api)
                schema, _ = helpers.resolve_schema_and_alias(
                    schema=spec.schema_,
                    skip_reqs_check=False,
                    subject=spec.entity_api,
                )
            except BaseException as exc:
                raise RuntimeError(f"Failed to resolve artifact schema for field '{key}' of catalog '{cls}'") from exc

            if not typing_utils.safe_issubclass(
                actual := Path if schema == framework._DummySchema() else schema.Artifact,  # noqa
                expected := cls._get_field_type(key),
            ):
                raise TypeError(
                    f"Artifact type mismatch for key '{key}' of catalog '{cls}' "
                    f"(expected_by_catalog={expected}, returned_by_schema={actual})"
                )

    @classmethod
    def _get_field_type(cls, field_name: str) -> Type[T]:
        """Get artifact type of field."""
        return cls.model_fields[field_name].annotation

    @classmethod
    @pydantic.validate_arguments
    def _load(
        cls,
        mlflow_api: BaseMlflowApi,
        artifact_specs: Dict[str, LoadArtifactSpec],
        dry_run: bool = False,
    ) -> dict:
        data = {}

        for name, spec in cls._iter_specs(artifact_specs):
            logger.debug("%s artifact '%s'", "Verifying" if dry_run else "Loading", name)
            data[name] = artifact_specs[name].using(mlflow_api).load(dry_run=dry_run)

        return data

    @classmethod
    def _iter_specs(cls, artifact_specs: Dict[str, LoadArtifactSpec]) -> Iterator[Tuple[str, LoadArtifactSpec]]:
        return ((name, spec) for name, spec in artifact_specs.items() if name in cls.model_fields)


class LoadedArtifact(pydantic.BaseModel, Generic[T]):
    """Loaded artifact with metadata.

    .. versionadded:: 1.4
    """

    value: T
    meta: LoadArtifactSpec

    @classmethod
    def get_artifact_type(cls) -> Type[T] | None:
        """Get the type of the loaded artifact value."""
        if type_params := cls.__pydantic_generic_metadata__.get("args"):
            return type_params[0]
        return None


class ArtifactsCatalogWithMetadata(ArtifactsCatalog):
    """Variant of :class:`ArtifactsCatalog` with added metadata to each field.

    A condition for usage is that every field must be typed as :class:`LoadedArtifact[T]`,
    were `T` is the actual artifact type returned by the configured artifact schema.

    .. versionadded:: 1.4

    Example usage:

    .. code-block:: python

        # Declare an artifact catalog
        class ArtifactsCatalog(mlopus.artschema.ArtifactsCatalogWithMetadata):
            foo: LoadedArtifact[Foo]
            bar: LoadedArtifact[Bar]

        # Load all artifacts using the specified schemas
        artifacts_catalog = ArtifactsCatalog.load(mlflow_api, artifact_specs)

        artifacts_catalog.foo.value  # `Foo(...)` instance
        artifacts_catalog.foo.meta.run.params  # Params of the artifact's source run on MLFlow
    """

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Validate that each field is typed as a `LoadedArtifact[A]`."""
        for field_name, field_info in cls.model_fields.items():
            cls._get_field_type(field_name)

    @classmethod
    def _get_field_type(cls, field_name: str) -> Type[T]:
        """Get artifact type of field."""
        if (typing_utils.safe_issubclass(field_type := cls.model_fields[field_name].annotation, LoadedArtifact)) and (
            artifact_type := field_type.get_artifact_type()  # noqa
        ):
            return artifact_type
        raise TypeError(f"Field '{field_name}' of catalog '{cls}' must typed as {LoadedArtifact}[{T}].")

    @classmethod
    def load(
        cls,
        mlflow_api: BaseMlflowApi,
        artifact_specs: Dict[str, LoadArtifactSpec],
    ) -> "ArtifactsCatalogWithMetadata":
        """Load artifacts catalog from specs using MLFlow API."""
        fields = {
            field_name: LoadedArtifact[cls._get_field_type(field_name)](
                value=field_value,
                meta=artifact_specs[field_name].using(mlflow_api),
            )
            for field_name, field_value in super()._load(mlflow_api, artifact_specs).items()
        }
        return cls.model_validate(fields)
