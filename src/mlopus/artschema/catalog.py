import logging
from pathlib import Path
from typing import Dict, Iterator, Tuple

from mlopus.mlflow.api.base import BaseMlflowApi
from mlopus.utils import pydantic
from .specs import LoadArtifactSpec

logger = logging.getLogger(__name__)


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
    ) -> Dict[str, Path]:
        """Export artifacts and metadata caches while preserving cache structure.

        See also:
            - :meth:`LoadArtifactSpec.export`

        :param mlflow_api:
        :param artifact_specs:
        :param target: Cache export target path.
        :param verify: | Use schemas for verification after export.
                       | See :meth:`~mlopus.artschema.Dumper.verify`.
        """
        paths = {}

        for name, spec in cls._iter_specs(artifact_specs):
            logger.debug("Exporting artifact '%s'", name)
            paths[name] = (spec := artifact_specs[name]).using(mlflow_api).export(Path(target))

            if verify:
                target_api = mlflow_api.in_offline_mode.copy(update={"cache_dir": target})
                spec.using(target_api).load(dry_run=True)

        return paths

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
        return ((name, spec) for name, spec in artifact_specs.items() if name in cls.__fields__)
