import logging
import warnings
from typing import Dict, Type, TypeVar, Set, Sequence

import importlib_metadata

import mlopus.mlflow
from mlopus.utils import pydantic, packaging, dicts, import_utils, typing_utils
from .framework import Schema

logger = logging.getLogger(__name__)

DEFAULT_ALIAS = "default"  # Default artifact schema alias

DEFAULT_CONSTRAINT = ">="  # Default constraint for package version

T = TypeVar("T")  # Any type

Entity = (
    mlopus.mlflow.schema.Experiment
    | mlopus.mlflow.schema.Run
    | mlopus.mlflow.schema.Model
    | mlopus.mlflow.schema.ModelVersion
)

API = mlopus.mlflow.ExpApi | mlopus.mlflow.RunApi | mlopus.mlflow.ModelApi | mlopus.mlflow.ModelVersionApi


def _maybe_warn_editable_dist(cls: str, dist: importlib_metadata.Distribution):
    if packaging.is_editable_dist(dist):
        warnings.warn(
            UserWarning(
                f"Class '{cls}' depends on package '{dist.name} v{dist.version}' "
                f"which was installed from editable source code."
            )
        )


class PkgSpec(pydantic.BaseModel, pydantic.MappingMixin):
    """Specifies a required Python package with version constraint and required extras."""

    name: str
    version: str
    extras: Set[str] | None = None
    constraint: packaging.VersionConstraint

    @pydantic.model_validator(mode="after")
    def _validate_version(self):
        if "+" in self.version and self.constraint != "==":
            warnings.warn(
                UserWarning(
                    "Semantic version with build metadata suffix cannot be used with range constraint "
                    f"'{self.constraint}', replacing with a pinned constraint: =={self.version}"
                )
            )
            self.constraint = "=="
        return self

    @property
    def dist(self) -> packaging.Dist:
        """Get currently installed distribution of this package requirement."""
        return packaging.get_dist(self.name)

    def check_requirement(self):
        """Validate this package requirement against the current Python environment."""
        if not packaging.check_dist(self.dist, specifier=self.constraint + self.version):
            raise RuntimeError(f"Python requirement not matched: {self.name}{self.constraint}{self.version}")

    def check_extras(self):
        """Validate that this package is installed with the required extras."""
        if missing := packaging.get_missing_extras(self.dist, self.extras):
            raise RuntimeError(f"Missing the following extras for package '{self.name}': {missing}")

    def __iter__(self):
        return (x for x in super().__iter__() if x != "extras" or self.extras)


class ClassSpec(pydantic.BaseModel, pydantic.MappingMixin):
    """Class specification with package requirement info."""

    cls: str
    pkg: PkgSpec

    def load(self, type_: Type[T] | None = None, skip_reqs_check: bool = False) -> Type[T]:
        """Load class by fully qualified name.

        :param type_: If specified, the loaded class must inherit this type.
        :param skip_reqs_check: Ignore package requirements for the loaded class.
        :return: Loaded class as subclass of :paramref:`type_`
        """
        if not skip_reqs_check:
            self.pkg.check_requirement()
            self.pkg.check_extras()
            _maybe_warn_editable_dist(self.cls, self.pkg.dist)

        cls = import_utils.find_type(self.cls)

        if type_ is not None:
            typing_utils.assert_issubclass(cls, type_)

        return cls

    @classmethod
    @pydantic.validate_arguments
    def parse_class(
        cls,
        cls_: type,
        from_package: str | None = None,
        at_version: str | None = None,
        with_constraint: packaging.VersionConstraint = DEFAULT_CONSTRAINT,
        and_extras: Sequence[str] | None = None,
    ) -> "ClassSpec":
        """Parse class specification."""
        dist = None
        get_dist = lambda: packaging.pkg_dist_of_cls(cls_)  # noqa: E731

        pkg = from_package or (dist := get_dist()).name
        version = at_version or (dist or (dist := get_dist())).version

        if (extras := set(and_extras) if and_extras else None) and (
            not_available := extras.difference(packaging.get_available_dist_extras(dist or get_dist()))
        ):
            raise ValueError(f"Requested extras are not provided by package distribution '{pkg}': {not_available}")

        return cls(
            cls=import_utils.fq_name(cls_),
            pkg=PkgSpec(
                name=pkg,
                extras=extras,
                version=version,
                constraint=with_constraint,
            ),
        )


class Tags(pydantic.BaseModel, pydantic.MappingMixin):
    """Artifact schema tags for MLflow experiments, runs, registered models and model versions.

    Example:

    .. code-block:: python

        # Get a model version API.
        version = mlopus.mlflow \\
            .get_api(...) \\
            .get_model(...) \\
            .get_version(...)

        # Register a schema for that version.
        mlopus.artschema.Tags() \\
            .using(MySchema, aliased_as="my_schema") \\
            .register(version)

        # Inspect the resulting tags.
        version.tags["schemas"]
        {
            "my_schema": {
                "cls": "my_schemas:MySchema",
                "pkg": {  # this package requirement will be validated when importing the schema class
                    "name": "my_schemas",
                    "version": "1.2.3",
                    "constraint": ">=",
                    "extras": [],
                }
            }
        }

        # Load the model using the aliased schema
        mlopus.artschema.load_artifact(version, schema="my_schema")

        # Package requirements can be ignored with `skip_reqs_check=True`
        # Package details are inferred automatically unless specified when calling `using(...)`.
    """

    schemas: Dict[str, ClassSpec] = pydantic.Field(
        default_factory=dict, description="Mapping of alias to artifact schema class specification."
    )

    @pydantic.validate_arguments
    def using(
        self,
        cls: Type[Schema],
        aliased_as: str | None = None,
        from_package: str | None = None,
        at_version: str | None = None,
        with_constraint: packaging.VersionConstraint = DEFAULT_CONSTRAINT,
        and_extras: Sequence[str] | None = None,
    ) -> "Tags":
        """Add aliased artifact schema to this `Tags` object.

        If not specified, the params :paramref:`from_package` and :paramref:`at_version` are inferred from
        the metadata of the provided :paramref:`cls` and the packages installed in the current environment.

        :param cls: Artifact schema class.
        :param aliased_as: Schema alias. Defaults to `default`.
        :param from_package: Required package for this schema.
        :param at_version: Required package version.
        :param with_constraint: Version requirement constraint.
        :param and_extras: Required package extras for using the schema.
        """
        if (alias := aliased_as or DEFAULT_ALIAS) in self.schemas:
            raise ValueError(f"Found duplicated schema alias: {alias}")
        self.schemas[alias] = ClassSpec.parse_class(cls, from_package, at_version, with_constraint, and_extras)
        return self

    def register(self, subject: API):
        """Register these artifact schema tags for the specified :paramref:`subject`.

        :param subject: | Experiment, run, model or model version with API handle.
        """
        logger.info("Registering artifact schemas for %s\n%s", subject, self.model_dump_json(indent=4))
        subject.set_tags(self)

    def get_schema(self, alias: str | None = None) -> ClassSpec:
        """Get artifact schema by alias."""
        if not (cls_spec := self.schemas.get(alias := alias or DEFAULT_ALIAS)):
            raise ValueError(f"No artifact schema configured with alias '{alias}'")
        return cls_spec

    @classmethod
    def parse_subject(cls, subject: Entity) -> "Tags":
        """Parse artifact schema tags from :paramref:`subject`.

        - Runs inherit schemas of their parent experiment.
        - Model versions inherit schemas of their parent model.

        :param subject: | Experiment, run, model or model version.
        """
        key = "schemas"

        match subject:
            case mlopus.mlflow.schema.Model() | mlopus.mlflow.schema.Experiment():
                tags = subject.tags.get(key, {})
            case mlopus.mlflow.schema.Run():
                # Merge schema tags from run and parent exp (run takes precedence)
                tags = dicts.deep_merge(subject.exp.tags.get(key, {}), subject.tags.get(key, {}))
            case mlopus.mlflow.schema.ModelVersion():
                # Merge schema tags from model version and parent model (version takes precedence)
                tags = dicts.deep_merge(subject.model.tags.get(key, {}), subject.tags.get(key, {}))
            case _:
                raise TypeError(subject)

        return cls.parse_obj({key: tags})
