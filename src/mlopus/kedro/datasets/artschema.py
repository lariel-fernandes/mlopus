from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar, Generic, Any, Type, Dict

from kedro.io import AbstractDataset

import mlopus
from mlopus.artschema import helpers, Dumper, Loader, Schema
from mlopus.mlflow.api.entity import EntityApi
from mlopus.mlflow.api.exp import ExpApi
from mlopus.mlflow.api.model import ModelApi
from mlopus.mlflow.api.mv import ModelVersionApi
from mlopus.mlflow.api.run import RunApi
from mlopus.utils import pydantic, paths

A = TypeVar("A", bound=object)  # Type of artifact
D = TypeVar("D", bound=Dumper)  # Type of Dumper
L = TypeVar("L", bound=Loader)  # Type of Loader

T = TypeVar("T", bound=EntityApi)  # Type of entity API


class SchemaSubject(mlopus.mlflow.MlflowApiMixin, pydantic.ExcludeEmptyMixin, ABC, Generic[T]):
    """Subject for artifact schema inference by alias."""

    def with_defaults(self, **__) -> "SchemaSubject":
        """Complement eventually missing fields in schema subject using the provided defaults."""
        return self

    @abstractmethod
    def resolve(self) -> EntityApi:
        """Get entity API."""


class ExpSubject(SchemaSubject[ExpApi]):
    """Specifies an experiment as subject for artifact schema inferrence."""

    exp_name: str = pydantic.Field(
        description=(
            "Experiment name. "
            "Defaults to the experiment used in the :attr:`~mlopus.kedro.ArtifactSchemaDataset.run_manager`"
        ),
    )

    def with_defaults(self, exp_name: str | None = None, **__) -> "ExpSubject":
        """Complement eventually missing fields in schema subject using the provided defaults."""
        self.exp_name = self.exp_name or exp_name
        return self

    def resolve(self) -> ExpApi:
        """Get entity API."""
        return self.mlflow_api.get_or_create_exp(self.exp_name)


class RunSubject(SchemaSubject[RunApi]):
    """Specifies a run as subject for artifact schema inferrence."""

    run_id: str = pydantic.Field(
        description="Run ID. Defaults to the run used in the :attr:`~mlopus.kedro.ArtifactSchemaDataset.run_manager`",
    )

    def with_defaults(self, run_id: str | None = None, **__) -> "RunSubject":
        """Complement eventually missing fields in schema subject using the provided defaults."""
        self.run_id = self.run_id or run_id
        return self

    def resolve(self, default_run_id: str | None = None, **__) -> RunApi:
        """Get entity API."""
        return self.mlflow_api.get_run(self.run_id or default_run_id)


M = TypeVar("M", ModelApi, ModelVersionApi)  # Type of subject API for `ModelSubject`


class ModelSubject(SchemaSubject[M]):
    """Specifies a registered model or model version as subject for artifact schema inferrence."""

    model_name: str = pydantic.Field(description="Registered model name.")
    model_version: str | None = pydantic.Field(default=None, description="Model version.")

    def resolve(self, **__) -> ModelApi | ModelVersionApi:
        model = self.mlflow_api.get_model(self.model_name)
        return model.get_version(v) if (v := self.model_version) else model


@pydantic.validate_arguments()
def _parse_subject(subj: ExpSubject | RunSubject | ModelSubject | None) -> SchemaSubject | None:
    return subj


class SchemaInfo(pydantic.BaseModel, pydantic.ExcludeEmptyMixin, Generic[A, D, L]):
    """Schema information for tracking purposes."""

    cls: Type[Schema]
    alias: str | None
    subject: SchemaSubject | None = None
    reqs_checked: bool | None = None

    _parse_subject = pydantic.validator("subject", pre=True, allow_reuse=True)(_parse_subject)


class ArtifactSchemaDataset(
    mlopus.mlflow.MlflowRunMixin,
    pydantic.EmptyStrAsMissing,
    pydantic.EmptyDictAsMissing,
    pydantic.ExcludeEmptyMixin,
    AbstractDataset[A, A],
    Generic[A, D, L],
):
    """Saves/loads data using inferred or explicitly specified artifact schema.

    See also:
        - :mod:`mlopus.artschema`

    Usage with explicit schema
    ==========================

        .. code-block:: yaml

            # conf/<env>/catalog.yml
            model:
                type: mlopus.kedro.ArtifactSchemaDataset
                path: data/model
                schema: my_package.artschema:TorchModelSchema  # fully qualified class name

    Usage with inferred schema
    ==========================

    1. Register the schema in the model's tags (also valid for experiments, runs and model versions):

        .. code-block:: python

            import mlopus

            mlopus.artschema.Tags() \\
                .with(
                    my_package.artschema.TorchModelSchema,
                    aliased_as="torch_model",
                ) \\
                .register(
                    mlopus.mlflow.get_api().get_or_create_model("my_lang_model")
                )

    2. Reference the schema by alias:

        .. code-block:: yaml

            # conf/<env>/catalog.yml
            model:
                type: mlopus.kedro.ArtifactSchemaDataset
                path: data/model
                schema: torch_model                   # Get schema with this alias
                subject: {model_name: my_lang_model}  # from this model's tags
                mlflow: ${globals:mlflow}             # using this MLflow API handle.
    """

    path: Path = pydantic.Field(description="Target path for saving/loading artifact file or dir.")

    overwrite: bool = pydantic.Field(default=True, description="Overwrite :attr:`path` if exists.")

    skip_reqs_check: bool = pydantic.Field(
        default=False,
        description="See :paramref:`~mlopus.artschema.load_artifact.skip_reqs_check`",
    )

    subject: ExpSubject | RunSubject | ModelSubject | None = pydantic.Field(
        default=None,
        exclude=True,
        description=(
            "If :attr:`schema_` is an alias to a previously registered artifact schema, load the respective "
            "schema class from this subject's tags. See also :meth:`~mlopus.artschema.Tags.parse_subject`."
        ),
    )

    schema_: str | None | Schema | Type[Schema] = pydantic.Field(
        exclude=True,
        alias="schema",
        description="See :paramref:`~mlopus.artschema.load_artifact.schema`",
    )

    dumper: dict | None = pydantic.Field(
        default=None,
        description="See :paramref:`~mlopus.artschema.Schema.get_dumper.dumper`",
    )

    loader: dict | None = pydantic.Field(
        default=None,
        description="See :paramref:`~mlopus.artschema.Schema.get_loader.loader`",
    )

    schema_info: SchemaInfo | None = None

    _parse_subject = pydantic.validator("subject", pre=True, allow_reuse=True)(_parse_subject)

    def __init__(self, **kwargs):
        kwargs.setdefault("mlflow", None)
        super().__init__(**kwargs, schema_info=None)

        self.schema_, alias = helpers.resolve_schema_and_alias(
            schema=self.schema_,
            subject=self._subject_api,
            skip_reqs_check=self.skip_reqs_check,
        )

        self.schema_info = SchemaInfo(
            alias=alias,
            cls=self.schema_.__class__,
            subject=self._subject if alias else None,  # subject is only relevant when inferring schema by alias
            reqs_checked=not self.skip_reqs_check if alias else None,  # reqs check only happens when inferring by alias
        )

    @property
    def _subject(self) -> SchemaSubject | None:
        if (subj := self.subject) is None:
            if self.run_manager is None:
                return None
            else:
                subj = RunSubject(run_id=self.run_manager.run.id)
        elif self.run_manager is None:
            raise RuntimeError("Cannot resolve subject for schema inference when `mlflow=None`")

        return subj.with_defaults(  # noqa
            run_id=self.run_manager.run.id,
            exp_name=self.run_manager.run.exp.name,
        ).using(self.run_manager.mlflow_api)

    @property
    def _subject_api(self) -> EntityApi | None:
        if (subj := self._subject) is None:
            return None

        return subj.resolve()

    def load(self) -> A:
        return self._load()

    def save(self, data: A | dict | Path) -> None:
        return self._save(data)

    def _load(self) -> A:
        return self.schema_.get_loader(self.loader)(self.path)

    def _save(self, data: A | dict | Path) -> None:
        paths.ensure_only_parents(self.path, force=self.overwrite)

        if isinstance(source := self.schema_.get_dumper(data, self.dumper), Path):
            paths.place_path(source, self.path, mode="link")
        else:
            source(self.path)

    def _describe(self) -> Dict[str, Any]:
        return self.dict()
