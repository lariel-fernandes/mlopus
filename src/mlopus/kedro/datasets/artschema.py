from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar, Generic, Any, Type, Dict

import mlopus
from kedro.io import AbstractDataset
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
    """Looks for artifact schema alias in experiment tags (defaults to parent experiment of Kedro session run)."""

    exp_name: str

    def with_defaults(self, exp_name: str | None = None, **__) -> "ExpSubject":
        """Complement eventually missing fields in schema subject using the provided defaults."""
        self.exp_name = self.exp_name or exp_name
        return self

    def resolve(self) -> ExpApi:
        """Get entity API."""
        return self.mlflow_api.get_or_create_exp(self.exp_name)


class RunSubject(SchemaSubject[RunApi]):
    """Looks for artifact schema alias in run and parent experiment tags (defaults to Kedro session run)."""

    run_id: str

    def with_defaults(self, run_id: str | None = None, **__) -> "RunSubject":
        """Complement eventually missing fields in schema subject using the provided defaults."""
        self.run_id = self.run_id or run_id
        return self

    def resolve(self, default_run_id: str | None = None, **__) -> RunApi:
        """Get entity API."""
        return self.mlflow_api.get_run(self.run_id or default_run_id)


class ModelSubject(SchemaSubject[ModelApi | ModelVersionApi]):
    """Looks for artifact schema alias in model version and parent model tags (version is optional)."""

    model_name: str
    model_version: str | None = None

    def resolve(self, **__) -> ModelApi | ModelVersionApi:
        model = self.mlflow_api.get_model(self.model_name)
        return model.get_version(v) if (v := self.model_version) else model


@pydantic.validate_arguments()
def _parse_subject(subj: ExpSubject | RunSubject | ModelSubject | None) -> SchemaSubject | None:
    return subj


class SchemaInfo(pydantic.BaseModel, pydantic.ExcludeEmptyMixin, Generic[A, D, L]):
    """Schema information for tracking purposes."""

    cls: Type[Schema[A, D, L]]
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
    """Dump/Load artifact using inferred or explicitly specified artifact schema."""

    path: Path
    overwrite: bool = True
    skip_reqs_check: bool = pydantic.Field(False, exclude=True)
    subject: SchemaSubject = pydantic.Field(None, exclude=True)
    schema_: str | None | Schema[A, D, L] | Type[Schema[A, D, L]] = pydantic.Field(alias="schema", exclude=True)
    dumper: dict | D | None = None
    loader: dict | L | None = None
    schema_info: SchemaInfo = None

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
