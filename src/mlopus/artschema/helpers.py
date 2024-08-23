"""Utility functions for working with artifact schemas."""

import logging
from pathlib import Path
from typing import TypeVar, Type, Tuple, Mapping

from mlopus.mlflow.api.common.schema import BaseEntity
from mlopus.mlflow.api.contract import RunIdentifier
from mlopus.mlflow.api.model import ModelApi
from mlopus.mlflow.api.mv import ModelVersionApi
from mlopus.mlflow.api.run import RunApi
from mlopus.utils import import_utils, typing_utils, dicts

from .framework import Dumper, Loader, Schema
from .tags import Tags, ClassSpec, DEFAULT_ALIAS

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseEntity)  # Type of entity

A = TypeVar("A", bound=object)  # Type of artifact
D = TypeVar("D", bound=Dumper)  # Type of Dumper
L = TypeVar("L", bound=Loader)  # Type of Loader


def get_schemas(subject: T) -> Tags:
    """Based on tags, find the artifact schemas for the specified experiment, run, registered model or model version."""
    return Tags.parse_subject(subject)


def get_schema(subject: T, alias: str | None = None) -> ClassSpec:
    """Based on tags, infer artifact schema for the specified experiment, run, registered model or model version."""
    return Tags.parse_subject(subject).get_schema(alias)


def log_run_artifact(
    artifact: A | dict | Path,
    run: RunApi,
    path_in_run: str | None = None,
    schema: Schema[A, D, L] | Type[Schema[A, D, L]] | str | None = None,
    dumper_conf: D | dict | None = None,
    skip_reqs_check: bool = False,
    auto_register: bool | dict = False,
    keep_the_source: bool | None = None,
    allow_duplication: bool | None = None,
    use_cache: bool | None = None,
) -> None:
    """Log run artifact using schema (either provided or inferred from tags)."""
    schema, alias = resolve_schema_and_alias(schema, run, skip_reqs_check)

    run.log_artifact(
        path_in_run=path_in_run,
        source=schema.get_dumper(artifact, dumper_conf),
        use_cache=use_cache,
        keep_the_source=keep_the_source,
        allow_duplication=allow_duplication,
    )

    if auto_register:
        register_kwargs = auto_register if isinstance(auto_register, dict) else {}
        dicts.set_if_empty(register_kwargs, "aliased_as", alias) if alias else None
        run.set_tags(Tags().using(schema.__class__, **register_kwargs))


def log_model_version(
    artifact: A | dict | Path,
    model: ModelApi,
    run: RunIdentifier,
    schema: Schema[A, D, L] | Type[Schema[A, D, L]] | str | None = None,
    dumper_conf: D | dict | None = None,
    skip_reqs_check: bool = False,
    auto_register: bool | dict = False,
    path_in_run: str | None = None,
    keep_the_source: bool | None = None,
    allow_duplication: bool | None = None,
    use_cache: bool | None = None,
    version: str | None = None,
    tags: Mapping | None = None,
) -> ModelVersionApi:
    """Log artifact as model version using schema (either provided or inferred from tags)."""
    schema, alias = resolve_schema_and_alias(schema, model, skip_reqs_check)

    mv = model.log_version(
        run,
        path_in_run=path_in_run,
        source=schema.get_dumper(artifact, dumper_conf),
        tags=tags,
        version=version,
        use_cache=use_cache,
        keep_the_source=keep_the_source,
        allow_duplication=allow_duplication,
    )

    if auto_register:
        register_kwargs = auto_register if isinstance(auto_register, dict) else {}
        dicts.set_if_empty(register_kwargs, "aliased_as", alias) if alias else None
        mv.set_tags(Tags().using(schema.__class__, **register_kwargs))

    return mv


def load_artifact(
    subject: RunApi | ModelVersionApi,
    path_in_run: str | None = None,
    schema: Schema[A, D, L] | Type[Schema[A, D, L]] | str | None = None,
    loader_conf: L | dict | None = None,
    skip_reqs_check: bool = False,
    dry_run: bool = False,
) -> A:
    """Load artifact of run or model version using schema (either provided or inferred from tags)."""
    kwargs = {}

    if isinstance(subject, RunApi):
        assert path_in_run, "`path_in_run` must be specified when loading run artifact."
        kwargs["path_in_run"] = path_in_run
    else:
        assert not path_in_run, "`path_in_run` is not compatible with `ModelVersionApi`"

    schema, _ = resolve_schema_and_alias(schema, subject, skip_reqs_check)
    return subject.load_artifact(schema.get_loader(loader_conf, dry_run=dry_run), **kwargs)


def resolve_schema_and_alias(
    schema: Schema | Type[Schema] | str | None, subject: T | None, skip_reqs_check: bool
) -> Tuple[Schema, str | None]:
    alias = None
    if isinstance(schema, str) and ":" in schema:
        schema = import_utils.find_type(schema, Schema)
    if isinstance(schema, str) or schema is None:
        assert subject, "Cannot resolve schema by alias without a subject (exp, run, model or version)."
        logger.info("Using schema '%s' for subject %s", alias := schema or DEFAULT_ALIAS, subject)
        schema = get_schema(subject, alias).load(Schema, skip_reqs_check=skip_reqs_check)
    if typing_utils.safe_issubclass(schema, Schema):
        schema = schema()
    if not isinstance(schema, Schema):
        raise TypeError(f"Cannot resolve schema from '{schema}'")
    return schema, alias
