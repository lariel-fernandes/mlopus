"""Utility functions for working with artifact schemas."""

import logging
from pathlib import Path
from typing import Type, Tuple, Mapping

import mlopus
from mlopus.mlflow.api.contract import RunIdentifier
from mlopus.mlflow.api.model import ModelApi
from mlopus.mlflow.api.mv import ModelVersionApi
from mlopus.mlflow.api.run import RunApi
from mlopus.utils import import_utils, typing_utils, dicts
from .framework import Schema, A, D, L
from .tags import Tags, ClassSpec, DEFAULT_ALIAS

logger = logging.getLogger(__name__)

T = (
    mlopus.mlflow.schema.Experiment
    | mlopus.mlflow.schema.Run
    | mlopus.mlflow.schema.Model
    | mlopus.mlflow.schema.ModelVersion
)


def get_schemas(subject: T) -> Tags:
    """Parse artifact schema tags from :paramref:`subject`.

    See also: :meth:`mlopus.artschema.Tags.parse_subject`

    :param subject: | Experiment, run, model or model version.
    """
    return Tags.parse_subject(subject)


def get_schema(subject: T, alias: str | None = None) -> ClassSpec:
    """Get artifact schema class specification from :paramref:`subject`.

    :param subject: | Experiment, run, model or model version.
    :param alias: | Alias of a previously registered schema for this :paramref:`subject`.
                  | Defaults to `default`.
    """
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
    """Publish run artifact using schema.

    :param artifact: | See :paramref:`Schema.get_dumper.artifact`

    :param run: | Run API object.

    :param path_in_run: | See :paramref:`mlopus.mlflow.BaseMlflowApi.log_run_artifact.path_in_run`

    :param schema:
        - Type or instance of :class:`Schema`
        - Fully qualified name of a :class:`Schema` class (e.g.: `package.module:Class`)
        - Alias of a schema previously registered for this run or its parent experiment
          (see :class:`mlopus.artschema.Tags`).

    :param dumper_conf: | See :paramref:`Schema.get_dumper.dumper`

    :param skip_reqs_check: | If :paramref:`schema` is specified by alias, ignore the registered package requirement.
                            | See :meth:`mlopus.artschema.ClassSpec.load`

    :param auto_register: | After a successful :paramref:`artifact` publish, register the used :paramref:`schema` in the :paramref:`run` tags.
                          | If a non-empty `dict` is passed, it is used as keyword arguments for :meth:`Tags.using`.
                          | If the :paramref:`schema` was specified by alias, that alias is used by default.

    :param keep_the_source: | See :paramref:`~mlopus.mlflow.BaseMlflowApi.log_run_artifact.keep_the_source`
                              (the `source` in this case is a callback, unless :paramref:`artifact` is a `Path`)

    :param allow_duplication: | See :paramref:`~mlopus.mlflow.BaseMlflowApi.log_run_artifact.allow_duplication`

    :param use_cache: | See :paramref:`~mlopus.mlflow.BaseMlflowApi.log_run_artifact.use_cache`
    """
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
    path_in_run: str | None = None,
    schema: Schema[A, D, L] | Type[Schema[A, D, L]] | str | None = None,
    dumper_conf: D | dict | None = None,
    skip_reqs_check: bool = False,
    auto_register: bool | dict = False,
    keep_the_source: bool | None = None,
    allow_duplication: bool | None = None,
    use_cache: bool | None = None,
    version: str | None = None,
    tags: Mapping | None = None,
) -> ModelVersionApi:
    """Log artifact as model version using schema.

    Example:

    .. code-block:: python

        mlflow = mlopus.mlflow.get_api()

        version = mlopus.artschema.log_model_version(
            my_artifact,
            schema=MySchema,
            run=mlflow.start_run(...),
            model=mlflow.get_or_create_model(...),
            auto_register={"aliased_as": "foobar"}  # register `MySchema` as `foobar`
        )

        mlopus.artschema.load_artifact(version, schema="foobar")

    :param artifact: | See :paramref:`Schema.get_dumper.artifact`

    :param model: | Model API object.

    :param run: | Run API object.

    :param path_in_run: | See :paramref:`mlopus.mlflow.BaseMlflowApi.log_model_version.path_in_run`

    :param schema:
        - Type or instance of :class:`Schema`
        - Fully qualified name of a :class:`Schema` class (e.g.: `package.module:Class`)
        - Alias of a schema previously registered for this run or its parent experiment
          (see :class:`mlopus.artschema.Tags`).

    :param dumper_conf: | See :paramref:`Schema.get_dumper.dumper`

    :param skip_reqs_check: | If :paramref:`schema` is specified by alias, ignore the registered package requirement.
                            | See :meth:`mlopus.artschema.ClassSpec.load`

    :param auto_register: | After a successful :paramref:`artifact` publish, register the used :paramref:`schema` in the new model version tags.
                          | See also :paramref:`log_run_artifact.auto_register`

    :param keep_the_source: | See :paramref:`~mlopus.mlflow.BaseMlflowApi.log_run_artifact.keep_the_source`
                              (the `source` in this case is a callback, unless :paramref:`artifact` is a `Path`)

    :param allow_duplication: | See :paramref:`~mlopus.mlflow.BaseMlflowApi.log_run_artifact.allow_duplication`

    :param use_cache: | See :paramref:`~mlopus.mlflow.BaseMlflowApi.log_run_artifact.use_cache`

    :param version: | See :paramref:`mlopus.mlflow.BaseMlflowApi.log_model_version.version`

    :param tags: | See :paramref:`mlopus.mlflow.BaseMlflowApi.log_model_version.version`
    """
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
) -> A | Path:
    """Load artifact of run or model version using schema.

    :param subject: | Run or model version with API handle.

    :param path_in_run: | See :paramref:`mlopus.mlflow.BaseMlflowApi.load_run_artifact.path_in_run`
                        | If :paramref:`subject` is a model version, defaults to model name.

    :param schema:
        - Type or instance of :class:`Schema`
        - Fully qualified name of a :class:`Schema` class (e.g.: `package.module:Class`)
        - Alias of a schema previously registered for this run/model version or its parent experiment/model
          (see :class:`mlopus.artschema.Tags`).

    :param loader_conf: | See :paramref:`Schema.get_loader.loader`

    :param skip_reqs_check: | If :paramref:`schema` is specified by alias, ignore the registered package requirement.
                            | See :meth:`mlopus.artschema.ClassSpec.load`

    :param dry_run: | See :paramref:`~mlopus.artschema.Loader.load.dry_run`

    :return:
        - If :paramref:`dry_run` is `True`: A `Path` to the cached artifact, after being verified.
        - Otherwise: An instance of :attr:`~mlopus.artschema.Schema.Artifact`
    """
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
        try:
            assert subject, "Cannot resolve schema by alias without a subject (exp, run, model or version)."
            logger.info("Using schema '%s' for subject %s", alias := schema or DEFAULT_ALIAS, subject)
            schema = get_schema(subject, alias).load(Schema, skip_reqs_check=skip_reqs_check)
        except Exception as e:
            logger.error(
                "If you intended to specify the schema by fully qualified name, "
                "make sure to use the format `package.module:Class`"
            )
            raise e
    if typing_utils.safe_issubclass(schema, Schema):
        schema = schema()
    if not isinstance(schema, Schema):
        raise TypeError(f"Cannot resolve schema from '{schema}'")
    return schema, alias
