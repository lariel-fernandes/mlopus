"""This module provides tools for registering inputs/outputs of runs, such as published/loaded
model versions and run artifacts.

It also allows building structured queries for searching runs by their inputs/outputs
in MongoDB query syntax using the `find_runs` method of the `BaseMlflowApi`.
"""

from typing import Dict, Set

from mlopus.mlflow.api.run import RunApi
from mlopus.utils import pydantic, dicts


class _LineageArg(pydantic.BaseModel):
    """Base class for lineage arguments."""

    key: str
    val: str | None

    def __init__(self, key: str, val: str | None):
        super().__init__(key=key, val=val)


class _ModelLineageArg(_LineageArg):
    """Argument for adding a model artifact as run input or output."""


class _RunLineageArg(_LineageArg):
    """Argument for adding a run artifact as run input or output."""

    @pydantic.validator("val")  # noqa
    @classmethod
    def _valid_path_in_run(cls, value: str) -> str:
        return value if value is None else value.strip("/")


class _LineageInfo(pydantic.BaseModel, pydantic.ExcludeEmptyMixin):
    """Mapping of run artifacts and models that have been used as either input or output."""

    class Config:
        """Pydantic class config."""

        repr_empty: bool = False

    runs: Dict[str, Set[str]] = pydantic.Field(
        default={},
        arg_type=_RunLineageArg,
        description="Mapping of `run_id` -> `[path_in_run]`",
    )

    models: Dict[str, Set[str]] = pydantic.Field(
        default={},
        arg_type=_ModelLineageArg,
        description="Mapping of `model_name` -> `[versions]`",
    )

    @property
    def runs_by_path(self) -> Dict[str, Set[str]]:
        """Reverse mapping of `path_in_run` -> `[run_ids]`"""
        lookup = {}

        for run_id, paths in self.runs.items():
            for path in paths:
                lookup.setdefault(path, set()).add(run_id)

        return lookup

    def add(self, arg: _LineageArg) -> "_LineageInfo":
        for key, field in self.model_fields.items():
            if isinstance(arg, field.json_schema_extra.get("arg_type")):
                values = getattr(self, key).setdefault(arg.key, set())
                if arg.val is not None:
                    values.add(arg.val)

        return self


class Inputs(_LineageInfo):
    """Mapping of run artifacts and models that have been used as input."""


class Outputs(_LineageInfo):
    """Mapping of run artifacts and models that have been used as output."""


class _LineageTags(pydantic.BaseModel, pydantic.ExcludeEmptyMixin):
    """Base class for lineage tags representation."""

    class Config:
        """Pydantic class config."""

        repr_empty: bool = False

    inputs: Inputs = pydantic.Field(
        default=None,
        description="Run inputs.",
    )

    outputs: Outputs = pydantic.Field(
        default=None,
        description="Run outputs.",
    )

    def with_input(self, arg: _LineageArg) -> "_LineageTags":
        """Add input model or run artifact."""
        self.inputs = (self.inputs or Inputs()).add(arg)
        return self

    def with_output(self, arg: _LineageArg) -> "_LineageTags":
        """Add output model or run artifact."""
        self.outputs = (self.outputs or Outputs()).add(arg)
        return self

    def with_input_model(self, name: str, version: str | None = None) -> "_LineageTags":
        """Add input model.

        :param name: Model name.
        :param version: Model version.
        """
        return self.with_input(_ModelLineageArg(name, version))

    def with_input_artifact(self, run_id: str, path_in_run: str | None = None) -> "_LineageTags":
        """Add input run artifact.

        :param run_id: Run ID.
        :param path_in_run: Plain relative path inside run artifacts (e.g.: `a/b/c`)
        """
        return self.with_input(_RunLineageArg(run_id, path_in_run))

    def with_output_model(self, name: str, version: str | None = None) -> "_LineageTags":
        """Add output model.

        :param name: Model name.
        :param version: Model version.
        """
        return self.with_output(_ModelLineageArg(name, version))

    def with_output_artifact(self, run_id: str, path_in_run: str | None = None) -> "_LineageTags":
        """Add output run artifact.

        :param run_id: Run ID.
        :param path_in_run: Plain relative path inside run artifacts (e.g.: `a/b/c`)
        """
        return self.with_output(_RunLineageArg(run_id, path_in_run))


class Lineage(_LineageTags):
    """Representation of an experiment run's lineage tags (i.e.: inputs and outputs).

    Example:

    .. code-block:: python

        run = mlopus.mlflow. \\
            get_api(...) \\
            get_run(...)  # or start_run(), resume_run(), etc...

        mlopus.lineage.of(run) \\
            .with_input_model(name, version) \\
            .with_output_model(name, version) \\
            .with_input_artifact(run_id, path) \\
            .with_output_artifact(run_id, path) \\
            .register()
    """

    run: RunApi = pydantic.Field(exclude=True)

    @classmethod
    def of(cls, run: RunApi) -> "Lineage":
        """Parse lineage tags from experiment run with API handle.

        :param run: Run metadata with API handle.
        """
        return cls(run=run, **run.tags.get("lineage", {}))

    def register(self) -> "Lineage":
        """Set these lineage tags on experiment run."""
        self.run.set_tags({"lineage": dicts.filter_empty_leaves(self.dict())})
        return self.of(self.run)


class Query(_LineageTags):
    """Query builder for searching runs by their inputs/outputs.

    Example:

    .. code-block:: python

        query = mlopus.lineage.Query() \\
            .with_input_model(name, version) \\
            .with_output_model(name, version) \\
            .with_input_artifact(run_id, path) \\
            .with_output_artifact(run_id, path) \\
            .render()  # Omitting the `version` or `path` in the methods above creates a wildcard (*)

        results = mlopus.mlflow \\
            .get_api() \\
            .get_exp("1") \\
            .find_runs(query)
    """

    def render(self) -> Dict[str, str | Set[str]]:
        """Render query dict in MongoDB syntax."""
        query = {}

        for i in ("inputs", "outputs"):
            for j in ("runs", "models"):
                for k, values in getattr(getattr(self, i), j).items():
                    key = ".".join(("tags", "lineage", i, j, k))

                    if (n := (len(values))) == 0:
                        val = {"$exists": True}
                    elif n == 1:
                        val = list(values)[0]
                    else:
                        raise ValueError(f"Cannot query for multiple values of {key} (ambiguous AND/OR intention)")

                    query[key] = val

        return query


def of(run: RunApi) -> Lineage:
    """Parse lineage tags from run API.

    :param run: Run metadata with API handle.
    """
    return Lineage.of(run)
