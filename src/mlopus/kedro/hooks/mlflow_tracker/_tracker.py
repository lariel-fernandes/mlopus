import logging
from pathlib import Path
from typing import List, Mapping, Dict

from mlopus.utils import pydantic, dicts
from ._rules import (
    _NodeRuleSet,
    _ScopedRuleSet,
    _ScopedPrefixSuffixRuleSet,
    _PrefixSuffix,
    _PipelinesRuleSet,
    _RuleSet,
    _PrefixSuffixRuleSet,
)

logger = logging.getLogger(__name__)


class Report(pydantic.BaseModel):
    """Configure the session report."""

    enabled: bool = pydantic.Field(default=True, description="Add session report file to MLflow run artifacts.")
    path: str = pydantic.Field(
        default="kedro-session.yml",
        description="Session report file path inside run artifacts.",
    )


class MetricsMlflow(_PrefixSuffix):
    """Configure how to log metrics in the MLflow run."""

    enabled: bool = pydantic.Field(default=True, description="Log metrics in the MLflow run.")
    prepend_dataset: bool = pydantic.Field(default=True, description="Prepend dataset name to metric keys.")

    def apply(self, data: dict) -> dict:
        if not self.prepend_dataset:
            data = dicts.deep_merge(*data.values())
        return super().apply(data)


class Metrics(pydantic.BaseModel):
    """Configure the collection of MLflow metrics."""

    report: bool = pydantic.Field(default=True, description="Include metrics in session report file.")
    datasets: List[str] = pydantic.Field(
        default_factory=list,
        description=(
            "If a node outputs to any of these datasets, collect the output data as metrics (must be a `dict`). "
            "See :attr:`mlopus.mlflow.schema.Run.metrics`"
        ),
    )
    mlflow: MetricsMlflow = pydantic.Field(
        default=MetricsMlflow(enabled=True),
        description="Configure how to log metrics in the MLflow run.",
    )


class ConfigMlflow(_ScopedPrefixSuffixRuleSet):
    """Configure how to store in MLflow run params."""

    enabled: bool = pydantic.Field(default=True, description="Store in MLflow run params.")


class Config(_ScopedRuleSet):
    """Configure how to track the Kedro config."""

    report: bool = pydantic.Field(default=True, description="Include in session report file.")
    mlflow: ConfigMlflow = pydantic.Field(
        default=ConfigMlflow(enabled=False),
        description=(
            "Configure how to store in MLflow run params. "
            "The :attr:`~ConfigMlflow.rules` and :attr:`~ConfigMlflow.scopes` defined in the :attr:`mlflow` "
            "section will further narrow down the :attr:`rules` and :attr:`scopes` already defined here."
        ),
    )


class Overrides(Config):
    """Configure how to track Kedro config overrides."""


class NodesMlflow(_PipelinesRuleSet):
    """Configure how to store in MLflow run params."""

    enabled: bool = pydantic.Field(default=True, description="Store in MLflow run params.")


class Nodes(_NodeRuleSet):
    """Configure how to track Kedro node settings."""

    report: bool = pydantic.Field(default=True, description="Include in session report file.")
    mlflow: NodesMlflow = pydantic.Field(
        default=NodesMlflow(enabled=False),
        description=(
            "Configure how to store in MLflow run params. "
            "The :attr:`~NodesMlflow.rules` defined in the :attr:`mlflow` section will further narrow down the "
            ":attr:`rules` already defined here."
        ),
    )


class DatasetsMlflow(_PrefixSuffixRuleSet):
    """Configure how to store dataset settings in MLFlow run params."""

    enabled: bool = pydantic.Field(default=True, description="Store dataset settings in MLflow run params.")


class Datasets(_RuleSet):
    """Configure how to track Kedro dataset settings.

    Important:
        - The tracking of a dataset is only triggered when a node attempts to save/load to/from it.
    """

    report: bool = pydantic.Field(default=True, description="Include in session report file.")
    include_non_pydantic: bool = pydantic.Field(
        default=False,
        description="Use the ``__dict__`` attribute to describe settings of non-pydantic datasets.",
    )
    mlflow: DatasetsMlflow = pydantic.Field(
        default=DatasetsMlflow(enabled=False),
        description="Configure how to store dataset settings in MLFlow run params.",
    )


class TagsMlflow(_PrefixSuffix):
    """Configure how extra tags are set in the MLflow run."""

    enabled: bool = pydantic.Field(default=True, description="Enable setting extra tags in MLflow run.")

    def apply(self, data: dict) -> dict:
        return super().apply(dicts.filter_empty_leaves(data))


class Tags(pydantic.BaseModel):
    """Configure extra tags."""

    report: bool = pydantic.Field(
        default=False,
        description="Include extra tags in session report file.",
    )
    values: dict = pydantic.Field(
        default_factory=dict,
        description="Extra tags dict. See :attr:`mlopus.mlflow.schema.Run.tags`",
    )
    mlflow: TagsMlflow = pydantic.Field(
        default=TagsMlflow(enabled=True),
        description="Configure how extra tags are set in the MLflow run.",
    )


class LogFile(pydantic.BaseModel):
    """Log file settings."""

    path: str = pydantic.Field(description="Path to local log file.")
    alias: str = pydantic.Field(default=None, description="File alias when uploading.")
    cleanup: bool = pydantic.Field(default=True, description="Clear file contents before the first pipeline runs.")

    @classmethod
    def parse_obj(cls, *args, **kwargs):
        """Also accept just a `path`."""
        if not kwargs and len(args) == 1 and isinstance(path := args[0], (str, Path)):
            return cls(path=str(path))
        return super().parse_obj(*args, **kwargs)


class Logs(pydantic.BaseModel):
    """Configure how to track logs."""

    enabled: bool = pydantic.Field(default=True, description="Upload logs to MLflow run.")
    path: str = pydantic.Field(default="logs", description="Path to logs dir inside run artifacts.")
    files: List[LogFile] = pydantic.Field(default_factory=list, description="Local log files to upload.")

    @pydantic.root_validator(pre=True)  # noqa
    @classmethod
    def _parse_files(cls, values: dict) -> dict:
        """Parse file paths into `_LogFile` objects."""
        values["files"] = [LogFile.parse_obj(file) for file in values.get("files", [])]
        return values


class _ParamMapping(pydantic.BaseModel):
    tgt: str
    src: str

    @classmethod
    def parse_obj(cls, *args, **kwargs):
        """Also accept a sequence of (tgt, src)."""
        if not kwargs and len(args) == 1 and isinstance(mapping := args[0], (list, tuple)):
            tgt, src = mapping
            return cls(tgt=tgt, src=src)
        return super().parse_obj(*args, **kwargs)


class ParamsMlflow(_PrefixSuffix):
    """Configure how to log collected params in the MLflow run."""

    enabled: bool = pydantic.Field(default=True, description="Log collected params in MLflow run.")


class Params(pydantic.BaseModel):
    """Configure the collection of MLflow params."""

    report: bool = pydantic.Field(default=True, description="Include collected params in session report file.")
    mappings: Dict[str, List[_ParamMapping]] = pydantic.Field(
        default_factory=dict,
        description=(
            "A mapping of target metric keys to arbitrary paths inside the Kedro configuration from where the params "
            "for those keys will be obtained. Example: ``sampling_ratio: parameters.sampler.ratio``"
        ),
    )
    mlflow: ParamsMlflow = pydantic.Field(
        default=ParamsMlflow(enabled=True),
        description="Configure how to log collected params in the MLflow run.",
    )

    @pydantic.root_validator(pre=True)  # noqa
    @classmethod
    def _parse_mappings(cls, values: dict) -> dict:
        """Parse (src, tgt) tuples into `_ParamMapping` objects."""
        values["mappings"] = {k: [_ParamMapping.parse_obj(x) for x in v] for k, v in values.get("mappings", {}).items()}
        return values

    def apply(self, pipeline_name: str, conf: Mapping) -> dict:
        params = {}

        for mapping in self.mappings.get(pipeline_name, []):
            logger.debug(f"Mapping Kedro config key to params: '{mapping.src}' -> '{mapping.tgt}'")
            dicts.set_nested(params, mapping.tgt.split("."), dicts.get_nested(conf, mapping.src.split(".")))

        return params
