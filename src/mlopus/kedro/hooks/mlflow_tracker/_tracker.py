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


class _Report(pydantic.BaseModel):
    enabled: bool = True
    path: str = "kedro-session.yml"


class _MetricsMlflow(_PrefixSuffix):
    enabled: bool = True


class _Metrics(pydantic.BaseModel):
    report: bool = True
    datasets: List[str] = []
    mlflow: _MetricsMlflow = _MetricsMlflow(enabled=True)


class _ConfigMlflow(_ScopedPrefixSuffixRuleSet):
    enabled: bool = True


class _Config(_ScopedRuleSet):
    report: bool = True
    mlflow: _ConfigMlflow = _ConfigMlflow(enabled=False)


class _Overrides(_Config):
    pass


class _NodesMlflow(_PipelinesRuleSet):
    enabled: bool = True


class _Nodes(_NodeRuleSet):
    report: bool = True
    mlflow: _NodesMlflow = _NodesMlflow(enabled=False)


class _DatasetsMlflow(_PrefixSuffixRuleSet):
    enabled: bool = True


class _Datasets(_RuleSet):
    report: bool = True
    include_non_pydantic: bool = False
    mlflow: _DatasetsMlflow = _DatasetsMlflow(enabled=False)


class _TagsMlflow(_PrefixSuffix):
    enabled: bool = True

    def apply(self, data: dict) -> dict:
        return super().apply(dicts.filter_empty_leaves(data))


class _Tags(pydantic.BaseModel):
    report: bool = False
    values: dict = {}
    mlflow: _TagsMlflow = _TagsMlflow(enabled=True)


class _LogFile(pydantic.BaseModel):
    path: str
    alias: str = None
    cleanup: bool = True

    @classmethod
    def parse_obj(cls, *args, **kwargs):
        """Also accept just a `path`."""
        if not kwargs and len(args) == 1 and isinstance(path := args[0], (str, Path)):
            return cls(path=str(path))
        return super().parse_obj(*args, **kwargs)


class _Logs(pydantic.BaseModel):
    enabled: bool = True
    path: str = "logs"
    files: List[_LogFile] = []

    @pydantic.root_validator(pre=True)  # noqa
    @classmethod
    def _parse_files(cls, values: dict) -> dict:
        """Parse file paths into `_LogFile` objects."""
        values["files"] = [_LogFile.parse_obj(file) for file in values.get("files", [])]
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


class _ParamsMlflow(_PrefixSuffix):
    enabled: bool = True


class _Params(pydantic.BaseModel):
    report: bool = True
    mappings: Dict[str, List[_ParamMapping]] = {}
    mlflow: _ParamsMlflow = _ParamsMlflow(enabled=True)

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


class _TrackerSettings(pydantic.BaseModel):
    logs: _Logs = _Logs(enabled=True)
    report: _Report = _Report(enabled=True)
    tags: _Tags = _Tags(report=False)
    metrics: _Metrics = _Metrics(report=False)
    params: _Params = _Params(enable=False)
    overrides: _Overrides = _Overrides(report=False)
    config: _Config = _Config(report=False)
    nodes: _Nodes = _Nodes(report=True)
    datasets: _Datasets = _Datasets(report=True)
