from typing import Mapping, Dict

from kedro.io import AbstractDataset
from mlopus.utils import pydantic
from ._pipeline import _Pipeline


class _Report(pydantic.BaseModel, pydantic.ExcludeEmptyMixin):
    tags: dict = {}
    params: dict = {}
    overrides: dict = {}
    metrics: dict = {}
    config: dict = {}
    pipelines: Dict[str, _Pipeline] = {}
    datasets: dict = {}


class _State(pydantic.BaseModel):
    conf: Mapping = None
    report: _Report = _Report()
    datasets: Dict[str, AbstractDataset] = {}
