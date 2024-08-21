from datetime import datetime
from enum import Enum
from typing import Dict, Any

from mlopus.utils import pydantic, urls


class RunStatus(Enum):
    """Run status values."""

    FAILED = "FAILED"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    SCHEDULED = "SCHEDULED"


class BaseEntity(pydantic.BaseModel, pydantic.MappingMixin):
    """Base class for entity schemas."""


class Experiment(BaseEntity):
    """Type of Experiment used by MLOpus in generic MLflow-compliant APIs."""

    id: str
    name: str
    tags: Dict[str, Any] = pydantic.Field(repr=False)


class Run(BaseEntity):
    """Type of Run used by MLOpus in generic MLflow-compliant APIs."""

    id: str
    name: str
    repo: str = pydantic.Field(repr=False)
    exp: Experiment = pydantic.Field(repr=False)
    tags: Dict[str, Any] = pydantic.Field(repr=False)
    params: Dict[str, Any] = pydantic.Field(repr=False)
    metrics: Dict[str, Any] = pydantic.Field(repr=False)
    status: RunStatus | None = pydantic.Field(repr=False)
    end_time: datetime | None = pydantic.Field(repr=False)
    start_time: datetime | None = pydantic.Field(repr=False)

    @property
    def repo_url(self) -> urls.Url:
        """Artifacts repo URL."""
        return urls.parse_url(self.repo)


class Model(BaseEntity):
    """Type of registered Model used by MLOpus in generic MLflow-compliant APIs."""

    name: str
    tags: Dict[str, Any] = pydantic.Field(repr=False)


class ModelVersion(BaseEntity):
    """Type of ModelVersion used by MLOpus in generic MLflow-compliant APIs."""

    version: str
    model: Model
    run: Run = pydantic.Field(repr=False)
    path_in_run: str = pydantic.Field(repr=False)
    tags: Dict[str, Any] = pydantic.Field(repr=False)
