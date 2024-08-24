from datetime import datetime
from enum import Enum
from typing import Dict, Any

from mlopus.utils import pydantic, urls
from . import patterns


class RunStatus(Enum):
    """Run status values."""

    FAILED = "FAILED"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    SCHEDULED = "SCHEDULED"


class BaseEntity(pydantic.BaseModel, pydantic.MappingMixin):
    """Base class for entity schemas."""

    tags: Dict[str, Any] = pydantic.Field(
        repr=False,
        description=(
            "Nested tags dict with JSON-serializable leaf-values and keys "
            f"in the pattern: `{patterns.TAG_PARAM_OR_METRIC_KEY.pattern}`"
        ),
    )


class Experiment(BaseEntity):
    """Type of Experiment used by MLOpus in generic MLflow-compliant APIs."""

    id: str = pydantic.Field(description=f"Experiment ID in the pattern `{patterns.EXP_ID.pattern}`")
    name: str = pydantic.Field(description="Experiment Name.")


class Run(BaseEntity):
    """Type of Run used by MLOpus in generic MLflow-compliant APIs."""

    id: str = pydantic.Field(description=f"Run ID in the pattern `{patterns.RUN_ID.pattern}`")
    name: str = pydantic.Field(description="Run name.")
    exp: Experiment = pydantic.Field(repr=False, description="Parent experiment.")
    repo: str = pydantic.Field(repr=False, description="Artifacts repo URL.")
    params: Dict[str, Any] = pydantic.Field(
        repr=False,
        description="Nested params dict with JSON-serializable leaf-values.",
    )
    metrics: Dict[str, Any] = pydantic.Field(
        repr=False,
        description="Nested metrics dict with float leaf-values.",
    )
    status: RunStatus | None = pydantic.Field(repr=False, description="Run status.")
    end_time: datetime | None = pydantic.Field(
        repr=False,
        description="Start time in local timezone.",
    )
    start_time: datetime | None = pydantic.Field(
        repr=False,
        description="End time in local timezone.",
    )

    @property
    def repo_url(self) -> urls.Url:
        """Artifacts repo URL."""
        return urls.parse_url(self.repo)


class Model(BaseEntity):
    """Type of registered Model used by MLOpus in generic MLflow-compliant APIs."""

    name: str = pydantic.Field(description=f"Model vame in the pattern `{patterns.MODEL_NAME.pattern}`")


class ModelVersion(BaseEntity):
    """Type of ModelVersion used by MLOpus in generic MLflow-compliant APIs."""

    version: str = pydantic.Field(description=f"Model version in the pattern `{patterns.MODEL_VERSION.pattern}`")
    model: Model = pydantic.Field(description="Parent model.")
    run: Run = pydantic.Field(repr=False, description="Parent run.")
    path_in_run: str = pydantic.Field(repr=False, description="Path inside run artifacts.")
