from abc import abstractmethod, ABC
from typing import Callable

from kedro.config import AbstractConfigLoader
from kedro.pipeline import Pipeline

from mlopus.utils import pydantic


class PipelineFactory(ABC):
    """Base class for pipeline factories."""

    @abstractmethod
    def __call__(self, config: AbstractConfigLoader) -> Pipeline:
        """Use config loader to build pipeline."""


class AnonymousPipelineFactory(PipelineFactory, pydantic.BaseModel):
    """Pipeline factory for arbitrary function."""

    func: Callable[[AbstractConfigLoader], Pipeline]

    def __call__(self, config: AbstractConfigLoader) -> Pipeline:
        """Use config loader to build pipeline with arbitrary function."""
        return self.func(config)


def pipeline_factory(func: Callable[[AbstractConfigLoader], Pipeline]) -> AnonymousPipelineFactory:
    """Shortcut to turn a function into a pipeline factory."""
    return AnonymousPipelineFactory(func=func)
