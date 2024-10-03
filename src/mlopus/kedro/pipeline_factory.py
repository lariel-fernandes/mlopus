from abc import abstractmethod, ABC
from typing import Callable, Mapping

from kedro.pipeline import Pipeline

from mlopus.utils import pydantic


class PipelineFactory(ABC):
    """Base class for pipeline factories."""

    @abstractmethod
    def __call__(self, config: Mapping) -> Pipeline:
        """Use config loader to build pipeline."""


class AnonymousPipelineFactory(PipelineFactory, pydantic.BaseModel):
    """Pipeline factory for arbitrary function."""

    func: Callable[[Mapping], Pipeline]

    def __call__(self, config: Mapping) -> Pipeline:
        """Use config loader to build pipeline with arbitrary function."""
        return self.func(config)


def pipeline_factory(func: Callable[[Mapping], Pipeline]) -> AnonymousPipelineFactory:
    """Shortcut to turn a function into a pipeline factory."""
    return AnonymousPipelineFactory(func=func)
