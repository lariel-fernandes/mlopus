"""Kedro Pipelines."""
from typing import Mapping, Dict

from kedro.pipeline import Pipeline
from mlopus.kedro import pipeline_factory, PipelineFactory

from mlopus_kedro_example import nodes


def register_pipelines() -> Dict[str, Pipeline | PipelineFactory]:
    """Create mapping of {pipeline_name -> pipeline_instance_or_factory}."""
    return {
        "build": pipeline_factory(build),
        "eval": pipeline_factory(eval_),
    }


def build(config: Mapping) -> Pipeline:
    """Create pipeline `build`."""
    return Pipeline([
        nodes.BuildVectors.default_parser(config).to_node(  # Configured from `parameters.build_vectors`
            outputs=["labels", "vectors"],
        ),
        nodes.BuildModel.default_parser(config).to_node(  # Configured from `parameters.build_model`
            inputs=["labels", "vectors"],
            outputs="model",
        ),
    ])


def eval_(config: Mapping) -> Pipeline:
    """Create pipeline `eval`."""
    return Pipeline([
        nodes.EvalModel.default_parser(config).to_node(  # Configured from `parameters.eval_model`
            inputs="model",
            outputs="distances_at_k",
        ),
    ])
