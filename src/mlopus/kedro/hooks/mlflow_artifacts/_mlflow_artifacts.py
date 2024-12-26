import logging
from typing import TypeVar, Dict

from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.pipeline import Pipeline

import mlopus
from mlopus.utils import pydantic
from .input import PipelineInput
from .output import PipelineOutput
from ..hook_factory import HookWithFactory  # noqa: TID252

logger = logging.getLogger(__name__)

T = TypeVar("T")


class MlflowArtifacts(mlopus.mlflow.MlflowRunMixin, HookWithFactory):
    """Hook to set up inputs and collect outputs using MLflow artifacts.

    Find `here <https://github.com/lariel-fernandes/mlopus/tree/main/examples/2_a_kedro_project/conf/full/parameters/hooks/mlflow_artifacts.yml>`_
    a fully commented example covering all settings that can be customized in this hook.
    """

    collect_on_error: bool = pydantic.Field(
        default=False,
        description="Collect available outputs if pipeline fails.",
    )
    inputs: Dict[str, PipelineInput] = pydantic.Field(
        default_factory=dict,
        description="Configure inputs.",
    )
    outputs: Dict[str, PipelineOutput] = pydantic.Field(
        default_factory=dict,
        description="Configure outputs.",
    )

    @pydantic.model_validator(mode="before")  # noqa
    @classmethod
    def _backwards_compat_inputs_outputs(cls, values: dict):
        for key in ["inputs", "outputs"]:
            if isinstance(value := values.get(key), list):
                values[key] = {str(n): x for n, x in enumerate(value)}
        return values

    # =======================================================================================================
    # === Artifact handlers =================================================================================

    def _setup_inputs(self, pipeline_name: str):
        for name, i in self.inputs.items():
            if i.enabled and i.used_by(pipeline_name):
                logger.info("Setting up input '%s'", name)
                lineage_arg = i.using(self.run_manager.mlflow_api).setup(default_run_id=self.run_manager.run.id)
                if i.log_lineage and lineage_arg is not None:
                    mlopus.lineage.of(self.run_manager.run).with_input(lineage_arg).register()

    def _collect_outputs(self, pipeline_name: str):
        for name, o in self.outputs.items():
            if o.enabled and o.used_by(pipeline_name):
                logger.info("Collecting output '%s'", name)
                lineage_arg = o.using(self.run_manager.mlflow_api).collect(default_run_id=self.run_manager.run.id)
                if o.log_lineage:
                    mlopus.lineage.of(self.run_manager.run).with_output(lineage_arg).register()

    # =======================================================================================================
    # === Hook triggers =====================================================================================

    @hook_impl
    def before_pipeline_run(self, run_params: dict, pipeline: Pipeline, catalog: DataCatalog):  # noqa
        self._setup_inputs(run_params["pipeline_name"])

    @hook_impl
    def after_pipeline_run(self, run_params: dict, run_result: dict, pipeline: Pipeline, catalog: DataCatalog):  # noqa
        self._collect_outputs(run_params["pipeline_name"])

    @hook_impl
    def on_pipeline_error(self, error: Exception, run_params: dict, pipeline: Pipeline, catalog: DataCatalog):  # noqa
        if self.collect_on_error:
            self._collect_outputs(run_params["pipeline_name"])
