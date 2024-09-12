from pathlib import Path
from typing import List

import mlopus
from mlopus.lineage import _LineageArg
from mlopus.utils import pydantic


class PipelineOutput(mlopus.artschema.LogArtifactSpec, pydantic.EmptyStrAsMissing):
    """Specification of an artifact to be collected and published to MLFlow after a pipeline runs.

    See also:
        - :meth:`mlopus.mlflow.BaseMlflowApi.log_run_artifact`
        - :meth:`mlopus.mlflow.BaseMlflowApi.log_model_version`

    - If :attr:`schema_` is specified, it is used to verify the artifact before collecting it.
    - :attr:`subject` and :attr:`skip_reqs_check` are used if :attr:`schema_` is an alias.
    """

    path: Path = pydantic.Field(description="Path to collect the artifact file or dir from.")
    enabled: bool = pydantic.Field(default=True, description="Enable this output.")
    log_lineage: bool = pydantic.Field(
        default=True,
        description="Log lineage info in MLFlow run. See also :class:`mlopus.lineage.Lineage`.",
    )
    pipelines: List[str] | None = pydantic.Field(
        default=None,
        description="If specified, enable output for these pipelines only.",
    )

    def used_by(self, pipeline_name: str) -> None:
        """Check if this output is configured for the specified pipeline."""
        return self.pipelines is None or pipeline_name in self.pipelines

    def collect(self, default_run_id: str) -> _LineageArg:
        """Verify and publish the artifact."""
        return self.with_defaults(run_id=default_run_id, path_in_run=self.path.name)._log(artifact=self.path)[0]
