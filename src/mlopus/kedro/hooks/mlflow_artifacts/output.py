from pathlib import Path
from typing import List

import mlopus
from mlopus.lineage import _LineageArg
from mlopus.utils import pydantic


class PipelineOutput(mlopus.artschema.LogArtifactSpec, pydantic.EmptyStrAsMissing):
    """Specification of an artifact to be collected after a pipeline runs."""

    path: Path
    enabled: bool = True
    log_lineage: bool = True
    pipelines: List[str] | None = None

    def used_by(self, pipeline_name: str) -> None:
        """Check if this output is configured for the specified pipeline."""
        return self.pipelines is None or pipeline_name in self.pipelines

    def collect(self, default_run_id: str) -> _LineageArg:
        """Verify and publish the artifact."""
        return self.with_defaults(run_id=default_run_id, path_in_run=self.path.name)._log(artifact=self.path)[0]
