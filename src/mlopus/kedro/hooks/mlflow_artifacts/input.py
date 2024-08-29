from pathlib import Path
from typing import List

import mlopus
from mlopus.lineage import _LineageArg
from mlopus.utils import pydantic, paths


class PipelineInput(mlopus.artschema.LoadArtifactSpec, pydantic.EmptyStrAsMissing):
    """Specification of an artifact to be placed before a pipeline runs."""

    path: Path
    link: bool = True
    enabled: bool = True
    overwrite: bool = True
    log_lineage: bool = True
    pipelines: List[str] | None = None

    def used_by(self, pipeline_name: str) -> bool:
        """Check if this input is configured for the specified pipeline."""
        return self.pipelines is None or pipeline_name in self.pipelines

    def setup(self, default_run_id: str) -> _LineageArg:
        """Download, verify and place the artifact."""
        lineage_arg, cached_artifact = self.with_defaults(run_id=default_run_id)._load(dry_run=True)

        paths.place_path(
            tgt=self.path,
            src=cached_artifact,
            overwrite=self.overwrite,
            mode="link" if self.link else "copy",
        )

        if self.path.is_dir() and not self.link:
            paths.rchmod(self.path, paths.Mode.rwx)

        return lineage_arg
